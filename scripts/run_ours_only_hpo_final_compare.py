#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config_with_cli_overrides, validate_config
from src.final import TRIALS_HEADER as FINAL_TRIALS_HEADER
from src.final import run_final
from src.hpo import (
    _assert_capacity_unified,
    _atomic_write_json,
    _baseline_lora_from_method,
    _copy_metrics_csv,
    _run_bayes_after_refine,
    _run_coordinate_descent,
    _run_local_refine,
)


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False), encoding="utf-8")


def _ensure_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _filter_set_args(set_args: Iterable[str], *, overrides: Dict[str, str], drops: Iterable[str]) -> List[str]:
    drop_keys = set(str(x) for x in drops)
    kept: Dict[str, str] = {}
    order: List[str] = []
    for raw in set_args:
        s = str(raw).strip()
        if not s or "=" not in s:
            continue
        key, _ = s.split("=", 1)
        if key in drop_keys:
            continue
        if key not in kept:
            order.append(key)
        kept[key] = s
    for key, value in overrides.items():
        kept[key] = f"{key}={value}"
        if key not in order:
            order.append(key)
    return [kept[k] for k in order]


def _copytree_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob(pattern):
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)


def _append_final_rows(dst_csv: Path, rows: List[Dict[str, Any]]) -> None:
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not dst_csv.exists()) or (dst_csv.stat().st_size == 0)
    with dst_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_TRIALS_HEADER)
        if need_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FINAL_TRIALS_HEADER})


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _source_best_path(src_final_dir: Path, src_hpo_dir: Optional[Path]) -> Path:
    if src_hpo_dir is not None:
        return _ensure_exists(src_hpo_dir / "best_hparams.json", "source best_hparams")
    prov = _read_json(_ensure_exists(src_final_dir / "final_provenance.json", "source final_provenance.json"))
    best_path = prov.get("best_path", None)
    if not best_path:
        raise RuntimeError("source final_provenance.json does not contain best_path")
    return _ensure_exists(Path(str(best_path)), "source best_hparams")


def _derive_new_hpo_dir(runs_parent: Path, src_hpo_dir: Path, *, trials: int, hpo_steps: int, tag: str) -> Path:
    return runs_parent / f"hpo__ours-only__from_{src_hpo_dir.name}__t{trials}__ms{hpo_steps}__{tag}__{_now_ts()}"


def _derive_new_final_dir(runs_parent: Path, src_final_dir: Path, *, hpo_steps: int, tag: str) -> Path:
    return runs_parent / f"final__ours-only-compare__from_{src_final_dir.name}__ms{hpo_steps}__{tag}__{_now_ts()}"


def _write_hpo_provenance(
    *,
    hpo_run_dir: Path,
    cfg: Dict[str, Any],
    base_config_path: str,
    schedule_path: Optional[str],
    set_args: List[str],
) -> None:
    used_dir = hpo_run_dir / "configs_used"
    used_dir.mkdir(parents=True, exist_ok=True)
    bp = ROOT / str(base_config_path) if not Path(base_config_path).is_absolute() else Path(base_config_path)
    if bp.exists() and bp.is_file():
        shutil.copy2(bp, used_dir / bp.name)
    if schedule_path:
        sp = ROOT / str(schedule_path) if not Path(schedule_path).is_absolute() else Path(schedule_path)
        if sp.exists() and sp.is_file():
            shutil.copy2(sp, used_dir / sp.name)
    _write_json(
        hpo_run_dir / "config_provenance.json",
        {
            "base_config_path": str(base_config_path),
            "schedule_path": str(schedule_path) if schedule_path else None,
            "cli_set_args": list(set_args or []),
            "mode": "ours_only",
        },
    )
    _atomic_write_json(hpo_run_dir / "config_snapshot.json", cfg)
    (hpo_run_dir / "base_merged.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    _write_json(hpo_run_dir / "cli_overrides.json", {"set_args": list(set_args or [])})


def _run_ours_only_hpo(
    *,
    cfg: Dict[str, Any],
    base_config_path: str,
    schedule_path: Optional[str],
    set_args: List[str],
    hpo_run_dir: Path,
    old_best_obj: Dict[str, Any],
) -> Path:
    _assert_capacity_unified(cfg)
    hpo_run_dir.mkdir(parents=True, exist_ok=False)
    _write_hpo_provenance(
        hpo_run_dir=hpo_run_dir,
        cfg=cfg,
        base_config_path=base_config_path,
        schedule_path=schedule_path,
        set_args=set_args,
    )

    trials_csv = hpo_run_dir / "trials.csv"
    best_path = hpo_run_dir / "best_hparams.json"
    plan_path = hpo_run_dir / "grid_plan.json"
    status_path = hpo_run_dir / "plan_status.json"
    best_curves_dir = hpo_run_dir / "best_curves"

    hpo_cfg = cfg["hpo"]
    bandit = hpo_cfg["bandit"]
    score_cfg = bandit["score"]
    score_weights = (
        float(score_cfg["w_max"]),
        float(score_cfg["w_final"]),
        float(score_cfg["w_avg"]),
    )
    refine_seeds = [int(x) for x in bandit["refine_seeds"]][:2]
    if not refine_seeds:
        raise RuntimeError("hpo.bandit.refine_seeds is empty")

    grid_cfg = hpo_cfg["grid"]
    coord_cfg = hpo_cfg["coord"]
    coord_epochs = int(grid_cfg["baseline_search_epochs"])
    coord_max_steps = int(grid_cfg["baseline_search_max_steps"])
    refine_epochs = int(grid_cfg["rerank"]["epochs"])
    refine_max_steps = int(grid_cfg["rerank"]["max_steps"])
    bayes_trials = int(coord_cfg.get("bayes_trials", 48))
    total_trials = int(hpo_cfg["budget"]["total_trials"])
    rng = np.random.default_rng(int(grid_cfg["rng_seed"]))

    hpo_train_schedule = {
        "mode": "max_steps",
        "global_train_max_steps": int(((cfg.get("train", {}) or {}).get("max_steps", 0) or 0)),
        "coord_search": {
            "epochs": int(coord_epochs),
            "max_steps": int(coord_max_steps),
            "configured_max_steps": int(coord_max_steps),
        },
        "local_refine": {
            "epochs": int(refine_epochs),
            "max_steps": int(refine_max_steps),
            "configured_max_steps": int(refine_max_steps),
        },
        "bayes": {
            "epochs": int(refine_epochs),
            "max_steps": int(refine_max_steps),
            "trials": int(bayes_trials),
        },
    }
    _atomic_write_json(
        plan_path,
        {
            "budget": {
                "total_trials_per_family": int(total_trials),
                "shared_order": list(coord_cfg["shared_order"]),
                "method_orders": {"ours": json.loads(json.dumps(coord_cfg["method_orders"]["ours"]))},
                "top_k": int(coord_cfg["top_k"]),
                "refine_radix": int(coord_cfg["refine_radix"]),
                "bayes_trials": int(bayes_trials),
            },
            "hpo_train_schedule": hpo_train_schedule,
            "mode": "ours_only",
        },
    )
    _atomic_write_json(status_path, {"status": "running", "updated_at": int(time.time()), "mode": "ours_only"})

    ours_coord = _run_coordinate_descent(
        cfg=cfg,
        base_config_path=base_config_path,
        schedule_path=schedule_path,
        set_args=set_args,
        hpo_run_dir=hpo_run_dir,
        trials_csv=trials_csv,
        score_weights=score_weights,
        rng=rng,
        family="ours",
        tag=None,
        budget_trials=total_trials,
        seeds=refine_seeds,
        stage_epochs=coord_epochs,
        stage_max_steps=coord_max_steps,
    )
    ours_refine = _run_local_refine(
        cfg=cfg,
        base_config_path=base_config_path,
        schedule_path=schedule_path,
        set_args=set_args,
        hpo_run_dir=hpo_run_dir,
        trials_csv=trials_csv,
        score_weights=score_weights,
        family="ours",
        tag=None,
        coord_result=ours_coord,
        seeds=refine_seeds,
        stage_epochs=refine_epochs,
        stage_max_steps=refine_max_steps,
        top_k=int(coord_cfg["top_k"]),
        radix=int(coord_cfg["refine_radix"]),
    )
    ours_bayes = _run_bayes_after_refine(
        cfg=cfg,
        base_config_path=base_config_path,
        schedule_path=schedule_path,
        set_args=set_args,
        hpo_run_dir=hpo_run_dir,
        trials_csv=trials_csv,
        score_weights=score_weights,
        family="ours",
        tag=None,
        coord_result=ours_coord,
        refine_result=ours_refine,
        seeds=refine_seeds,
        stage_epochs=refine_epochs,
        stage_max_steps=refine_max_steps,
        trials_budget=bayes_trials,
    )

    best_overall = ours_coord["best_row"]
    best_source = "coord_search"
    if float(ours_refine["best_row"].get("score", "-inf")) >= float(best_overall.get("score", "-inf")):
        best_overall = ours_refine["best_row"]
        best_source = "local_refine"
    if ours_bayes.get("enabled", False) and float(ours_bayes["best_row"].get("score", "-inf")) >= float(best_overall.get("score", "-inf")):
        best_overall = ours_bayes["best_row"]
        best_source = "bayes"

    best_curves_dir.mkdir(parents=True, exist_ok=True)
    _copy_metrics_csv(best_overall, best_curves_dir / "best_ours.csv")

    merged_best = json.loads(json.dumps(old_best_obj))
    merged_best["best"] = best_overall
    merged_best["best_source"] = best_source
    merged_best["weights"] = {
        "w_max": float(score_cfg["w_max"]),
        "w_final": float(score_cfg["w_final"]),
        "w_avg": float(score_cfg["w_avg"]),
    }
    merged_best["plan_path"] = str(plan_path)
    merged_best["status_path"] = str(status_path)
    merged_best["hpo_dir"] = str(hpo_run_dir)
    merged_best["hpo_train_schedule"] = hpo_train_schedule
    merged_best["chosen_ours_alpha"] = float(_baseline_lora_from_method(cfg, "baseline_R")["alpha"])
    merged_best["use_bayes"] = bool(cfg["hpo"]["use_bayes"])
    merged_best.setdefault("coord_report", {})
    merged_best["coord_report"]["ours"] = {
        "points_per_param": int(ours_coord["points_per_param"]),
        "sweeps": ours_coord["sweeps"],
    }
    merged_best.setdefault("local_refine_report", {})
    merged_best["local_refine_report"]["ours"] = ours_refine
    merged_best.setdefault("bayes_report", {})
    merged_best["bayes_report"]["ours"] = ours_bayes
    _atomic_write_json(best_path, merged_best)
    _atomic_write_json(status_path, {"status": "done", "updated_at": int(time.time()), "mode": "ours_only"})
    return best_path


def _seed_compare_final_from_source(
    *,
    src_final_dir: Path,
    dst_final_dir: Path,
) -> None:
    methods = ["baseline_r", "baseline_R", "baseline_cagrad_r", "baseline_cagrad_R"]
    for method in methods:
        _copytree_if_exists(src_final_dir / "trial_runs" / method, dst_final_dir / "trial_runs" / method)
    _copy_glob(src_final_dir / "all_curves", "baseline*.csv", dst_final_dir / "all_curves")
    _copy_glob(src_final_dir / "best_curves", "best_baseline*.csv", dst_final_dir / "best_curves")

    rows = _read_csv_rows(src_final_dir / "trials.csv")
    picked = [r for r in rows if str(r.get("method", "")) in set(methods)]
    _append_final_rows(dst_final_dir / "trials.csv", picked)


def _run_plot_commands(final_dir: Path, python_bin: str) -> None:
    cmds = [
        [python_bin, "scripts/run.py", "plot", "--config", "configs/base.yaml", "--runs_dir", str(final_dir)],
        [python_bin, "scripts/plot_final_4lines_abs.py", str(final_dir / "trial_runs"), "val/acc"],
        [python_bin, "scripts/plot_mechanism_diagnostics.py", str(final_dir / "trial_runs")],
    ]
    for cmd in cmds:
        print("[CMD]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ours-only HPO/final and compare with existing baselines.")
    ap.add_argument("--src-final", required=True, help="Existing final dir with baseline/cagrad runs to compare against.")
    ap.add_argument("--src-hpo", default="", help="Existing HPO dir. If empty, infer from src-final/final_provenance.json.")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--final-schedule", default="configs/schedules/final.yaml")
    ap.add_argument("--hpo-steps", type=int, default=300)
    ap.add_argument("--trials", type=int, default=96)
    ap.add_argument("--tag", default="ours300")
    args = ap.parse_args()

    src_final_dir = _ensure_exists(Path(args.src_final).resolve(), "source final dir")
    src_hpo_dir = Path(args.src_hpo).resolve() if str(args.src_hpo).strip() else None
    if src_hpo_dir is not None:
        _ensure_exists(src_hpo_dir, "source hpo dir")

    src_best_path = _source_best_path(src_final_dir, src_hpo_dir)
    src_hpo_dir = src_best_path.parent
    old_best_obj = _read_json(src_best_path)
    hpo_prov = _read_json(_ensure_exists(src_hpo_dir / "config_provenance.json", "source hpo config_provenance.json"))

    base_config_path = str(hpo_prov.get("base_config_path") or args.base_config)
    hpo_schedule_path = hpo_prov.get("schedule_path", None)
    src_set_args = list(hpo_prov.get("cli_set_args", []) or [])

    runs_parent = src_final_dir.parent
    new_hpo_dir = _derive_new_hpo_dir(runs_parent, src_hpo_dir, trials=args.trials, hpo_steps=args.hpo_steps, tag=args.tag)
    new_final_dir = _derive_new_final_dir(runs_parent, src_final_dir, hpo_steps=args.hpo_steps, tag=args.tag)
    if new_hpo_dir.exists():
        raise FileExistsError(f"new hpo dir already exists: {new_hpo_dir}")
    if new_final_dir.exists():
        raise FileExistsError(f"new final dir already exists: {new_final_dir}")

    hpo_set_args = _filter_set_args(
        src_set_args,
        overrides={
            "hpo.budget.total_trials": str(args.trials),
            "hpo.grid.baseline_search_max_steps": str(args.hpo_steps),
            "hpo.grid.grid_max_steps": str(args.hpo_steps),
            "hpo.grid.rerank.max_steps": str(args.hpo_steps),
            "io.run_dir": str(new_hpo_dir),
            "io.overwrite": "resume",
        },
        drops={
            "io.run_dir",
            "io.overwrite",
            "hpo.budget.total_trials",
            "hpo.grid.baseline_search_max_steps",
            "hpo.grid.grid_max_steps",
            "hpo.grid.rerank.max_steps",
        },
    )

    cfg_hpo = load_config_with_cli_overrides(
        config_path=base_config_path,
        schedule_path=hpo_schedule_path,
        set_args=hpo_set_args,
    )
    cfg_hpo["stage"] = "hpo"
    validate_config(cfg_hpo, cmd="hpo")

    print(f"[INFO] source best: {src_best_path}")
    print(f"[INFO] new hpo dir: {new_hpo_dir}")
    print(f"[INFO] new final dir: {new_final_dir}")
    merged_best_path = _run_ours_only_hpo(
        cfg=cfg_hpo,
        base_config_path=base_config_path,
        schedule_path=hpo_schedule_path,
        set_args=hpo_set_args,
        hpo_run_dir=new_hpo_dir,
        old_best_obj=old_best_obj,
    )

    new_final_dir.mkdir(parents=True, exist_ok=False)
    _seed_compare_final_from_source(src_final_dir=src_final_dir, dst_final_dir=new_final_dir)

    final_set_args = _filter_set_args(
        src_set_args,
        overrides={
            "io.run_dir": str(new_final_dir),
            "io.overwrite": "resume",
        },
        drops={"io.run_dir", "io.overwrite"},
    )
    cfg_final = load_config_with_cli_overrides(
        config_path=base_config_path,
        schedule_path=args.final_schedule,
        set_args=final_set_args,
    )
    cfg_final["stage"] = "final"
    validate_config(cfg_final, cmd="final")

    print(f"[INFO] merged best path: {merged_best_path}")
    run_final(
        cfg_final,
        base_config_path=base_config_path,
        schedule_path=args.final_schedule,
        set_args=final_set_args,
        best_path=str(merged_best_path),
    )

    _copy_glob(src_final_dir / "all_curves", "baseline*.csv", new_final_dir / "all_curves")
    _copy_glob(src_final_dir / "best_curves", "best_baseline*.csv", new_final_dir / "best_curves")
    _run_plot_commands(new_final_dir, args.python_bin)

    _write_json(
        new_final_dir / "ours_only_compare_provenance.json",
        {
            "src_final_dir": str(src_final_dir),
            "src_hpo_dir": str(src_hpo_dir),
            "src_best_path": str(src_best_path),
            "new_hpo_dir": str(new_hpo_dir),
            "new_final_dir": str(new_final_dir),
            "merged_best_path": str(merged_best_path),
            "hpo_steps": int(args.hpo_steps),
            "trials": int(args.trials),
            "tag": str(args.tag),
        },
    )
    print(f"[OK] compare final ready: {new_final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
