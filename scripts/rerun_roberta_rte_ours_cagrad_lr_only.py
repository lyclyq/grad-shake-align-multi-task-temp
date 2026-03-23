#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/home/lyclyq/miniconda3/envs/optimization/bin/python")


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _run(cmd: List[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(ROOT / ".hf_cache"),
            "HF_DATASETS_CACHE": str(ROOT / ".hf_cache" / "datasets"),
            "TRANSFORMERS_CACHE": str(ROOT / ".hf_cache" / "transformers"),
        }
    )
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _read_eval_curve(metrics_csv: Path, key: str = "val/acc") -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if key not in df.columns or "step" not in df.columns:
        raise KeyError(f"missing columns in {metrics_csv}: step/{key}")
    vals = pd.to_numeric(df[key], errors="coerce")
    steps = pd.to_numeric(df["step"], errors="coerce")
    mask = vals.notna() & steps.notna()
    out = pd.DataFrame({"step": steps[mask].astype(int), "value": vals[mask].astype(float)})
    out = out.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    if out.empty:
        raise ValueError(f"no eval curve in {metrics_csv}")
    return out


def _score_from_metrics(metrics_csv: Path) -> Dict[str, float]:
    curve = _read_eval_curve(metrics_csv, "val/acc")
    vals = curve["value"].to_numpy(dtype=float)
    val_max = float(vals.max())
    val_final = float(vals[-1])
    val_avg = float(vals.mean())
    score = 0.5 * val_max + 0.4 * val_final + 0.1 * val_avg
    return {
        "score": score,
        "val_max": val_max,
        "val_final": val_final,
        "val_avg": val_avg,
    }


def _copy_metrics_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _pick_best_seed(run_dir: Path, seeds: List[int]) -> Tuple[int, Dict[str, float], Path]:
    rows: List[Tuple[int, Dict[str, float], Path]] = []
    for seed in seeds:
        metrics_csv = run_dir / f"s{seed}" / "metrics.csv"
        if not metrics_csv.exists():
            continue
        stats = _score_from_metrics(metrics_csv)
        rows.append((seed, stats, metrics_csv))
    if not rows:
        raise FileNotFoundError(f"no metrics found under {run_dir}")
    rows.sort(key=lambda x: (x[1]["score"], x[1]["val_final"], x[1]["val_max"]), reverse=True)
    return rows[0]


def _derive_dirs(src_final: Path, tag: str) -> Tuple[Path, Path]:
    parent = src_final.parent
    src_name = src_final.name
    hpo_dir = parent / f"hpo__ours-cagrad-lr20__from_{src_name}__{tag}__{_now_ts()}"
    final_dir = parent / f"final__ours-cagrad-lr20-compare__from_{src_name}__{tag}__{_now_ts()}"
    return hpo_dir, final_dir


def _infer_src_hpo_dir(src_final: Path) -> Path:
    final_prov = _load_json(src_final / "final_provenance.json")
    best_path = Path(str(final_prov["best_path"])).resolve()
    return best_path.parent


def _source_cfg(src_final: Path, source_method_dir: str) -> dict:
    cfg_path = src_final / "trial_runs" / source_method_dir / "s2" / "config_resolved.json"
    return _load_json(cfg_path)


def _method_specs(src_final: Path, src_hpo_dir: Path) -> Tuple[dict, List[Dict[str, object]]]:
    hpo_prov = _load_json(src_hpo_dir / "config_provenance.json")
    best_hparams = _load_json(src_hpo_dir / "best_hparams.json")
    ours_final_cfg = _source_cfg(src_final, "ours")
    cagrad_r_final_cfg = _source_cfg(src_final, "baseline_cagrad_r")
    cagrad_R_final_cfg = _source_cfg(src_final, "baseline_cagrad_R")
    ours_hpo_cfg = json.loads(best_hparams["best"]["trial_cfg_json"])
    cagrad_hpo_cfgs = dict(best_hparams["baseline_cagrad_best_cfg_by_tag"])

    return hpo_prov, [
        {
            "name": "ours",
            "train_method_name": "ours",
            "hpo_cfg": ours_hpo_cfg,
            "final_cfg": ours_final_cfg,
            "extra_overrides": {},
            "drop_keys_extra": [],
        },
        {
            "name": "baseline_cagrad_r",
            "train_method_name": "baseline_r",
            "hpo_cfg": dict(cagrad_hpo_cfgs["baseline_r"]),
            "final_cfg": cagrad_r_final_cfg,
            "extra_overrides": {
                "method.baseline_r.grad_solver": "cagrad",
                "method.baseline_r.cagrad.c": repr(float(cagrad_hpo_cfgs["baseline_r"]["method"]["baseline_r"]["cagrad"]["c"])),
                "method.baseline_r.lora.r": str(int(cagrad_hpo_cfgs["baseline_r"]["method"]["baseline_r"]["lora"]["r"])),
            },
            "drop_keys_extra": ["method.baseline_r.cagrad.c"],
        },
        {
            "name": "baseline_cagrad_R",
            "train_method_name": "baseline_R",
            "hpo_cfg": dict(cagrad_hpo_cfgs["baseline_R"]),
            "final_cfg": cagrad_R_final_cfg,
            "extra_overrides": {
                "method.baseline_R.grad_solver": "cagrad",
                "method.baseline_R.cagrad.c": repr(float(cagrad_hpo_cfgs["baseline_R"]["method"]["baseline_R"]["cagrad"]["c"])),
                "method.baseline_R.lora.r": str(int(cagrad_hpo_cfgs["baseline_R"]["method"]["baseline_R"]["lora"]["r"])),
            },
            "drop_keys_extra": ["method.baseline_R.cagrad.c"],
        },
    ]


def _run_lr_sweep_for_method(
    *,
    hpo_dir: Path,
    hpo_prov: Dict[str, object],
    spec: Dict[str, object],
    lrs: List[float],
    hpo_seeds: List[int],
    hpo_max_steps: int,
) -> Dict[str, object]:
    cfg = dict(spec["hpo_cfg"])
    set_args = list(hpo_prov["cli_set_args"])
    base_config_path = str(hpo_prov["base_config_path"])
    schedule_path = str(hpo_prov.get("schedule_path"))
    train_cfg = cfg["train"]
    batch_size = int(train_cfg["batch_size"])
    warmup_ratio = float(train_cfg["warmup_ratio"])
    train_epochs = 1

    method_name = str(spec["name"])
    train_method_name = str(spec["train_method_name"])
    method_hpo_dir = hpo_dir / method_name
    method_hpo_dir.mkdir(parents=True, exist_ok=False)
    trial_rows: List[Dict[str, object]] = []

    drop_keys = {
        "io.run_dir",
        "io.overwrite",
        "method.name",
        "method.baseline_r.lora.r",
        "method.baseline_r.grad_solver",
        "method.baseline_R.lora.r",
        "method.baseline_R.grad_solver",
        "train.seed",
        "train.lr",
        "train.warmup_ratio",
        "train.epochs",
        "train.max_steps",
        "train.batch_size",
        "trial_tag",
        *list(spec.get("drop_keys_extra", [])),
    }
    common_overrides = {
        "method.name": train_method_name,
        "train.warmup_ratio": repr(warmup_ratio),
        "train.batch_size": str(batch_size),
        "train.epochs": str(train_epochs),
        "train.max_steps": str(hpo_max_steps),
        **dict(spec.get("extra_overrides", {})),
    }

    for idx, lr in enumerate(lrs):
        seed_scores = []
        for seed in hpo_seeds:
            run_dir = method_hpo_dir / "trial_runs" / method_name / f"lr_{idx:02d}" / f"s{seed}"
            overrides = dict(common_overrides)
            overrides.update(
                {
                    "train.lr": repr(float(lr)),
                    "train.seed": str(seed),
                    "io.run_dir": str(run_dir),
                    "io.overwrite": "force",
                }
            )
            cmd = [str(PYTHON_BIN), "scripts/run.py", "train", "--config", base_config_path]
            if schedule_path and schedule_path != "None":
                cmd.extend(["--schedule", schedule_path])
            for item in _filter_set_args(set_args, overrides=overrides, drops=drop_keys):
                cmd.extend(["--set", item])
            _run(cmd, cwd=ROOT)
            stats = _score_from_metrics(run_dir / "metrics.csv")
            seed_scores.append((seed, stats, run_dir / "metrics.csv"))

        mean_score = float(np.mean([row[1]["score"] for row in seed_scores]))
        mean_val_max = float(np.mean([row[1]["val_max"] for row in seed_scores]))
        mean_val_final = float(np.mean([row[1]["val_final"] for row in seed_scores]))
        mean_val_avg = float(np.mean([row[1]["val_avg"] for row in seed_scores]))
        trial_rows.append(
            {
                "method": method_name,
                "trial_idx": idx,
                "lr": float(lr),
                "score": mean_score,
                "val_max": mean_val_max,
                "val_final": mean_val_final,
                "val_avg": mean_val_avg,
                "seed_metrics": [
                    {
                        "seed": seed,
                        **stats,
                        "metrics_csv": str(metrics_csv),
                    }
                    for seed, stats, metrics_csv in seed_scores
                ],
            }
        )

    trials_df = pd.DataFrame([{k: v for k, v in row.items() if k != "seed_metrics"} for row in trial_rows]).sort_values(
        "score", ascending=False
    )
    trials_df.to_csv(method_hpo_dir / "lr_trials.csv", index=False)
    best = max(trial_rows, key=lambda row: (float(row["score"]), float(row["val_final"]), float(row["val_max"])))
    (method_hpo_dir / "best_lr.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def _run_final_method(
    *,
    dst_final: Path,
    hpo_prov: Dict[str, object],
    spec: Dict[str, object],
    best_lr: float,
    final_seeds: List[int],
) -> None:
    cfg = dict(spec["final_cfg"])
    set_args = list(hpo_prov["cli_set_args"])
    base_config_path = str(hpo_prov["base_config_path"])
    schedule_path = str(hpo_prov.get("schedule_path"))

    train_cfg = cfg["train"]
    batch_size = int(train_cfg["batch_size"])
    warmup_ratio = float(train_cfg["warmup_ratio"])
    train_epochs = int(train_cfg["epochs"])
    method_name = str(spec["name"])
    train_method_name = str(spec["train_method_name"])

    drop_keys = {
        "io.run_dir",
        "io.overwrite",
        "method.name",
        "method.baseline_r.lora.r",
        "method.baseline_r.grad_solver",
        "method.baseline_R.lora.r",
        "method.baseline_R.grad_solver",
        "train.seed",
        "train.lr",
        "train.warmup_ratio",
        "train.epochs",
        "train.max_steps",
        "train.batch_size",
        *list(spec.get("drop_keys_extra", [])),
    }
    common_overrides = {
        "method.name": train_method_name,
        "train.warmup_ratio": repr(warmup_ratio),
        "train.batch_size": str(batch_size),
        "train.epochs": str(train_epochs),
        "train.lr": repr(float(best_lr)),
        **dict(spec.get("extra_overrides", {})),
    }

    for seed in final_seeds:
        run_dir = dst_final / "trial_runs" / method_name / f"s{seed}"
        overrides = dict(common_overrides)
        overrides.update(
            {
                "train.seed": str(seed),
                "io.run_dir": str(run_dir),
                "io.overwrite": "force",
            }
        )
        cmd = [str(PYTHON_BIN), "scripts/run.py", "train", "--config", base_config_path]
        if schedule_path and schedule_path != "None":
            cmd.extend(["--schedule", schedule_path])
        for item in _filter_set_args(set_args, overrides=overrides, drops=drop_keys):
            cmd.extend(["--set", item])
        _run(cmd, cwd=ROOT)

    all_curves_dir = dst_final / "all_curves"
    best_curves_dir = dst_final / "best_curves"
    for path in all_curves_dir.glob(f"{method_name}_s*.csv"):
        path.unlink()
    for seed in final_seeds:
        src = dst_final / "trial_runs" / method_name / f"s{seed}" / "metrics.csv"
        _copy_metrics_csv(src, all_curves_dir / f"{method_name}_s{seed}.csv")
    best_seed, _, best_metrics = _pick_best_seed(dst_final / "trial_runs" / method_name, final_seeds)
    print(f"[INFO] best final seed for {method_name}: s{best_seed}", flush=True)
    _copy_metrics_csv(best_metrics, best_curves_dir / f"best_{method_name}.csv")


def _write_provenance(
    *,
    dst_final: Path,
    src_final: Path,
    hpo_dir: Path,
    best_rows: Dict[str, Dict[str, object]],
    final_seeds: List[int],
    hpo_seeds: List[int],
    hpo_max_steps: int,
    lr_count: int,
) -> None:
    payload = {
        "src_final": str(src_final),
        "dst_final": str(dst_final),
        "hpo_dir": str(hpo_dir),
        "hpo_seeds": hpo_seeds,
        "hpo_max_steps": hpo_max_steps,
        "lr_trials_per_method": lr_count,
        "final_seeds": final_seeds,
        "best_rows": best_rows,
        "note": "Only ours / baseline_cagrad_r / baseline_cagrad_R were rerun; baseline avg artifacts were copied from src_final.",
    }
    (dst_final / "ours_cagrad_lr20_rerun_provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rerun_methods(src_final: Path, *, lr_trials: int, hpo_max_steps: int, hpo_seeds: List[int], tag: str) -> Path:
    if not src_final.exists():
        raise FileNotFoundError(f"src_final not found: {src_final}")
    final_seeds = [2, 3, 5, 7, 11]
    hpo_dir, dst_final = _derive_dirs(src_final, tag)

    print(f"[INFO] hpo_dir={hpo_dir}")
    print(f"[INFO] dst_final={dst_final}")
    lrs = np.geomspace(2e-6, 1e-3, num=lr_trials).tolist()
    src_hpo_dir = _infer_src_hpo_dir(src_final)
    print(f"[INFO] src_hpo_dir={src_hpo_dir}")
    hpo_prov, specs = _method_specs(src_final, src_hpo_dir)

    best_rows: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        method_name = str(spec["name"])
        print(f"[INFO] rerun hpo for {method_name}")
        best_rows[method_name] = _run_lr_sweep_for_method(
            hpo_dir=hpo_dir,
            hpo_prov=hpo_prov,
            spec=spec,
            lrs=lrs,
            hpo_seeds=hpo_seeds,
            hpo_max_steps=hpo_max_steps,
        )
        print(f"[OK] best {method_name}: lr={best_rows[method_name]['lr']} score={best_rows[method_name]['score']}", flush=True)

    shutil.copytree(src_final, dst_final, dirs_exist_ok=False)
    for spec in specs:
        method_name = str(spec["name"])
        _run_final_method(
            dst_final=dst_final,
            hpo_prov=hpo_prov,
            spec=spec,
            best_lr=float(best_rows[method_name]["lr"]),
            final_seeds=final_seeds,
        )

    _write_provenance(
        dst_final=dst_final,
        src_final=src_final,
        hpo_dir=hpo_dir,
        best_rows=best_rows,
        final_seeds=final_seeds,
        hpo_seeds=hpo_seeds,
        hpo_max_steps=hpo_max_steps,
        lr_count=lr_trials,
    )

    _run([str(PYTHON_BIN), "scripts/run.py", "plot", "--config", "configs/base.yaml", "--runs_dir", str(dst_final)], cwd=ROOT)
    _run([str(PYTHON_BIN), "scripts/plot_final_4lines_abs.py", str(dst_final / "trial_runs"), "val/acc"], cwd=ROOT)
    _run([str(PYTHON_BIN), "scripts/plot_mechanism_diagnostics.py", str(dst_final / "trial_runs")], cwd=ROOT)

    print(f"[OK] compare final ready: {dst_final}")
    return dst_final


def main() -> int:
    ap = argparse.ArgumentParser(description="Rerun roberta-rte ours/cagrad lr-only HPO and refresh final compare plots.")
    ap.add_argument("--src-final", required=True, help="Existing final dir to copy baseline avg runs from.")
    ap.add_argument("--lr-trials", type=int, default=20)
    ap.add_argument("--hpo-max-steps", type=int, default=200)
    ap.add_argument("--hpo-seeds", default="2,3")
    ap.add_argument("--tag", default="ours_cagrad_lr20")
    args = ap.parse_args()

    src_final = Path(args.src_final).expanduser().resolve()
    hpo_seeds = [int(x.strip()) for x in args.hpo_seeds.split(",") if x.strip()]
    rerun_methods(
        src_final,
        lr_trials=int(args.lr_trials),
        hpo_max_steps=int(args.hpo_max_steps),
        hpo_seeds=hpo_seeds,
        tag=str(args.tag),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
