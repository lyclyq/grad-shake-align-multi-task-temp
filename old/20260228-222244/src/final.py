# /home/lyclyq/Optimization/grad-shake-align/src/final.py
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .artifacts import dump_json


TRIALS_HEADER = [
    "method",
    "seed",
    "trial_tag",
    "run_dir",
    "metrics_csv",
    "val_max",
    "val_final",
    "val_avg",
    "train_max",
    "train_final",
    "score",
    "trial_cfg_json",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIALS_HEADER)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRIALS_HEADER})


def _read_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def _normalize_ours_cfg_from_best(best_trial_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compatible normalization for legacy best_hparams formats:
      - old key: method.ours.noise_gate.{tau_n,kappa_n} -> gate0_noise.{tau,kappa}
      - missing method.ours.lora -> fill from current cfg.method.ours.lora
      - partial lora block -> fill missing keys from cfg defaults
    """
    try:
        ours_cfg = json.loads(json.dumps(best_trial_cfg["method"]["ours"]))
    except Exception as e:
        raise RuntimeError("[final] best_hparams best.trial_cfg_json missing method.ours") from e

    if not isinstance(ours_cfg, dict):
        raise RuntimeError("[final] best_hparams method.ours must be a JSON object")

    ng = ours_cfg.get("noise_gate", None)
    if isinstance(ng, dict) and "gate0_noise" not in ours_cfg:
        ours_cfg["gate0_noise"] = {
            "tau": float(ng.get("tau_n", -10.0)),
            "kappa": float(ng.get("kappa_n", 10.0)),
        }

    lora_default = json.loads(json.dumps(cfg["method"]["ours"]["lora"]))
    lora_best = ours_cfg.get("lora", None)
    if not isinstance(lora_best, dict):
        ours_cfg["lora"] = lora_default
    else:
        merged = dict(lora_default)
        merged.update(lora_best)
        ours_cfg["lora"] = merged

    return ours_cfg


def _pick_best_lr_for_baseline(rows: List[Dict[str, Any]], tag: str) -> Optional[float]:
    best_score = float("-inf")
    best_lr: Optional[float] = None
    for r in rows:
        stage = str(r.get("stage", "")).strip()
        trial_tag = str(r.get("trial_tag", "")).strip()
        stage_match = stage in {f"baseline.{tag}", f"baseline_{tag}", tag}
        search_match = stage == "baseline_search" and trial_tag.startswith(f"bl_search__{tag}__")
        if not (stage_match or search_match):
            continue

        score = _safe_float(r.get("score", None))
        if score is None:
            continue

        try:
            trial_cfg = json.loads(str(r.get("trial_cfg_json", "{}")))
            lr = _safe_float((trial_cfg.get("train", {}) or {}).get("lr", None))
        except Exception:
            lr = None
        if lr is None:
            continue

        if float(score) > best_score:
            best_score = float(score)
            best_lr = float(lr)

    return best_lr


def _derive_baseline_best_lr_refined(best_obj: Dict[str, Any], best_path: Path) -> Optional[Dict[str, float]]:
    bl = best_obj.get("baseline_best_lr_refined", None)
    if isinstance(bl, dict) and ("baseline_r" in bl) and ("baseline_R" in bl):
        try:
            return {
                "baseline_r": float(bl["baseline_r"]),
                "baseline_R": float(bl["baseline_R"]),
            }
        except Exception:
            pass

    hpo_dir_raw = best_obj.get("hpo_dir", None)
    hpo_dir = Path(str(hpo_dir_raw)) if hpo_dir_raw else best_path.parent
    trials_csv = hpo_dir / "trials.csv"
    if not trials_csv.exists():
        return None

    rows = _read_rows(trials_csv)
    lr_r = _pick_best_lr_for_baseline(rows, "baseline_r")
    lr_R = _pick_best_lr_for_baseline(rows, "baseline_R")
    if lr_r is None or lr_R is None:
        return None
    return {"baseline_r": float(lr_r), "baseline_R": float(lr_R)}


def _resolve_plan_path(best_obj: Dict[str, Any], best_path: Path) -> Optional[Path]:
    pp = best_obj.get("plan_path", None)
    if pp:
        p = Path(str(pp))
        if p.exists():
            return p

    hpo_dir_raw = best_obj.get("hpo_dir", None)
    if hpo_dir_raw:
        p2 = Path(str(hpo_dir_raw)) / "grid_plan.json"
        if p2.exists():
            return p2

    p3 = best_path.parent / "grid_plan.json"
    if p3.exists():
        return p3
    return None


def _metric_values_from_csv(metrics_csv: Path, key: str, *, max_epoch: Optional[int] = None) -> List[float]:
    p = Path(metrics_csv)
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    if not p.exists():
        return []

    out: List[float] = []
    with p.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = _safe_float(row.get(key, ""))
            if v is None:
                continue
            if max_epoch is not None:
                ep = _safe_float(row.get("epoch", ""))
                if ep is not None and ep > float(max_epoch):
                    continue
            out.append(float(v))
    return out


def _summary_stats(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n_points": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n_points": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _aggregate_ours_diagnostics(rows: List[Dict[str, Any]], *, dense_early_epochs: int) -> Dict[str, Any]:
    ours_rows = [r for r in rows if str(r.get("method", "")) == "ours"]
    metric_specs = [
        ("gate0_activation", ["train/gate0_trigger_rate", "train/gate0_triggered_blocks"]),
        ("pull_direction_to_r", ["train/pull_to_r_rate", "train/pull_to_r_blocks"]),
        ("pull_direction_to_R", ["train/pull_to_R_rate", "train/pull_to_R_blocks"]),
        ("pull_strength_alpha", ["train/alpha_pull_mean"]),
    ]

    out: Dict[str, Any] = {
        "dense_window_epochs": int(dense_early_epochs),
        "n_ours_seeds": int(len(ours_rows)),
        "metrics": {},
    }

    for metric_name, keys in metric_specs:
        merged_vals: List[float] = []
        seeds_with_data = set()
        used_key_counts: Dict[str, int] = {k: 0 for k in keys}

        for r in ours_rows:
            metrics_csv = str(r.get("metrics_csv", "") or "").strip()
            if not metrics_csv:
                continue
            seed = int(float(r.get("seed", 0)))

            picked_key = None
            picked_vals: List[float] = []
            for k in keys:
                vals = _metric_values_from_csv(
                    Path(metrics_csv),
                    k,
                    max_epoch=int(dense_early_epochs),
                )
                if vals:
                    picked_key = k
                    picked_vals = vals
                    break

            if picked_key is None:
                continue

            used_key_counts[picked_key] += 1
            seeds_with_data.add(seed)
            merged_vals.extend(picked_vals)

        metric_out: Dict[str, Any] = {
            "candidate_keys": list(keys),
            "used_key_counts": {k: int(v) for k, v in used_key_counts.items() if v > 0},
            "n_seeds_with_data": int(len(seeds_with_data)),
        }
        metric_out.update(_summary_stats(merged_vals))
        out["metrics"][metric_name] = metric_out

    return out


def _safe_name(x: str) -> str:
    s = str(x)
    for ch in ["/", "\\", " ", ":", ";", "|", "\t", "\n", "\r"]:
        s = s.replace(ch, "_")
    return s


def _final_run_dir(cfg: Dict[str, Any]) -> Path:
    """
    STRICT:
      requires cfg.io.root / cfg.io.overwrite
      if cfg.io.run_dir is set, use it
      else require cfg.final_dir (we don't auto-name in strict mode)
    """
    io = cfg["io"]
    overwrite = str(io["overwrite"])
    run_dir_cfg = io.get("run_dir", None)

    if not run_dir_cfg:
        raise RuntimeError(
            "[final] strict mode requires io.run_dir to be provided "
            "(pipeline should set --set io.run_dir=...)"
        )

    p = Path(str(run_dir_cfg))
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        if overwrite == "resume":
            return p
        if overwrite == "force":
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
            return p
        # ask
        ans = input(f"[artifacts] {p} exists. Overwrite? [y/N] ").strip().lower()
        if ans in {"y", "yes"}:
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
            return p
        return p

    p.mkdir(parents=True, exist_ok=True)
    return p


def _trial_run_dir(final_dir: Path, method: str, seed: int) -> Path:
    return final_dir / "trial_runs" / _safe_name(method) / f"s{int(seed)}"


def _flatten_sets(prefix: str, d: Dict[str, Any], out: List[str]) -> None:
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_sets(p, v, out)
        else:
            out.append(f"{p}={v}")


def _run_one(
    *,
    base_config_path: str,
    schedule_path: Optional[str],
    set_args: List[str],
    final_dir: Path,
    method_name: str,
    seed: int,
    override: Dict[str, Any],
) -> Tuple[Path, Path]:
    run_dir = _trial_run_dir(final_dir, method=method_name, seed=seed)

    override = dict(override or {})
    io_over = dict(override.get("io", {}) or {})
    io_over["run_dir"] = str(run_dir)
    io_over["overwrite"] = "resume"
    override["io"] = io_over

    cmd = ["python", "scripts/run.py", "train", "--config", base_config_path]
    if schedule_path:
        cmd += ["--schedule", schedule_path]

    sets: List[str] = []
    sets.extend(list(set_args or []))

    trial_sets: List[str] = []
    _flatten_sets("", override, trial_sets)
    sets.extend(trial_sets)

    for s in sets:
        cmd += ["--set", s]

    trial_tag = f"final_{method_name}_s{int(seed)}"
    cmd += ["--trial_tag", trial_tag]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())

    metrics_csv = run_dir / "metrics.csv"
    return run_dir, metrics_csv


def _salvage_if_finished(run_dir: Path) -> Optional[Dict[str, Any]]:
    sp = run_dir / "summary.json"
    cp = run_dir / "config_resolved.json"
    if not sp.exists() or not cp.exists():
        return None
    summary = _read_json(sp)
    cfg = _read_json(cp)
    if not isinstance(summary, dict) or not isinstance(cfg, dict):
        return None
    return {"summary": summary, "cfg": cfg}


def _score_from_summary(summary: Dict[str, Any], weights: Tuple[float, float, float]) -> float:
    w_max, w_final, w_avg = weights
    val_max = float(summary["val_max"])
    val_final = float(summary["val_final"])
    val_avg = float(summary["val_avg"])
    return w_max * val_max + w_final * val_final + w_avg * val_avg


def _aggregate_method(rows: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    rs = [r for r in rows if str(r.get("method", "")) == method]

    def _f(key: str) -> List[float]:
        out = []
        for r in rs:
            out.append(float(r[key]))
        return out

    keys = ["val_max", "val_final", "val_avg", "train_max", "train_final", "score"]
    agg: Dict[str, Any] = {"method": method, "n": len(rs)}
    for k in keys:
        vals = _f(k)
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return agg


def run_final(
    cfg: Dict[str, Any],
    base_config_path: str,
    schedule_path: Optional[str],
    set_args: List[str],
    best_path: str,
) -> None:
    # -------- strict requirements --------
    final_dir = _final_run_dir(cfg)
    used_dir = final_dir / "configs_used"
    used_dir.mkdir(parents=True, exist_ok=True)
    bp_cfg = Path(str(base_config_path))
    if bp_cfg.exists() and bp_cfg.is_file():
        shutil.copy2(bp_cfg, used_dir / bp_cfg.name)
    if schedule_path:
        sp_cfg = Path(str(schedule_path))
        if sp_cfg.exists() and sp_cfg.is_file():
            shutil.copy2(sp_cfg, used_dir / sp_cfg.name)
    (final_dir / "final_merged.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    dump_json(final_dir / "cli_overrides.json", {"set_args": list(set_args or [])})

    final_epochs = int(cfg["final"]["epochs"])
    final_seeds = [int(x) for x in cfg["final"]["seeds"]]
    if not final_seeds:
        raise RuntimeError("[final] final.seeds must be non-empty")

    bandit = cfg["hpo"]["bandit"]
    fixed_wu = float(bandit["fixed_warmup_ratio"])

    score_cfg = bandit["score"]
    w_max = float(score_cfg["w_max"])
    w_final = float(score_cfg["w_final"])
    w_avg = float(score_cfg["w_avg"])
    weights = (w_max, w_final, w_avg)

    bp = Path(best_path)
    if not bp.exists():
        raise FileNotFoundError(f"[final] best_hparams.json not found: {bp}")

    # copy best for provenance
    try:
        shutil.copy2(bp, final_dir / "best_hparams.json")
    except Exception:
        pass

    best_obj = _read_json(bp)
    if not isinstance(best_obj, dict):
        raise RuntimeError("[final] best_hparams.json must be a JSON dict")

    # best trial cfg -> ours knobs + lr + lora r/R
    best_row = best_obj["best"]
    best_trial_cfg = json.loads(best_row["trial_cfg_json"])

    ours_cfg = _normalize_ours_cfg_from_best(best_trial_cfg, cfg)
    ours_lora = ours_cfg["lora"]
    ours_lr = _safe_float((best_trial_cfg.get("train", {}) or {}).get("lr", None))
    if ours_lr is None:
        raise RuntimeError("[final] best_hparams best.trial_cfg_json missing train.lr")
    ours_r = int(ours_lora["r"])
    ours_R = int(ours_lora["R"])

    chosen_ours_alpha = best_obj.get("chosen_ours_alpha", None)
    if chosen_ours_alpha is not None:
        if float(ours_lora.get("alpha", 0.0)) != float(chosen_ours_alpha):
            raise RuntimeError(
                f"[final] chosen_ours_alpha mismatch: best_trial alpha={ours_lora.get('alpha')} "
                f"!= chosen_ours_alpha={chosen_ours_alpha}"
            )

    # plan path -> baseline best lr by variant
    plan_path_opt = _resolve_plan_path(best_obj, bp)
    if plan_path_opt is None:
        raise FileNotFoundError("[final] plan_path not found in best_hparams and fallback locations")
    plan_path = plan_path_opt

    plan = _read_json(plan_path)
    if not isinstance(plan, dict):
        raise RuntimeError("[final] grid_plan.json must be a JSON dict")

    # baseline best lr by variant (new format) or legacy fallback from hpo_dir/trials.csv
    bl = _derive_baseline_best_lr_refined(best_obj, bp)
    if not isinstance(bl, dict):
        raise RuntimeError("[final] missing baseline_best_lr_refined and failed to derive from hpo trials.csv")
    baseline_lr_r = float(bl["baseline_r"])
    baseline_lr_R = float(bl["baseline_R"])

    # ensure final uses ranks derived from ours_r/ours_R
    methods = [
        {
            "name": "baseline_r",
            "override": {
                "method": {
                    "name": "baseline_r",
                    "baseline_r": {
                        "lora": {
                            "r": int(ours_r),
                            # alpha/dropout come from base.yaml (strict); do not override.
                        }
                    },
                }
            },
            "lr": baseline_lr_r,
        },
        {
            "name": "baseline_R",
            "override": {
                "method": {
                    "name": "baseline_R",
                    "baseline_R": {
                        "lora": {
                            "r": int(ours_R),
                        }
                    },
                }
            },
            "lr": baseline_lr_R,
        },
        {
            "name": "ours",
            "override": {
                "method": {
                    "name": "ours",
                    "ours": ours_cfg,
                }
            },
            "lr": ours_lr,
        },
    ]

    trials_csv = final_dir / "trials.csv"
    summary_json = final_dir / "final_summary.json"
    best_curves_dir = final_dir / "best_curves"
    all_curves_dir = final_dir / "all_curves"

    # resume awareness via existing rows
    rows_existing = _read_rows(trials_csv)
    done_pairs = {(str(r["method"]), int(r["seed"])) for r in rows_existing}

    repro_runs: List[Dict[str, Any]] = []

    for m in methods:
        mname = str(m["name"])
        base_override = dict(m["override"])
        lr = float(m["lr"])

        for sd in final_seeds:
            key = (mname, int(sd))
            if key in done_pairs:
                continue

            run_dir = _trial_run_dir(final_dir, method=mname, seed=int(sd))
            salv = _salvage_if_finished(run_dir)

            if salv is None:
                override = dict(base_override)
                override.setdefault("train", {})
                override["train"]["seed"] = int(sd)
                override["train"]["epochs"] = int(final_epochs)
                override["train"]["warmup_ratio"] = float(fixed_wu)
                override["train"]["lr"] = float(lr)

                _run_one(
                    base_config_path=base_config_path,
                    schedule_path=schedule_path,
                    set_args=set_args,
                    final_dir=final_dir,
                    method_name=mname,
                    seed=int(sd),
                    override=override,
                )
                salv = _salvage_if_finished(run_dir)

            if salv is None:
                continue

            summary = salv["summary"]
            cfg_resolved = salv["cfg"]

            metrics_csv = run_dir / "metrics.csv"
            all_curves_dir.mkdir(parents=True, exist_ok=True)
            if metrics_csv.exists():
                shutil.copyfile(metrics_csv, all_curves_dir / f"{mname}_s{int(sd)}.csv")

            score = _score_from_summary(summary, weights)

            row = {
                "method": mname,
                "seed": int(sd),
                "trial_tag": f"final_{mname}_s{int(sd)}",
                "run_dir": str(run_dir),
                "metrics_csv": str(metrics_csv) if metrics_csv.exists() else "",
                "val_max": float(summary["val_max"]),
                "val_final": float(summary["val_final"]),
                "val_avg": float(summary["val_avg"]),
                "train_max": float(summary["train_max"]) if "train_max" in summary else np.nan,
                "train_final": float(summary["train_final"]) if "train_final" in summary else np.nan,
                "score": float(score),
                "trial_cfg_json": json.dumps(cfg_resolved, sort_keys=True),
            }
            _append_row(trials_csv, row)
            done_pairs.add(key)

            method_cfg = cfg_resolved["method"]
            train_cfg = cfg_resolved["train"]
            task_cfg = cfg_resolved["task"]
            model_cfg = cfg_resolved["model"]

            if mname == "ours":
                lora_cfg = method_cfg["ours"]["lora"]
            else:
                lora_cfg = method_cfg[mname]["lora"]

            repro_runs.append(
                {
                    "method": mname,
                    "seed": int(sd),
                    "run_dir": str(run_dir),
                    "train": {
                        "lr": float(train_cfg["lr"]),
                        "warmup_ratio": float(train_cfg["warmup_ratio"]),
                        "epochs": int(train_cfg["epochs"]),
                        "batch_size": int(train_cfg["batch_size"]),
                        "seed": int(train_cfg["seed"]),
                    },
                    "model": model_cfg,
                    "task": task_cfg,
                    "lora": {
                        "r": int(lora_cfg["r"]),
                        "R": int(lora_cfg["R"]) if lora_cfg.get("R", None) is not None else None,
                        "alpha": float(lora_cfg["alpha"]),
                        "dropout": float(lora_cfg["dropout"]),
                    },
                    "method_cfg": method_cfg[mname] if mname in method_cfg else method_cfg["ours"],
                }
            )

    # aggregate
    rows = _read_rows(trials_csv)
    methods_present = sorted({str(r["method"]) for r in rows})
    agg = {
        "final_dir": str(final_dir),
        "seeds": [int(x) for x in final_seeds],
        "weights": {"w_max": w_max, "w_final": w_final, "w_avg": w_avg},
        "methods": {},
    }
    for m in methods_present:
        agg["methods"][m] = _aggregate_method(rows, m)
    dense_early_epochs = int((((cfg.get("train", {}) or {}).get("eval", {}) or {}).get("dense_early_epochs", 2)))
    agg["ours_diagnostics"] = _aggregate_ours_diagnostics(rows, dense_early_epochs=dense_early_epochs)

    dump_json(summary_json, agg)
    print(f"[final] saved: {summary_json}")

    # provenance (strict, minimal)
    prov = {
        "best_path": str(bp),
        "plan_path": str(plan_path),
        "ours_r": int(ours_r),
        "ours_R": int(ours_R),
        "ours_lr": float(ours_lr),
        "baseline_lr_r": float(baseline_lr_r),
        "baseline_lr_R": float(baseline_lr_R),
        "final_epochs": int(final_epochs),
        "fixed_warmup_ratio": float(fixed_wu),
        "final_seeds": [int(x) for x in final_seeds],
        "repro_runs": repro_runs,
    }
    dump_json(final_dir / "final_provenance.json", prov)

    # copy best curves per method (by score)
    best_curves_dir.mkdir(parents=True, exist_ok=True)
    for m in methods_present:
        rs = [r for r in rows if str(r["method"]) == m]
        rs_sorted = sorted(rs, key=lambda r: float(r["score"]), reverse=True)
        if not rs_sorted:
            continue
        best = rs_sorted[0]
        p = Path(str(best["metrics_csv"]))
        if p.exists():
            shutil.copyfile(p, best_curves_dir / f"best_{m}.csv")
