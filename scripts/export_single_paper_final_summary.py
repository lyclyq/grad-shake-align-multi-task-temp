#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["ours_r", "ours_R", "baseline_r", "baseline_R"]
METHOD_LABELS = {
    "ours_r": "ours r",
    "ours_R": "ours R",
    "baseline_r": "baseline r",
    "baseline_R": "baseline R",
}
OURS_DIAG_SPECS = [
    ("train/gate0_trigger_rate", "gate0_trigger_rate", "Gate0 Trigger Rate", "rate"),
    ("train/pull_to_r_rate", "pull_to_r_rate", "Pull Toward r Rate", "rate"),
    ("train/pull_to_R_rate", "pull_to_R_rate", "Pull Toward R Rate", "rate"),
    ("train/alpha_pull_to_r_mean", "alpha_pull_to_r_mean", "Pull Strength Toward r", "alpha"),
    ("train/alpha_pull_to_R_mean", "alpha_pull_to_R_mean", "Pull Strength Toward R", "alpha"),
]


@dataclass
class DatasetRun:
    dataset_key: str
    final_dir: Path
    trial_runs_dir: Path


def _discover_single_final_dirs(runs_root: Path) -> List[DatasetRun]:
    out: List[DatasetRun] = []
    for group_dir in sorted(runs_root.glob("paper_suite_main_single_*")):
        if not group_dir.is_dir():
            continue
        compare_dirs = sorted(
            [p for p in group_dir.glob("final__ours-only-compare__*") if p.is_dir()],
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        if compare_dirs:
            chosen = compare_dirs[-1]
        else:
            final_dirs = sorted(
                [p for p in group_dir.glob("final__paper-suite-main-single*") if p.is_dir()],
                key=lambda p: (p.stat().st_mtime, p.name),
            )
            if not final_dirs:
                continue
            chosen = final_dirs[-1]
        out.append(
            DatasetRun(
                dataset_key=group_dir.name.removeprefix("paper_suite_main_single_"),
                final_dir=chosen.resolve(),
                trial_runs_dir=(chosen / "trial_runs").resolve(),
            )
        )
    return out


def _load_seed_curve(metrics_csv: Path, value_key: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if "step" not in df.columns or value_key not in df.columns:
        raise KeyError(f"missing required columns in {metrics_csv}: step / {value_key}")
    step = pd.to_numeric(df["step"], errors="coerce")
    vals = pd.to_numeric(df[value_key], errors="coerce")
    mask = step.notna() & vals.notna()
    out = pd.DataFrame({"step": step[mask].astype(int), "value": vals[mask].astype(float)})
    out = out.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    if out.empty:
        raise ValueError(f"no valid curve for {metrics_csv} key={value_key}")
    return out


def _collect_method_seed_curves(trial_runs_dir: Path, variant: str, value_key: str) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for metrics_csv in sorted((trial_runs_dir / variant).glob("s*/metrics.csv")):
        seed_name = metrics_csv.parent.name
        try:
            seed = int(seed_name.lstrip("s"))
        except ValueError:
            continue
        out[seed] = _load_seed_curve(metrics_csv, value_key)
    if not out:
        raise FileNotFoundError(f"no metrics found for {variant} under {trial_runs_dir}")
    return out


def _mean_and_var_across_seeds(seed_curves: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    common_steps: Optional[set[int]] = None
    for curve in seed_curves.values():
        steps = set(int(x) for x in curve["step"].tolist())
        common_steps = steps if common_steps is None else (common_steps & steps)
    if not common_steps:
        raise ValueError("no common steps across seeds")
    xs = np.array(sorted(common_steps), dtype=int)
    rows = []
    for step in xs.tolist():
        vals = []
        for curve in seed_curves.values():
            row = curve.loc[curve["step"] == step, "value"]
            vals.append(float(row.iloc[-1]))
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "step": int(step),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "var": float(arr.var()),
                "n": int(arr.size),
            }
        )
    return pd.DataFrame(rows)


def _copy_truncated_csv(src: Path, dest: Path, cutoff_step: int) -> None:
    df = pd.read_csv(src)
    if "step" in df.columns:
        step = pd.to_numeric(df["step"], errors="coerce")
        df = df.loc[step.isna() | (step <= int(cutoff_step))].copy()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)


def _extract_method_hparams(config: dict, method_name: str) -> dict:
    train = config.get("train", {}) or {}
    method = config.get("method", {}) or {}
    out = {
        "train.lr": train.get("lr"),
        "train.warmup_ratio": train.get("warmup_ratio"),
        "train.epochs": train.get("epochs"),
        "train.batch_size": train.get("batch_size"),
    }
    if method_name == "baseline_r":
        cfg = (method.get("baseline_r") or {}).copy()
        out.update(
            {
                "method.name": "baseline_r",
                "lora.r": ((cfg.get("lora") or {}).get("r")),
                "grad_solver": cfg.get("grad_solver", "avg"),
                "cagrad.c": ((cfg.get("cagrad") or {}).get("c")),
            }
        )
    elif method_name == "baseline_R":
        cfg = (method.get("baseline_R") or {}).copy()
        out.update(
            {
                "method.name": "baseline_R",
                "lora.r": ((cfg.get("lora") or {}).get("r")),
                "grad_solver": cfg.get("grad_solver", "avg"),
                "cagrad.c": ((cfg.get("cagrad") or {}).get("c")),
            }
        )
    elif method_name == "ours":
        cfg = (method.get("ours") or {}).copy()
        out.update(
            {
                "method.name": "ours",
                "lora.r": ((cfg.get("lora") or {}).get("r")),
                "lora.R": ((cfg.get("lora") or {}).get("R")),
                "lora.alpha": ((cfg.get("lora") or {}).get("alpha")),
                "lora.dropout": ((cfg.get("lora") or {}).get("dropout")),
                "routing_delta": cfg.get("routing_delta"),
                "tau_N": ((cfg.get("trigger_gate0") or {}).get("tau_N")),
                "tau_D": ((cfg.get("trigger_gate0") or {}).get("tau_D")),
                "gamma_pull": ((cfg.get("pulling") or {}).get("gamma_pull")),
                "k_pull": ((cfg.get("pulling") or {}).get("k_pull")),
                "votes": cfg.get("votes"),
                "samples_per_vote": ((cfg.get("voting") or {}).get("samples_per_vote")),
                "allow_tail": ((cfg.get("voting") or {}).get("allow_tail")),
                "keep_single_votes": ((cfg.get("voting") or {}).get("keep_single_votes")),
                "ema_H": cfg.get("ema_H"),
                "compensation.enabled": ((cfg.get("compensation") or {}).get("enabled")),
                "compensation.ridge_lambda": ((cfg.get("compensation") or {}).get("ridge_lambda")),
            }
        )
    else:
        raise ValueError(f"unsupported method_name={method_name}")
    return out


def _plot_dataset_mean_curve(df: pd.DataFrame, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.6))
    x = df["step"].to_numpy(dtype=int)
    mu = df["mean"].to_numpy(dtype=float)
    sd = df["std"].to_numpy(dtype=float)
    plt.plot(x, mu, linewidth=2.0, color="tab:red")
    plt.fill_between(x, mu - sd, mu + sd, color="tab:red", alpha=0.18)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_bar(values: pd.Series, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(7.6, 4.8))
    x = np.arange(len(values))
    plt.bar(x, values.to_numpy(dtype=float), color=["tab:red", "indianred", "tab:blue", "steelblue"])
    plt.xticks(x, [METHOD_LABELS[idx] for idx in values.index], rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def export_single_summary(runs_root: Path, output_root: Path) -> None:
    dataset_runs = _discover_single_final_dirs(runs_root)
    if not dataset_runs:
        raise FileNotFoundError(f"no single paper final dirs found under {runs_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    per_dataset_rows = []
    hparam_rows = []
    val_acc_table: Dict[str, Dict[str, float]] = {m: {} for m in METHOD_ORDER}
    val_loss_table: Dict[str, Dict[str, float]] = {m: {} for m in METHOD_ORDER}
    variance_table: Dict[str, Dict[str, float]] = {m: {} for m in METHOD_ORDER}

    for run in dataset_runs:
        ds_out = output_root / run.dataset_key
        ds_out.mkdir(parents=True, exist_ok=True)

        curves = {
            "baseline_r": _mean_and_var_across_seeds(_collect_method_seed_curves(run.trial_runs_dir, "baseline_r", "val/acc")),
            "baseline_R": _mean_and_var_across_seeds(_collect_method_seed_curves(run.trial_runs_dir, "baseline_R", "val/acc")),
            "ours_R": _mean_and_var_across_seeds(_collect_method_seed_curves(run.trial_runs_dir, "ours", "val/acc")),
            "ours_r": _mean_and_var_across_seeds(_collect_method_seed_curves(run.trial_runs_dir, "ours", "val/acc_r_only")),
        }

        merged = curves["ours_r"][["step", "mean"]].rename(columns={"mean": "ours_r"})
        for key in ["ours_R", "baseline_r", "baseline_R"]:
            merged = merged.merge(curves[key][["step", "mean"]].rename(columns={"mean": key}), on="step", how="inner")
        merged["advantage_mean"] = (
            (merged["ours_r"] - merged["baseline_r"]) + (merged["ours_R"] - merged["baseline_R"])
        ) / 2.0
        best_row = merged.sort_values(["advantage_mean", "step"], ascending=[False, True]).iloc[0]
        cutoff_step = int(best_row["step"])
        merged.to_csv(ds_out / "cutoff_selection_curve.csv", index=False)

        per_dataset_rows.append(
            {
                "dataset": run.dataset_key,
                "final_dir": str(run.final_dir),
                "cutoff_step": cutoff_step,
                "advantage_mean": float(best_row["advantage_mean"]),
                "ours_r_at_cutoff": float(best_row["ours_r"]),
                "ours_R_at_cutoff": float(best_row["ours_R"]),
                "baseline_r_at_cutoff": float(best_row["baseline_r"]),
                "baseline_R_at_cutoff": float(best_row["baseline_R"]),
            }
        )

        trunc_root = ds_out / "truncated_raw"
        for variant in ["baseline_r", "baseline_R", "ours"]:
            for src in sorted((run.trial_runs_dir / variant).glob("s*/metrics.csv")):
                dest = trunc_root / "trial_runs" / variant / src.parent.name / "metrics.csv"
                _copy_truncated_csv(src, dest, cutoff_step)
        for csv_src in sorted((run.final_dir / "all_curves").glob("*.csv")):
            if csv_src.name.startswith(("baseline_r_", "baseline_R_", "ours_")):
                _copy_truncated_csv(csv_src, trunc_root / "all_curves" / csv_src.name, cutoff_step)
        for csv_src in sorted((run.final_dir / "best_curves").glob("*.csv")):
            if csv_src.name in {"best_baseline_r.csv", "best_baseline_R.csv", "best_ours.csv"}:
                _copy_truncated_csv(csv_src, trunc_root / "best_curves" / csv_src.name, cutoff_step)

        # Reproducible hyperparameters: use one seed config per method.
        baseline_r_cfg = json.loads((run.trial_runs_dir / "baseline_r" / "s2" / "config_resolved.json").read_text(encoding="utf-8"))
        baseline_R_cfg = json.loads((run.trial_runs_dir / "baseline_R" / "s2" / "config_resolved.json").read_text(encoding="utf-8"))
        ours_cfg = json.loads((run.trial_runs_dir / "ours" / "s2" / "config_resolved.json").read_text(encoding="utf-8"))
        repro_json = {
            "dataset": run.dataset_key,
            "final_dir": str(run.final_dir),
            "cutoff_step": cutoff_step,
            "baseline_r": _extract_method_hparams(baseline_r_cfg, "baseline_r"),
            "baseline_R": _extract_method_hparams(baseline_R_cfg, "baseline_R"),
            "ours": _extract_method_hparams(ours_cfg, "ours"),
        }
        (ds_out / "repro_hparams.json").write_text(json.dumps(repro_json, indent=2, ensure_ascii=False), encoding="utf-8")
        for method_line, source_method, cfg in [
            ("baseline_r", "baseline_r", baseline_r_cfg),
            ("baseline_R", "baseline_R", baseline_R_cfg),
            ("ours_r", "ours", ours_cfg),
            ("ours_R", "ours", ours_cfg),
        ]:
            row = {"dataset": run.dataset_key, "method_line": method_line}
            row.update(_extract_method_hparams(cfg, source_method))
            hparam_rows.append(row)

        # Tables at cutoff step.
        value_specs = {
            "baseline_r": ("baseline_r", "val/acc", "val/loss"),
            "baseline_R": ("baseline_R", "val/acc", "val/loss"),
            "ours_r": ("ours", "val/acc_r_only", "val/loss_r_only"),
            "ours_R": ("ours", "val/acc", "val/loss"),
        }
        for method_line, (variant, acc_key, loss_key) in value_specs.items():
            acc_mean = _mean_and_var_across_seeds(_collect_method_seed_curves(run.trial_runs_dir, variant, acc_key))
            loss_mean = _mean_and_var_across_seeds(_collect_method_seed_curves(run.trial_runs_dir, variant, loss_key))
            val_acc = float(acc_mean.loc[acc_mean["step"] == cutoff_step, "mean"].iloc[-1])
            val_loss = float(loss_mean.loc[loss_mean["step"] == cutoff_step, "mean"].iloc[-1])
            variance_mean = float(acc_mean.loc[acc_mean["step"] <= cutoff_step, "var"].mean())
            val_acc_table[method_line][run.dataset_key] = val_acc
            val_loss_table[method_line][run.dataset_key] = val_loss
            variance_table[method_line][run.dataset_key] = variance_mean

        # Truncated ours gate / pull diagnostics.
        ours_diag_dir = ds_out / "ours_truncated_diagnostics"
        ours_diag_dir.mkdir(parents=True, exist_ok=True)
        ours_diag_rows = []
        for metric_key, stem, title, ylabel in OURS_DIAG_SPECS:
            seed_curves = _collect_method_seed_curves(run.trial_runs_dir, "ours", metric_key)
            agg = _mean_and_var_across_seeds(seed_curves)
            agg = agg.loc[agg["step"] <= cutoff_step].copy()
            if agg.empty:
                continue
            agg.to_csv(ours_diag_dir / f"{stem}.csv", index=False)
            _plot_dataset_mean_curve(
                agg,
                title=f"{run.dataset_key}: {title} (truncated)",
                ylabel=ylabel,
                out_path=ours_diag_dir / f"{stem}.png",
            )
            ours_diag_rows.append({"metric": metric_key, "csv": f"{stem}.csv", "png": f"{stem}.png"})
        if ours_diag_rows:
            pd.DataFrame(ours_diag_rows).to_csv(ours_diag_dir / "manifest.csv", index=False)

        # Keep lightweight metadata in each dataset folder.
        (ds_out / "source_final_dir.txt").write_text(f"{run.final_dir}\n", encoding="utf-8")

    cutoff_df = pd.DataFrame(per_dataset_rows).sort_values("dataset")
    cutoff_df.to_csv(output_root / "cutoff_steps.csv", index=False)
    pd.DataFrame(hparam_rows).sort_values(["dataset", "method_line"]).to_csv(output_root / "repro_hparams.csv", index=False)

    val_acc_df = pd.DataFrame.from_dict(val_acc_table, orient="index").reindex(METHOD_ORDER)
    val_acc_df.index = [METHOD_LABELS[idx] for idx in val_acc_df.index]
    val_acc_df = val_acc_df.reindex(sorted(val_acc_df.columns), axis=1)
    val_acc_df.to_csv(output_root / "final_val_acc_at_cutoff.csv")

    val_loss_df = pd.DataFrame.from_dict(val_loss_table, orient="index").reindex(METHOD_ORDER)
    val_loss_df.index = [METHOD_LABELS[idx] for idx in val_loss_df.index]
    val_loss_df = val_loss_df.reindex(sorted(val_loss_df.columns), axis=1)
    val_loss_df.to_csv(output_root / "final_val_loss_at_cutoff.csv")

    variance_df = pd.DataFrame.from_dict(variance_table, orient="index").reindex(METHOD_ORDER)
    variance_df.index = [METHOD_LABELS[idx] for idx in variance_df.index]
    variance_df = variance_df.reindex(sorted(variance_df.columns), axis=1)
    variance_df.to_csv(output_root / "val_acc_seed_variance_mean.csv")

    final_val_acc_global = pd.Series(
        {method: float(np.mean(list(cols.values()))) for method, cols in val_acc_table.items()},
        name="mean_final_val_acc",
    ).reindex(METHOD_ORDER)
    final_val_acc_global.to_csv(output_root / "global_mean_final_val_acc.csv", header=True)
    _plot_bar(
        final_val_acc_global,
        title="Mean Final Val Acc Across Single-Paper Datasets",
        ylabel="val/acc",
        out_path=output_root / "global_mean_final_val_acc.png",
    )

    global_variance = pd.Series(
        {method: float(np.mean(list(cols.values()))) for method, cols in variance_table.items()},
        name="mean_seed_variance_over_steps",
    ).reindex(METHOD_ORDER)
    global_variance.to_csv(output_root / "global_mean_seed_variance.csv", header=True)
    _plot_bar(
        global_variance,
        title="Mean Seed Variance Across Single-Paper Datasets",
        ylabel="mean variance",
        out_path=output_root / "global_mean_seed_variance.png",
    )

    summary = {
        "runs_root": str(runs_root),
        "output_root": str(output_root),
        "datasets": cutoff_df["dataset"].tolist(),
        "assumptions": {
            "cutoff_step_rule": "argmax over common steps of 0.5*((ours_r_only_mean-baseline_r_mean)+(ours_full_mean-baseline_R_mean))",
            "variance_metric": "mean over steps<=cutoff of seed variance on validation accuracy for each method line",
            "mrpc_roberta_source": "prefer latest ours-only compare final when present",
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export truncated single-paper final summaries.")
    ap.add_argument("--runs-root", default="runs", help="Runs root under grad-shake-align")
    ap.add_argument("--output-root", default="paper_final/single", help="Output directory")
    args = ap.parse_args()

    cwd = Path.cwd()
    runs_root = (cwd / args.runs_root).resolve()
    output_root = (cwd / args.output_root).resolve()
    export_single_summary(runs_root, output_root)
    print(f"[OK] exported summary -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
