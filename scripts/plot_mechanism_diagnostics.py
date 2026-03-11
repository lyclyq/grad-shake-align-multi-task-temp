#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Curve = Tuple[np.ndarray, np.ndarray]

_METRICS: List[Tuple[str, str, str]] = [
    ("val/acc", "val_acc", "Validation Accuracy"),
    ("diag/info_retention_ratio", "info_retention_ratio", "Information Retention Ratio"),
    ("diag/residual_visibility", "residual_visibility", "Residual Visibility"),
    ("diag/conflict_resolution_rate", "conflict_resolution_rate", "Conflict Resolution Rate"),
]


def _read_eval_curve(metrics_csv: Path, ykey: str) -> Optional[Curve]:
    try:
        df = pd.read_csv(metrics_csv)
    except Exception:
        return None
    if "step" not in df.columns or ykey not in df.columns:
        return None

    step = pd.to_numeric(df["step"], errors="coerce")
    y = pd.to_numeric(df[ykey], errors="coerce")
    mask = step.notna() & y.notna()

    if "probe/is_eval" in df.columns:
        is_eval = pd.to_numeric(df["probe/is_eval"], errors="coerce")
        mask = mask & is_eval.notna() & (is_eval > 0.5)

    if not mask.any():
        return None

    data = pd.DataFrame({"step": step[mask].astype(int), "y": y[mask].astype(float)})
    data = data.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    if data.empty:
        return None
    return data["step"].to_numpy(dtype=int), data["y"].to_numpy(dtype=float)


def _collect_seed_curves(trial_runs_dir: Path, metric: str) -> Dict[int, Curve]:
    out: Dict[int, Curve] = {}
    for metrics_csv in sorted((trial_runs_dir / "ours").glob("s*/metrics.csv")):
        seed_dir = metrics_csv.parent
        try:
            seed = int(seed_dir.name.lstrip("s"))
        except Exception:
            continue
        curve = _read_eval_curve(metrics_csv, metric)
        if curve is not None:
            out[seed] = curve
    return out


def _aggregate_seed_curves(seed_curves: Dict[int, Curve]) -> Optional[pd.DataFrame]:
    if not seed_curves:
        return None
    pooled: Dict[int, List[float]] = {}
    for _, (steps, vals) in seed_curves.items():
        for step, val in zip(steps.tolist(), vals.tolist()):
            pooled.setdefault(int(step), []).append(float(val))

    if not pooled:
        return None

    rows = []
    for step in sorted(pooled.keys()):
        vals = np.asarray(pooled[step], dtype=float)
        rows.append(
            {
                "step": int(step),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def _save_metric_plot(df: pd.DataFrame, *, title: str, ylabel: str, out_path: Path) -> None:
    x = df["step"].to_numpy(dtype=int)
    mu = df["mean"].to_numpy(dtype=float)
    sd = df["std"].to_numpy(dtype=float)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(x, mu, color="tab:red", linewidth=2.2)
    plt.fill_between(x, mu - sd, mu + sd, color="tab:red", alpha=0.18)
    plt.xlabel("global step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _save_overview_plot(metric_frames: List[Tuple[str, str, pd.DataFrame]], out_path: Path) -> None:
    fig, axes = plt.subplots(len(metric_frames), 1, figsize=(8.0, 12.0), sharex=True)
    if len(metric_frames) == 1:
        axes = [axes]

    for ax, (title, ylabel, df) in zip(axes, metric_frames):
        x = df["step"].to_numpy(dtype=int)
        mu = df["mean"].to_numpy(dtype=float)
        sd = df["std"].to_numpy(dtype=float)
        ax.plot(x, mu, color="tab:red", linewidth=2.0)
        ax.fill_between(x, mu - sd, mu + sd, color="tab:red", alpha=0.18)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[-1].set_xlabel("global step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot seed-mean mechanism diagnostics from ours/final runs.")
    ap.add_argument("trial_runs_dir", help="Path to final_dir/trial_runs")
    args = ap.parse_args()

    tr = Path(args.trial_runs_dir).expanduser().resolve()
    if not tr.exists():
        raise FileNotFoundError(f"trial_runs_dir not found: {tr}")

    out_dir = tr / "_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    overview_frames: List[Tuple[str, str, pd.DataFrame]] = []
    for metric, stem, title in _METRICS:
        seed_curves = _collect_seed_curves(tr, metric)
        agg = _aggregate_seed_curves(seed_curves)
        if agg is None or agg.empty:
            print(f"[WARN] skip {metric}: no valid eval-step curves under {tr / 'ours'}")
            continue
        csv_path = out_dir / f"{stem}_seed_mean.csv"
        png_path = out_dir / f"{stem}_seed_mean.png"
        agg.to_csv(csv_path, index=False)
        _save_metric_plot(agg, title=title, ylabel=metric, out_path=png_path)
        overview_frames.append((title, metric, agg))
        print(f"[OK] saved {csv_path}")
        print(f"[OK] saved {png_path}")

    if overview_frames:
        overview_path = out_dir / "mechanism_diagnostics_seed_mean_overview.png"
        _save_overview_plot(overview_frames, overview_path)
        print(f"[OK] saved {overview_path}")
    else:
        print("[WARN] no mechanism diagnostics were plotted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
