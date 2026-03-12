#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Curve = Tuple[np.ndarray, np.ndarray]


def _fname(metric: str) -> str:
    return metric.replace("/", "_").replace(" ", "_").replace(":", "_")


def read_step_curve(metrics_csv: Path, ykey: str) -> Optional[Curve]:
    try:
        df = pd.read_csv(metrics_csv)
    except Exception:
        return None
    if "step" not in df.columns or ykey not in df.columns:
        return None
    x = pd.to_numeric(df["step"], errors="coerce")
    y = pd.to_numeric(df[ykey], errors="coerce")
    m = x.notna() & y.notna()
    if not m.any():
        return None
    d = pd.DataFrame({"x": x[m].astype(int), "y": y[m].astype(float)})
    d = d.sort_values("x").drop_duplicates(subset=["x"], keep="last")
    return d["x"].to_numpy(dtype=int), d["y"].to_numpy(dtype=float)


def collect_variant_curves(trial_runs_dir: Path, variant: str, metric: str) -> List[Curve]:
    out: List[Curve] = []
    for p in sorted((trial_runs_dir / variant).glob("s*/metrics.csv")):
        curve = read_step_curve(p, metric)
        if curve is not None:
            out.append(curve)
    return out


def collect_metrics(trial_runs_dir: Path, variants: List[str]) -> List[str]:
    metrics: Set[str] = set()
    skip = {"step", "epoch", "step_in_epoch", "probe/is_eval"}
    for variant in variants:
        for p in sorted((trial_runs_dir / variant).glob("s*/metrics.csv")):
            try:
                df = pd.read_csv(p, nrows=8)
            except Exception:
                continue
            for col in df.columns:
                name = str(col).strip()
                if not name or name in skip:
                    continue
                metrics.add(name)
    return sorted(metrics)


def align_and_stack(curves: List[Curve]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not curves:
        return None, None
    common = None
    for x, _ in curves:
        s = set(map(int, x.tolist()))
        common = s if common is None else (common & s)
    if not common:
        return None, None
    xs = np.array(sorted(common), dtype=int)
    ys = []
    for x, y in curves:
        mp = {int(a): float(b) for a, b in zip(x, y)}
        ys.append([mp[int(t)] for t in xs])
    return xs, np.asarray(ys, dtype=float)


def mean_std_plot(x: np.ndarray, ys: np.ndarray, label: str, *, color: str, linestyle: str = "-", linewidth: float = 2.0) -> None:
    mu = ys.mean(axis=0)
    sd = ys.std(axis=0)
    plt.plot(x, mu, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
    plt.fill_between(x, mu - sd, mu + sd, alpha=0.16, color=color)


def plot_metric(trial_runs_dir: Path, metric: str, out_path: Path) -> None:
    variants: Dict[str, Tuple[str, str]] = {
        "ours": ("ours", "tab:red"),
        "ablate_no_gate": ("No Gate", "tab:blue"),
        "ablate_no_compensation": ("No Compensation", "tab:green"),
    }
    plotted = False
    plt.figure(figsize=(8.5, 4.6), dpi=160)
    for variant, (label, color) in variants.items():
        curves = collect_variant_curves(trial_runs_dir, variant, metric)
        x, ys = align_and_stack(curves)
        if x is None:
            continue
        mean_std_plot(x, ys, label, color=color)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.title(f"Ablation vs Ours: {metric}")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_runs_dir", type=str)
    args = ap.parse_args()

    trial_runs_dir = Path(args.trial_runs_dir)
    out_dir = trial_runs_dir / "_plots_ablation"
    variants = ["ours", "ablate_no_gate", "ablate_no_compensation"]
    metrics = collect_metrics(trial_runs_dir, variants)
    for metric in metrics:
        plot_metric(trial_runs_dir, metric, out_dir / f"{_fname(metric)}.png")
    print(f"[OK] ablation plots saved under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
