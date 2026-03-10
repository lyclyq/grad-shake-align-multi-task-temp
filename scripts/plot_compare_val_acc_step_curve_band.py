#!/usr/bin/env python3
"""
Plot step-wise val/acc curve comparison between two settings across seeds.

This script expects per-seed run dirs produced by `scripts/run.py train` that contain:
  - metrics.csv

We extract rows where `probe/is_eval == 1` (evaluation probes), then aggregate by step:
  - mean across seeds
  - min/max across seeds (shaded band)

Note: If your run logs only evaluate at the final step (e.g., eval strategy 'steps' with every_steps=max_steps),
the resulting curve will have only one point. To get a true curve, run with `--set stage=final` so that
`dense_early` isn't auto-downgraded.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


_SEED_RE = re.compile(r"_s(\d+)$")


def _as_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _load_eval_curve(metrics_csv: Path, *, y_key: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with metrics_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            is_eval = str(row.get("probe/is_eval", "")).strip()
            if is_eval in {"", "0", "0.0"}:
                continue
            yv = str(row.get(y_key, "")).strip()
            if not yv:
                continue
            st = row.get("step", "")
            if not str(st).strip():
                continue
            step = int(float(st))
            out[step] = float(yv)
    return out


def _collect(glob_pat: str, *, y_key: str) -> Dict[int, Dict[int, float]]:
    """
    Returns: seed -> {step -> y}
    """
    base = Path()
    out: Dict[int, Dict[int, float]] = {}
    for d in sorted(base.glob(glob_pat)):
        if not d.is_dir():
            continue
        m = _SEED_RE.search(d.name)
        if not m:
            continue
        seed = int(m.group(1))
        mcsv = d / "metrics.csv"
        if not mcsv.exists():
            continue
        out[seed] = _load_eval_curve(mcsv, y_key=y_key)
    return out


def _agg_by_step(curves: Dict[int, Dict[int, float]]) -> Tuple[List[int], List[float], List[float], List[float], List[int]]:
    steps = sorted({st for c in curves.values() for st in c.keys()})
    mean: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    n: List[int] = []
    for st in steps:
        ys = [c.get(st) for c in curves.values() if st in c and not math.isnan(float(c.get(st)))]
        n.append(len(ys))
        if not ys:
            mean.append(float("nan"))
            lo.append(float("nan"))
            hi.append(float("nan"))
        else:
            mean.append(sum(ys) / len(ys))
            lo.append(min(ys))
            hi.append(max(ys))
    return steps, mean, lo, hi, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_glob", required=True)
    ap.add_argument("--b_glob", required=True)
    ap.add_argument("--a_label", required=True)
    ap.add_argument("--b_label", required=True)
    ap.add_argument("--y_key", default="val/acc", help="metrics.csv column to plot (default: val/acc)")
    ap.add_argument("--x_min", type=float, default=None, help="Optional x-axis min (global step)")
    ap.add_argument("--x_max", type=float, default=None, help="Optional x-axis max (global step)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = _collect(args.a_glob, y_key=str(args.y_key))
    B = _collect(args.b_glob, y_key=str(args.y_key))

    seeds_a = sorted(A.keys())
    seeds_b = sorted(B.keys())
    common_seeds = sorted(set(seeds_a) & set(seeds_b))
    print("[INFO] seeds A:", seeds_a)
    print("[INFO] seeds B:", seeds_b)
    print("[INFO] common:", common_seeds)

    steps_a, mean_a, lo_a, hi_a, n_a = _agg_by_step({s: A[s] for s in seeds_a})
    steps_b, mean_b, lo_b, hi_b, n_b = _agg_by_step({s: B[s] for s in seeds_b})

    all_steps = sorted(set(steps_a) | set(steps_b))
    if not all_steps:
        raise SystemExit("No eval points found. (No rows with probe/is_eval==1 and y_key present.)")

    # Map to a unified step axis for plotting.
    def remap(steps: List[int], ys: List[float]) -> Dict[int, float]:
        return {int(st): float(y) for st, y in zip(steps, ys)}

    ma = remap(steps_a, mean_a)
    la = remap(steps_a, lo_a)
    ha = remap(steps_a, hi_a)
    mb = remap(steps_b, mean_b)
    lb = remap(steps_b, lo_b)
    hb = remap(steps_b, hi_b)

    xa = [st for st in all_steps if st in ma]
    xb = [st for st in all_steps if st in mb]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4.2), dpi=160)
    if xa:
        ya = [ma[st] for st in xa]
        plt.plot(xa, ya, linewidth=2.2, marker="o", markersize=5, label=args.a_label)
        # With a single x point, fill_between is effectively invisible; keep it anyway for consistency.
        plt.fill_between(xa, [la[st] for st in xa], [ha[st] for st in xa], alpha=0.18)
    if xb:
        yb = [mb[st] for st in xb]
        plt.plot(xb, yb, linewidth=2.2, marker="o", markersize=5, label=args.b_label)
        plt.fill_between(xb, [lb[st] for st in xb], [hb[st] for st in xb], alpha=0.18)

    plt.xlabel("global step")
    plt.ylabel(str(args.y_key))
    plt.title(f"Step-wise {args.y_key} (mean + min/max range across seeds)")
    if args.x_min is not None or args.x_max is not None:
        xmin = float(args.x_min) if args.x_min is not None else None
        xmax = float(args.x_max) if args.x_max is not None else None
        plt.xlim(left=xmin, right=xmax)
    else:
        # If we only have one eval point (common when eval_every_steps == max_steps),
        # matplotlib draws a zero-length line, so we keep a small margin for readability.
        if len(all_steps) == 1:
            st = float(all_steps[0])
            plt.xlim(0.0, st + 1.0)
            plt.xticks([0, int(st)])
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outp.as_posix())
    print("[OK] wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
