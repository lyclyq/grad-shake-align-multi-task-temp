#!/usr/bin/env python3
"""
Aggregate a tau sweep (e.g., method.ours.trigger_gate0.tau_D) across seeds and plot a bar chart.

Assumes each run directory contains `summary.json` produced by the trainer.
We group runs by tau value extracted from directory name, average across seeds, and plot mean + range.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


_SEED_RE = re.compile(r"_s(\d+)$")


def _read_metric(run_dir: Path, key: str, *, w_max: float, w_final: float) -> float:
    p = run_dir / "summary.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    if key == "weighted":
        if "val_max" not in obj or "val_final" not in obj:
            raise KeyError(f"Missing val_max/val_final in {p}")
        return float(w_max) * float(obj["val_max"]) + float(w_final) * float(obj["val_final"])
    if key not in obj:
        raise KeyError(f"Missing {key} in {p}")
    return float(obj[key])


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/mvp_tauD_sweep", help="Sweep runs directory")
    ap.add_argument(
        "--sweep_key",
        default="tauD",
        help="Directory token before value, e.g. tauD or tauN (expects '<key>_0p123...')",
    )
    ap.add_argument(
        "--metric",
        default="val_max",
        help="summary.json metric key, or 'weighted' for w_max*val_max + w_final*val_final",
    )
    ap.add_argument("--w_max", type=float, default=0.5, help="Weight for val_max when --metric weighted")
    ap.add_argument("--w_final", type=float, default=0.5, help="Weight for val_final when --metric weighted")
    ap.add_argument("--out", default="runs/mvp_tauD_sweep/tauD_bar_val_max.png", help="Output .png path")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"missing root: {root}")

    grouped: Dict[float, List[Tuple[int, float, Path]]] = defaultdict(list)  # tau -> [(seed, val, dir)]
    tau_re = re.compile(rf"{re.escape(str(args.sweep_key))}_([0-9]+p[0-9]+)")

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m_tau = tau_re.search(d.name)
        m_seed = _SEED_RE.search(d.name)
        if not m_tau or not m_seed:
            continue
        tau_s = m_tau.group(1).replace("p", ".")
        tau = float(tau_s)
        seed = int(m_seed.group(1))
        try:
            v = _read_metric(d, str(args.metric), w_max=float(args.w_max), w_final=float(args.w_final))
        except Exception as e:
            print(f"[WARN] skip {d} ({type(e).__name__}: {e})")
            continue
        grouped[tau].append((seed, v, d))

    if not grouped:
        raise SystemExit(
            f"no runs found (expected dir names containing '{args.sweep_key}_0p...' and suffix '_s<seed>')"
        )

    taus = sorted(grouped.keys())
    rows = []
    for tau in taus:
        items = sorted(grouped[tau], key=lambda x: x[0])
        vals = [v for _, v, _ in items]
        rows.append(
            {
                "tau": tau,
                "n": len(vals),
                "mean": _mean(vals),
                "std": _std(vals),
                "min": min(vals),
                "max": max(vals),
                "seeds": [s for s, _, _ in items],
            }
        )

    rows_sorted = sorted(rows, key=lambda r: float(r["mean"]), reverse=True)
    best = rows_sorted[0]
    print(f"[BEST] tau={best['tau']:.10f} mean={best['mean']:.6f} std={best['std']:.6f} n={best['n']} seeds={best['seeds']}")
    for r in rows_sorted:
        print(f"tau={r['tau']:.10f} mean={r['mean']:.6f} std={r['std']:.6f} min={r['min']:.6f} max={r['max']:.6f} n={r['n']}")

    # Plot
    import matplotlib.pyplot as plt

    xs = list(range(len(taus)))
    means = [next(rr for rr in rows if rr["tau"] == t)["mean"] for t in taus]
    mins = [next(rr for rr in rows if rr["tau"] == t)["min"] for t in taus]
    maxs = [next(rr for rr in rows if rr["tau"] == t)["max"] for t in taus]

    plt.figure(figsize=(11, 4.2), dpi=160)
    bars = plt.bar(xs, means, color="#4C78A8", alpha=0.92)

    # Range whiskers (min/max)
    yerr_low = [m - lo for m, lo in zip(means, mins)]
    yerr_hi = [hi - m for m, hi in zip(means, maxs)]
    plt.errorbar(xs, means, yerr=[yerr_low, yerr_hi], fmt="none", ecolor="#222222", elinewidth=1.2, capsize=4)

    metric_label = str(args.metric)
    if metric_label == "weighted":
        metric_label = f"{args.w_max:g}*val_max + {args.w_final:g}*val_final"

    sweep_label = str(args.sweep_key)
    plt.xticks(xs, [f"{t:.4g}" for t in taus], rotation=0)
    plt.xlabel(sweep_label)
    plt.ylabel(metric_label)
    plt.title(f"{sweep_label} sweep ({metric_label}): mean across seeds, with min/max range")
    plt.grid(True, axis="y", alpha=0.25)

    # Highlight best bar
    best_tau = float(best["tau"])
    best_idx = taus.index(best_tau)
    bars[best_idx].set_color("#F58518")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outp.as_posix())
    print("[OK] wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
