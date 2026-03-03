#!/usr/bin/env python3
# /home/lyclyq/Optimization/grad-shake-align/scripts/plot_final_4lines_abs.py

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Curve = Tuple[np.ndarray, np.ndarray]


def read_epoch_curve(metrics_csv: Path, ykey: str) -> Tuple[Optional[Curve], str]:
    df = pd.read_csv(metrics_csv)

    if "epoch" not in df.columns:
        return None, f"[MISS] no 'epoch' col in {metrics_csv}"
    if ykey not in df.columns:
        return None, f"[MISS] no '{ykey}' in {metrics_csv}"

    ep = pd.to_numeric(df["epoch"], errors="coerce")
    y = pd.to_numeric(df[ykey], errors="coerce")
    m = ep.notna() & y.notna()
    if not m.any():
        return None, f"[MISS] epoch rows exist but '{ykey}' all NaN in {metrics_csv}"

    d = pd.DataFrame({"x": ep[m].astype(int), "y": y[m].astype(float)})
    d = d.sort_values("x").drop_duplicates(subset=["x"], keep="last")
    return (d["x"].to_numpy(dtype=int), d["y"].to_numpy(dtype=float)), f"[OK] {metrics_csv} points={len(d)}"


def read_step_curve(metrics_csv: Path, ykey: str, max_epoch: Optional[int] = 2) -> Tuple[Optional[Curve], str]:
    df = pd.read_csv(metrics_csv)

    if "step" not in df.columns:
        return None, f"[MISS] no 'step' col in {metrics_csv}"
    if ykey not in df.columns:
        return None, f"[MISS] no '{ykey}' in {metrics_csv}"

    x = pd.to_numeric(df["step"], errors="coerce")
    y = pd.to_numeric(df[ykey], errors="coerce")
    m = x.notna() & y.notna()

    if max_epoch is not None and "epoch" in df.columns:
        ep = pd.to_numeric(df["epoch"], errors="coerce")
        m = m & ep.notna() & (ep <= int(max_epoch))

    if not m.any():
        return None, f"[MISS] no valid step points for '{ykey}' in {metrics_csv}"

    d = pd.DataFrame({"x": x[m].astype(int), "y": y[m].astype(float)})
    d = d.sort_values("x").drop_duplicates(subset=["x"], keep="last")
    return (d["x"].to_numpy(dtype=int), d["y"].to_numpy(dtype=float)), f"[OK] {metrics_csv} points={len(d)}"


def read_step_curve_any(
    metrics_csv: Path,
    ykeys: Sequence[str],
    max_epoch: Optional[int] = 2,
) -> Tuple[Optional[Curve], str]:
    last_msg = f"[MISS] no keys for {metrics_csv}"
    for yk in ykeys:
        out, msg = read_step_curve(metrics_csv, yk, max_epoch=max_epoch)
        if out is not None:
            return out, f"{msg} key={yk}"
        last_msg = msg
    return None, f"{last_msg}; tried={list(ykeys)}"


def collect_variant_curves(
    trial_runs_dir: Path,
    variant: str,
    reader: Callable[[Path], Tuple[Optional[Curve], str]],
    *,
    verbose: bool = True,
) -> List[Curve]:
    vdir = trial_runs_dir / variant
    files = sorted(vdir.glob("s*/metrics.csv"))
    if verbose:
        print(f"\n== {variant} ==")
        print(f"dir: {vdir}")
        print(f"found {len(files)} files")

    curves: List[Curve] = []
    for f in files:
        out, msg = reader(f)
        if verbose:
            print(msg)
        if out is not None:
            curves.append(out)
    return curves


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


def mean_std_plot(x: np.ndarray, ys: np.ndarray, label: str, *, marker: Optional[str] = "o") -> None:
    mu = ys.mean(axis=0)
    sd = ys.std(axis=0)
    if marker is None:
        plt.plot(x, mu, label=label)
    else:
        plt.plot(x, mu, marker=marker, label=label)
    plt.fill_between(x, mu - sd, mu + sd, alpha=0.18)


def _plot_4lines_epoch(tr: Path, metric: str) -> None:
    plt.figure()

    curves = collect_variant_curves(tr, "baseline_r", lambda p: read_epoch_curve(p, metric), verbose=True)
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, f"baseline_r ({metric})")

    curves = collect_variant_curves(tr, "baseline_R", lambda p: read_epoch_curve(p, metric), verbose=True)
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, f"baseline_R ({metric})")

    curves = collect_variant_curves(tr, "ours", lambda p: read_epoch_curve(p, metric), verbose=True)
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, f"ours full ({metric})")

    r_only_key = metric.replace("/acc", "/acc_r_only")
    curves = collect_variant_curves(tr, "ours", lambda p: read_epoch_curve(p, r_only_key), verbose=True)
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, f"ours r-only ({r_only_key})")

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title("Final comparison (mean±std over seeds)")
    plt.legend()
    plt.tight_layout()

    out_dir = tr / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compare_4lines_{metric.replace('/','_')}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"\n[OK] saved figure -> {out_path}")


def _plot_4lines_dense(tr: Path, *, y_full: str, y_r_only: str, title: str, out_name: str, max_epoch: int = 2) -> None:
    plt.figure()
    plotted = False

    curves = collect_variant_curves(tr, "baseline_r", lambda p: read_step_curve(p, y_full, max_epoch=max_epoch), verbose=False)
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, f"baseline_r ({y_full})", marker=None)
        plotted = True

    curves = collect_variant_curves(tr, "baseline_R", lambda p: read_step_curve(p, y_full, max_epoch=max_epoch), verbose=False)
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, f"baseline_R ({y_full})", marker=None)
        plotted = True

    curves = collect_variant_curves(tr, "ours", lambda p: read_step_curve(p, y_full, max_epoch=max_epoch), verbose=False)
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, f"ours full ({y_full})", marker=None)
        plotted = True

    curves = collect_variant_curves(tr, "ours", lambda p: read_step_curve(p, y_r_only, max_epoch=max_epoch), verbose=False)
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, f"ours r-only ({y_r_only})", marker=None)
        plotted = True

    plt.xlabel("global step")
    plt.ylabel(y_full)
    plt.title(title)
    if plotted:
        plt.legend()
    else:
        print(f"[WARN] skip legend: no valid curves for dense plot '{out_name}'")
    plt.tight_layout()

    out_dir = tr / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] saved figure -> {out_path}")


def _plot_ours_gate_and_pull_dense(tr: Path, *, max_epoch: int = 2) -> None:
    # gate0 trigger rate (fallback to triggered blocks for old logs)
    plt.figure()
    plotted_gate = False
    curves = collect_variant_curves(
        tr,
        "ours",
        lambda p: read_step_curve_any(
            p,
            ["train/gate0_trigger_rate", "train/gate0_triggered_blocks"],
            max_epoch=max_epoch,
        ),
        verbose=False,
    )
    x, ys = align_and_stack(curves)
    if x is not None:
        mean_std_plot(x, ys, "gate0 trigger rate", marker=None)
        plotted_gate = True
    plt.xlabel("global step")
    plt.ylabel("rate")
    plt.title(f"Ours Gate0 Activation (dense, first {max_epoch} epochs)")
    if plotted_gate:
        plt.legend()
    else:
        print("[WARN] skip legend: no valid ours gate0 dense curves")
    plt.tight_layout()
    out_dir = tr / "_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ours_gate0_trigger_rate_dense.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] saved figure -> {out_path}")

    # pull direction (rate preferred, fallback to block-count for old logs)
    plt.figure()
    plotted_pull = False
    curves_r = collect_variant_curves(
        tr,
        "ours",
        lambda p: read_step_curve_any(
            p,
            ["train/pull_to_r_rate", "train/pull_to_r_blocks"],
            max_epoch=max_epoch,
        ),
        verbose=False,
    )
    x_r, ys_r = align_and_stack(curves_r)
    if x_r is not None:
        mean_std_plot(x_r, ys_r, "pull toward r rate", marker=None)
        plotted_pull = True

    curves_R = collect_variant_curves(
        tr,
        "ours",
        lambda p: read_step_curve_any(
            p,
            ["train/pull_to_R_rate", "train/pull_to_R_blocks"],
            max_epoch=max_epoch,
        ),
        verbose=False,
    )
    x_R, ys_R = align_and_stack(curves_R)
    if x_R is not None:
        mean_std_plot(x_R, ys_R, "pull toward R rate", marker=None)
        plotted_pull = True

    plt.xlabel("global step")
    plt.ylabel("rate")
    plt.title(f"Ours Pull Direction (dense, first {max_epoch} epochs)")
    if plotted_pull:
        plt.legend()
    else:
        print("[WARN] skip legend: no valid ours pull-direction dense curves")
    plt.tight_layout()
    out_path = out_dir / "ours_pull_direction_dense.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] saved figure -> {out_path}")

    # pull strength
    plt.figure()
    plotted_alpha = False
    curves_alpha = collect_variant_curves(
        tr,
        "ours",
        lambda p: read_step_curve(p, "train/alpha_pull_mean", max_epoch=max_epoch),
        verbose=False,
    )
    x_a, ys_a = align_and_stack(curves_alpha)
    if x_a is not None:
        mean_std_plot(x_a, ys_a, "alpha pull mean", marker=None)
        plotted_alpha = True

    plt.xlabel("global step")
    plt.ylabel("alpha")
    plt.title(f"Ours Pull Strength (dense, first {max_epoch} epochs)")
    if plotted_alpha:
        plt.legend()
    else:
        print("[WARN] skip legend: no valid ours pull-strength dense curves")
    plt.tight_layout()
    out_path = out_dir / "ours_pull_strength_alpha_dense.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] saved figure -> {out_path}")


def plot_4lines(trial_runs_dir: str, metric: str = "val/acc") -> None:
    tr = Path(trial_runs_dir).expanduser().resolve()
    assert tr.exists(), f"trial_runs_dir not found: {tr}"
    print(f"[INFO] trial_runs_dir = {tr}")

    _plot_4lines_epoch(tr, metric)
    _plot_4lines_dense(
        tr,
        y_full="val/acc",
        y_r_only="val/acc_r_only",
        title="Dense Val Acc (first 2 epochs, mean±std)",
        out_name="compare_4lines_dense_val_acc_first2ep.png",
        max_epoch=2,
    )
    _plot_4lines_dense(
        tr,
        y_full="train/acc",
        y_r_only="train/acc_r_only",
        title="Dense Train Acc (first 2 epochs, mean±std)",
        out_name="compare_4lines_dense_train_acc_first2ep.png",
        max_epoch=2,
    )
    _plot_4lines_dense(
        tr,
        y_full="train/loss_eval",
        y_r_only="train/loss_r_only_eval",
        title="Dense Train Loss (first 2 epochs, mean±std)",
        out_name="compare_4lines_dense_train_loss_first2ep.png",
        max_epoch=2,
    )
    _plot_ours_gate_and_pull_dense(tr, max_epoch=2)


if __name__ == "__main__":
    import sys

    trial_runs = "/home/lyclyq/Optimization/grad-shake-align/runs/final__glue_rte__bert-base-uncased__ours__ep7__bs32__lr8.046330436316312e-05__final__20260129-164311__1b79c799/trial_runs"
    metric_name = "val/acc"
    if len(sys.argv) >= 2:
        trial_runs = sys.argv[1]
    if len(sys.argv) >= 3:
        metric_name = sys.argv[2]
    plot_4lines(trial_runs, metric_name)
