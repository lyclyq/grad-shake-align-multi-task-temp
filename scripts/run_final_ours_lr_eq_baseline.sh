#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lyclyq/Optimization/grad-shake-align"
SRC_FINAL="$ROOT/runs/glue_rte_roberta/final__glue-rte-roberta__from_20260214-110919__f2325378"
DST_FINAL="$ROOT/runs/glue_rte_roberta/final__glue-rte-roberta__from_20260214-110919__f2325378__ourslr_baseline__20260215-192148"
BEST_PATCHED="$DST_FINAL/best_hparams_ours_lr_eq_baseline.json"
PY="/home/lyclyq/miniconda3/envs/optimization/bin/python"
export PATH="/home/lyclyq/miniconda3/envs/optimization/bin:$PATH"

cd "$ROOT"

export HF_HOME="$ROOT/.hf_cache"
export HF_DATASETS_CACHE="$ROOT/.hf_cache/datasets"
export TRANSFORMERS_CACHE="$ROOT/.hf_cache/transformers"

"$PY" scripts/run.py final \
  --config configs/base.yaml \
  --schedule configs/schedules/final.yaml \
  --best "$BEST_PATCHED" \
  --set "io.run_dir=$DST_FINAL" \
  --set "io.overwrite=resume"

"$PY" - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/lyclyq/Optimization/grad-shake-align")
src = ROOT / "runs/glue_rte_roberta/final__glue-rte-roberta__from_20260214-110919__f2325378"
dst = ROOT / "runs/glue_rte_roberta/final__glue-rte-roberta__from_20260214-110919__f2325378__ourslr_baseline__20260215-192148"

def read_curve(csv_path: Path, key: str = "val/acc"):
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or key not in df.columns:
        return None
    ep = pd.to_numeric(df["epoch"], errors="coerce")
    y = pd.to_numeric(df[key], errors="coerce")
    mask = ep.notna() & y.notna()
    if not mask.any():
        return None
    d = pd.DataFrame({"epoch": ep[mask].astype(int), "y": y[mask].astype(float)})
    d = d.sort_values("epoch").drop_duplicates("epoch", keep="last")
    return d["epoch"].to_numpy(), d["y"].to_numpy()

def collect(base_final: Path, variant: str):
    out = []
    for f in sorted((base_final / "trial_runs" / variant).glob("s*/metrics.csv")):
        c = read_curve(f, "val/acc")
        if c is not None:
            out.append(c)
    return out

def align(curves):
    if not curves:
        return None, None
    common = None
    for e, _ in curves:
        s = set(map(int, e.tolist()))
        common = s if common is None else (common & s)
    if not common:
        return None, None
    epochs = np.array(sorted(common), dtype=int)
    ys = []
    for e, y in curves:
        mp = {int(a): float(b) for a, b in zip(e, y)}
        ys.append([mp[int(t)] for t in epochs])
    return epochs, np.asarray(ys, dtype=float)

plt.figure(figsize=(7, 5))
series = [
    ("baseline_r (existing final)", collect(src, "baseline_r")),
    ("baseline_R (existing final)", collect(src, "baseline_R")),
    ("ours full (lr=baseline lr)", collect(dst, "ours")),
]
for label, curves in series:
    ep, ys = align(curves)
    if ep is None:
        continue
    mu = ys.mean(axis=0)
    sd = ys.std(axis=0)
    plt.plot(ep, mu, marker="o", label=label)
    plt.fill_between(ep, mu - sd, mu + sd, alpha=0.18)

plt.xlabel("epoch")
curves = []
for f in sorted((dst / "trial_runs" / "ours").glob("s*/metrics.csv")):
    c = read_curve(f, "val/acc_r_only")
    if c is not None:
        curves.append(c)
ep, ys = align(curves)
if ep is not None:
    mu = ys.mean(axis=0)
    sd = ys.std(axis=0)
    plt.plot(ep, mu, marker="o", label="ours r-only (lr=baseline lr)")
    plt.fill_between(ep, mu - sd, mu + sd, alpha=0.18)

plt.ylabel("val/acc")
plt.title("Final: baselines vs ours(full+r-only, lr=baseline)")
plt.legend()
plt.tight_layout()

out = dst / "_plots" / "compare_baselines_vs_ours_full_and_r_only_lr_eq_baseline_val_acc.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=180)
print(f"[OK] plot saved: {out}")
PY
