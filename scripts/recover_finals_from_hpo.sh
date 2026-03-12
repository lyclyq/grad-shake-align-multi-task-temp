#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON_BIN:-/home/lyclyq/miniconda3/envs/optimization/bin/python}"
LOGROOT="${LOGROOT:-$ROOT/runs/_recover_logs/final_recover_from_hpo_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOGROOT"

TARGET_GROUPS=(
  glue_rte_roberta
  glue_rte_distillbert
  glue_rte_deberta
  glue_mrpc_bert
  glue_mrpc_roberta
  glue_mrpc_distillbert_hpo4_rerank10e6
  glue_rte_bert
)

resolve_task_model() {
  local grp="$1"
  case "$grp" in
    glue_rte_roberta)
      echo "glue/rte roberta-base"
      ;;
    glue_rte_distillbert)
      echo "glue/rte distilbert-base-uncased"
      ;;
    glue_rte_deberta)
      echo "glue/rte microsoft/deberta-v3-base"
      ;;
    glue_rte_bert)
      echo "glue/rte bert-base-uncased"
      ;;
    glue_mrpc_bert)
      echo "glue/mrpc bert-base-uncased"
      ;;
    glue_mrpc_roberta)
      echo "glue/mrpc roberta-base"
      ;;
    glue_mrpc_distillbert_hpo4_rerank10e6)
      echo "glue/mrpc distilbert-base-uncased"
      ;;
    *)
      return 1
      ;;
  esac
}

pick_latest_final() {
  local grp="$1"
  ls -td "runs/${grp}/final"* 2>/dev/null | head -n 1 || true
}

pick_latest_hpo_best() {
  local grp="$1"
  local hpo_dir
  hpo_dir=$(
    find "runs/${grp}" -maxdepth 1 -type d \( -name 'hpo__*' -o -name 'debug__*' \) \
      -print0 2>/dev/null \
      | xargs -0 -r ls -td 2>/dev/null \
      | while read -r d; do
          if [[ -f "$d/best_hparams.json" ]]; then
            echo "$d/best_hparams.json"
            break
          fi
        done
  )
  echo "${hpo_dir:-}"
}

build_best_compat_if_needed() {
  local best_in="$1"
  local best_out="$2"
  "$PY" - <<'PY' "$best_in" "$best_out"
import csv
import json
import sys
from pathlib import Path

best_in = Path(sys.argv[1])
best_out = Path(sys.argv[2])

obj = json.loads(best_in.read_text(encoding="utf-8"))
best = obj.get("best", {})
trial_cfg = json.loads(best.get("trial_cfg_json", "{}"))
ours = ((trial_cfg.get("method", {}) or {}).get("ours", {}) or {})

has_new_lora = isinstance(ours.get("lora"), dict)
has_base_lr = isinstance(obj.get("baseline_best_lr_refined"), dict)
has_plan = bool(obj.get("plan_path"))

if has_new_lora and has_base_lr and has_plan:
    best_out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compat] passthrough: {best_out}")
    raise SystemExit(0)

hpo_dir = obj.get("hpo_dir", "")
if not hpo_dir:
    # fallback from best path layout .../hpo__/best_hparams.json
    hpo_dir = str(best_in.parent)
hpo_dir = Path(hpo_dir)
trials_csv = hpo_dir / "trials.csv"
plan_path = obj.get("plan_path") or str(hpo_dir / "grid_plan.json")

if not trials_csv.exists():
    raise RuntimeError(f"[compat] missing trials.csv: {trials_csv}")

rows = list(csv.DictReader(trials_csv.open("r", encoding="utf-8")))

def _best_lr(stage_name: str) -> float:
    cand = [r for r in rows if (r.get("stage") or "") == stage_name]
    if not cand:
        raise RuntimeError(f"[compat] no rows for stage={stage_name} in {trials_csv}")
    b = max(cand, key=lambda r: float(r.get("score") or "-inf"))
    tc = json.loads(b["trial_cfg_json"])
    return float(tc["train"]["lr"])

baseline_lr_r = _best_lr("baseline.baseline_r")
baseline_lr_R = _best_lr("baseline.baseline_R")

ours_cfg = dict(ours)

# old key -> new key
ng = ours_cfg.pop("noise_gate", None)
if isinstance(ng, dict) and "gate0_noise" not in ours_cfg:
    ours_cfg["gate0_noise"] = {
        "tau": float(ng.get("tau_n", -10.0)),
        "kappa": float(ng.get("kappa_n", 10.0)),
    }

if "voting" not in ours_cfg and "votes" in ours_cfg:
    ours_cfg["voting"] = {
        "samples_per_vote": int(ours_cfg.get("votes", 4)),
        "allow_tail": True,
        "keep_single_votes": True,
    }

# Fill required lora subtree for current final.py.
if "lora" not in ours_cfg:
    ours_cfg["lora"] = {
        "r": 32,
        "R": 128,
        "alpha": 16.0,
        "dropout": 0.1,
    }

ours_lr = float((trial_cfg.get("train", {}) or {}).get("lr"))

new_trial_cfg = {
    "method": {
        "name": "ours",
        "ours": ours_cfg,
    },
    "train": {
        "lr": ours_lr,
    },
}

new_obj = dict(obj)
new_obj["best"] = dict(best)
new_obj["best"]["trial_cfg_json"] = json.dumps(new_trial_cfg, sort_keys=True)
new_obj["baseline_best_lr_refined"] = {
    "baseline_r": baseline_lr_r,
    "baseline_R": baseline_lr_R,
}
new_obj["plan_path"] = str(plan_path)
new_obj["hpo_dir"] = str(hpo_dir)

best_out.write_text(json.dumps(new_obj, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[compat] rebuilt: {best_out}")
PY
}

run_group() {
  local grp="$1"
  local final_dir best_src best_tmp task_name model_name

  final_dir="$(pick_latest_final "$grp")"
  if [[ -z "$final_dir" ]]; then
    echo "[MISS] ${grp}: no final dir"
    return 1
  fi

  best_src="$(pick_latest_hpo_best "$grp")"
  if [[ -z "$best_src" || ! -f "$best_src" ]]; then
    echo "[MISS] ${grp}: no best_hparams under hpo/debug"
    return 1
  fi

  best_tmp="$LOGROOT/${grp}_best_hparams_compat.json"
  build_best_compat_if_needed "$best_src" "$best_tmp"

  if ! read -r task_name model_name < <(resolve_task_model "$grp"); then
    echo "[MISS] ${grp}: unknown task/model mapping"
    return 1
  fi

  echo "[RUN] ${grp}"
  echo "  final_dir=${final_dir}"
  echo "  best_from_hpo=${best_src}"
  echo "  best_used=${best_tmp}"
  echo "  task=${task_name}"
  echo "  model=${model_name}"

  "$PY" scripts/run.py final \
    --config configs/base.yaml \
    --schedule configs/schedules/final.yaml \
    --best "$best_tmp" \
    --set "task.name=${task_name}" \
    --set "model.name=${model_name}" \
    --set "io.run_dir=${final_dir}" \
    --set "io.overwrite=force" \
    --set "train.eval.strategy=dense_early" \
    --set "train.eval.dense_early_per_epoch=8" \
    --set "train.eval.dense_early_epochs=2" \
    >"$LOGROOT/${grp}_final.log" 2>&1

  "$PY" scripts/run.py plot \
    --config configs/base.yaml \
    --runs_dir "$final_dir" \
    >"$LOGROOT/${grp}_plot_compare.log" 2>&1

  "$PY" scripts/plot_final_4lines_abs.py "$final_dir/trial_runs" val/acc \
    >"$LOGROOT/${grp}_plot_4lines.log" 2>&1

  echo "[OK] ${grp}"
}

fail=0
for grp in "${TARGET_GROUPS[@]}"; do
  if ! run_group "$grp"; then
    fail=1
    echo "[FAIL] ${grp} (see logs under $LOGROOT)"
  fi
done

echo "[DONE] LOGROOT=${LOGROOT}"
exit "$fail"
