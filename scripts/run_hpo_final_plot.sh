#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline runner: HPO -> Final -> Plot
# Usage:
#   conda activate optimization
#   bash scripts/run_hpo_final_plot.sh
#
# Optional env overrides:
#   RUNS_GROUP=glue_mrpc_distillbert
#   DATASET=glue/mrpc
#   MODEL=distilbert-base-uncased
#   MULTI_ENABLED=true
#   MULTI_DATASETS='["glue/mrpc","glue/rte"]'   # or glue/mrpc,glue/rte
#   MULTI_STEPS_MODE=max_steps                  # max_steps / epochs
#   MAX_STEPS=2000
#   MULTI_DROP_LAST=true
#   EPOCHS=6
#   TRIALS=300
#   RERANK_ENABLED=true
#   RERANK_TOP_K=5
#   RERANK_EPOCHS=6
#   HPO_BASELINE_EPOCHS=2
#   HPO_GRID_EPOCHS=3
#   HPO_BASELINE_MAX_STEPS=600
#   HPO_GRID_MAX_STEPS=1200
#   HPO_RERANK_MAX_STEPS=1800
#   FINAL_EPOCHS=10
#   FINAL_SEEDS='[2,3,5,7,11]'
#   RESUME_DEBUG=/abs/path/to/hpo__...

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNS_GROUP="${RUNS_GROUP:-glue_mrpc_distillbert}"
DATASET="${DATASET:-glue/mrpc}"
MODEL="${MODEL:-distilbert-base-uncased}"
MULTI_ENABLED="${MULTI_ENABLED:-}"
MULTI_DATASETS="${MULTI_DATASETS:-}"
MULTI_STEPS_MODE="${MULTI_STEPS_MODE:-}"
MAX_STEPS="${MAX_STEPS:-}"
MULTI_DROP_LAST="${MULTI_DROP_LAST:-}"
EPOCHS="${EPOCHS:-}"
TRIALS="${TRIALS:-300}"
RERANK_ENABLED="${RERANK_ENABLED:-true}"
RERANK_TOP_K="${RERANK_TOP_K:-5}"
RERANK_EPOCHS="${RERANK_EPOCHS:-6}"
HPO_BASELINE_EPOCHS="${HPO_BASELINE_EPOCHS:-}"
HPO_GRID_EPOCHS="${HPO_GRID_EPOCHS:-}"
HPO_BASELINE_MAX_STEPS="${HPO_BASELINE_MAX_STEPS:-}"
HPO_GRID_MAX_STEPS="${HPO_GRID_MAX_STEPS:-}"
HPO_RERANK_MAX_STEPS="${HPO_RERANK_MAX_STEPS:-}"
FINAL_EPOCHS="${FINAL_EPOCHS:-10}"
FINAL_SEEDS="${FINAL_SEEDS:-[2,3,5,7,11]}"
RESUME_DEBUG="${RESUME_DEBUG:-}"

cmd=(
  python scripts/pipeline_oneclick.py
  --runs_group "$RUNS_GROUP"
  --dataset "$DATASET"
  --model "$MODEL"
  --trials "$TRIALS"
  --rerank_enabled "$RERANK_ENABLED"
  --rerank_top_k "$RERANK_TOP_K"
  --rerank_epochs "$RERANK_EPOCHS"
  --final_epochs "$FINAL_EPOCHS"
  --final_seeds "$FINAL_SEEDS"
)

if [[ -n "$MULTI_ENABLED" ]]; then
  cmd+=(--multi_enabled "$MULTI_ENABLED")
fi
if [[ -n "$MULTI_DATASETS" ]]; then
  cmd+=(--multi_datasets "$MULTI_DATASETS")
fi
if [[ -n "$MULTI_STEPS_MODE" ]]; then
  cmd+=(--multi_steps_mode "$MULTI_STEPS_MODE")
fi
if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "$MULTI_DROP_LAST" ]]; then
  cmd+=(--multi_drop_last "$MULTI_DROP_LAST")
fi
if [[ -n "$EPOCHS" ]]; then
  cmd+=(--epochs "$EPOCHS")
fi

if [[ -n "$RESUME_DEBUG" ]]; then
  cmd+=(--resume_debug "$RESUME_DEBUG")
fi
if [[ -n "$HPO_BASELINE_EPOCHS" ]]; then
  cmd+=(--hpo_baseline_epochs "$HPO_BASELINE_EPOCHS")
fi
if [[ -n "$HPO_GRID_EPOCHS" ]]; then
  cmd+=(--hpo_grid_epochs "$HPO_GRID_EPOCHS")
fi
if [[ -n "$HPO_BASELINE_MAX_STEPS" ]]; then
  cmd+=(--hpo_baseline_max_steps "$HPO_BASELINE_MAX_STEPS")
fi
if [[ -n "$HPO_GRID_MAX_STEPS" ]]; then
  cmd+=(--hpo_grid_max_steps "$HPO_GRID_MAX_STEPS")
fi
if [[ -n "$HPO_RERANK_MAX_STEPS" ]]; then
  cmd+=(--hpo_rerank_max_steps "$HPO_RERANK_MAX_STEPS")
fi

echo "[RUN] ${cmd[*]}"
"${cmd[@]}"
