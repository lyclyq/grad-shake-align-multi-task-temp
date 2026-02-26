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
#   TRIALS=300
#   RERANK_ENABLED=true
#   RERANK_TOP_K=5
#   RERANK_EPOCHS=6
#   FINAL_EPOCHS=10
#   FINAL_SEEDS='[2,3,5,7,11]'
#   RESUME_DEBUG=/abs/path/to/hpo__...

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNS_GROUP="${RUNS_GROUP:-glue_mrpc_distillbert}"
DATASET="${DATASET:-glue/mrpc}"
MODEL="${MODEL:-distilbert-base-uncased}"
TRIALS="${TRIALS:-300}"
RERANK_ENABLED="${RERANK_ENABLED:-true}"
RERANK_TOP_K="${RERANK_TOP_K:-5}"
RERANK_EPOCHS="${RERANK_EPOCHS:-6}"
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

if [[ -n "$RESUME_DEBUG" ]]; then
  cmd+=(--resume_debug "$RESUME_DEBUG")
fi

echo "[RUN] ${cmd[*]}"
"${cmd[@]}"
