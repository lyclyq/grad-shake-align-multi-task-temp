#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_smoke_common.sh"

# Force the smallest useful smoke budget unless the caller overrides it.
export SMOKE_TRIALS="${SMOKE_TRIALS:-1}"
export SMOKE_HPO_STEPS="${SMOKE_HPO_STEPS:-1}"
export SMOKE_FINAL_STEPS="${SMOKE_FINAL_STEPS:-1}"
export SMOKE_HPO_SEEDS="${SMOKE_HPO_SEEDS:-[1]}"
export SMOKE_FINAL_SEEDS="${SMOKE_FINAL_SEEDS:-[1]}"
export SMOKE_COORD_TOP_K="${SMOKE_COORD_TOP_K:-0}"
export SMOKE_REFINE_RADIX="${SMOKE_REFINE_RADIX:-1}"
export SMOKE_EVAL_MAX_BATCHES="${SMOKE_EVAL_MAX_BATCHES:-1}"
export SMOKE_COMPUTE_TRAIN_ACC="${SMOKE_COMPUTE_TRAIN_ACC:-false}"
export SMOKE_COMPILE="${SMOKE_COMPILE:-false}"
export SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"

MIX3='["glue/rte","glue/mrpc","glue/cola"]'
DATASETS='["glue/sst2","yelp_polarity","amazon_polarity"]'

echo "[SMOKE] single-task main"
run_single_task_smoke \
  "smoke_suite_main_single_roberta_rte" \
  "glue/rte" \
  "roberta-base"

echo "[SMOKE] multi-task main"
run_multi_source_smoke \
  "multi_task" \
  "smoke_suite_main_multi_roberta_mix3" \
  "roberta-base" \
  "$MIX3"

echo "[SMOKE] appendix single-task"
run_single_task_smoke \
  "smoke_suite_appendix_rank_single_deberta_mrpc_r16_R64" \
  "glue/mrpc" \
  "microsoft/deberta-v3-base" \
  --ours_r 16 \
  --ours_R 64

echo "[SMOKE] appendix multi-task"
run_multi_source_smoke \
  "multi_task" \
  "smoke_suite_appendix_rank_multi_deberta_mix3_r16_R64" \
  "microsoft/deberta-v3-base" \
  "$MIX3" \
  --ours_r 16 \
  --ours_R 64

echo "[SMOKE] extra multi-dataset"
run_multi_source_smoke \
  "multi_dataset" \
  "smoke_suite_extra_multidataset_deberta_sentiment" \
  "microsoft/deberta-v3-base" \
  "$DATASETS"

echo "[SMOKE] ablation resume"
run_multi_task_ablation_smoke_resume \
  "smoke_suite_main_multi_roberta_mix3" \
  "roberta-base" \
  "$MIX3"

echo "[SMOKE] all smoke experiments finished"
