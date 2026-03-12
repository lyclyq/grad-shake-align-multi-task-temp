#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_protocol_common.sh"

MIX3='["glue/rte","glue/mrpc","glue/cola"]'

# Rank robustness: single-task MRPC on DeBERTa.
run_single_task_protocol \
  "paper_suite_appendix_rank_single_deberta_mrpc_r16_R64" \
  "glue/mrpc" \
  "microsoft/deberta-v3-base" \
  100 \
  8 \
  16 \
  64

run_single_task_protocol \
  "paper_suite_appendix_rank_single_deberta_mrpc_r32_R128" \
  "glue/mrpc" \
  "microsoft/deberta-v3-base" \
  100 \
  8 \
  32 \
  128

# Rank robustness: 3-task mixture on DeBERTa.
run_multi_source_protocol \
  "multi_task" \
  "paper_suite_appendix_rank_multi_deberta_mix3_r16_R64" \
  "microsoft/deberta-v3-base" \
  "$MIX3" \
  80 \
  800 \
  8 \
  16 \
  64

run_multi_source_protocol \
  "multi_task" \
  "paper_suite_appendix_rank_multi_deberta_mix3_r32_R128" \
  "microsoft/deberta-v3-base" \
  "$MIX3" \
  80 \
  800 \
  8 \
  32 \
  128
