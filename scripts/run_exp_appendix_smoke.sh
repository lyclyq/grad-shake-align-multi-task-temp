#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_smoke_common.sh"

run_single_task_smoke \
  "smoke_appendix_rank_single_deberta_mrpc_r16_R64" \
  "glue/mrpc" \
  "microsoft/deberta-v3-base" \
  --ours_r 16 \
  --ours_R 64
