#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_smoke_common.sh"

MIX3='["glue/rte","glue/mrpc","glue/cola"]'

# Reuse the multi-task smoke run and only append ablation methods.
run_multi_task_ablation_smoke_resume \
  "smoke_multi_roberta_mix3" \
  "roberta-base" \
  "$MIX3"
