#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_smoke_common.sh"

run_single_task_smoke \
  "smoke_single_roberta_rte" \
  "glue/rte" \
  "roberta-base"
