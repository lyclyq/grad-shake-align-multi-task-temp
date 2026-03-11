#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_smoke_common.sh"

MIX3='["glue/rte","glue/mrpc","glue/cola"]'

run_multi_source_smoke \
  "multi_task" \
  "smoke_multi_roberta_mix3" \
  "roberta-base" \
  "$MIX3"
