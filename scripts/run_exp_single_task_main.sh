#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Backbone weights are loaded from HF pretrained checkpoints.
# Task heads and LoRA adapters are still initialized at run time by the model code.
source "$ROOT/scripts/exp_protocol_common.sh"

run_single_task_protocol "main_single_roberta_rte" "glue/rte" "roberta-base"
run_single_task_protocol "main_single_roberta_mrpc" "glue/mrpc" "roberta-base"
run_single_task_protocol "main_single_roberta_sst2" "glue/sst2" "roberta-base"

run_single_task_protocol "main_single_deberta_rte" "glue/rte" "microsoft/deberta-v3-base"
run_single_task_protocol "main_single_deberta_mrpc" "glue/mrpc" "microsoft/deberta-v3-base"
run_single_task_protocol "main_single_deberta_sst2" "glue/sst2" "microsoft/deberta-v3-base"
