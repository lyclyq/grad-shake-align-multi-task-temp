#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_protocol_common.sh"

MIX3='["glue/rte","glue/mrpc","glue/cola"]'
MIX4='["glue/rte","glue/mrpc","glue/cola","glue/qnli"]'

# This script reuses the latest HPO/final dirs under each runs_group and only
# adds ablation runs. Existing baseline/cagrad/ours finals are skipped by resume.
run_multi_task_ablation_resume "main_multi_roberta_mix3" "roberta-base" "$MIX3"
run_multi_task_ablation_resume "main_multi_roberta_mix4" "roberta-base" "$MIX4"

run_multi_task_ablation_resume "main_multi_deberta_mix3" "microsoft/deberta-v3-base" "$MIX3"
run_multi_task_ablation_resume "main_multi_deberta_mix4" "microsoft/deberta-v3-base" "$MIX4"

run_multi_task_ablation_resume "main_multi_gpt2_mix3" "gpt2-medium" "$MIX3"
run_multi_task_ablation_resume "main_multi_gpt2_mix4" "gpt2-medium" "$MIX4"
