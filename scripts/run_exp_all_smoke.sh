#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

bash "$ROOT/scripts/run_exp_single_task_smoke.sh"
bash "$ROOT/scripts/run_exp_multi_task_smoke.sh"
bash "$ROOT/scripts/run_exp_appendix_smoke.sh"
bash "$ROOT/scripts/run_exp_extra_smoke.sh"
bash "$ROOT/scripts/run_exp_ablation_smoke.sh"
