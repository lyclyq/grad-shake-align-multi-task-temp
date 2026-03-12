#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./smoke.sh <target>

Targets:
  single    Run single-task smoke
  multi     Run multi-task smoke
  appendix  Run appendix smoke
  extra     Run extra multi-dataset smoke
  ablation  Run ablation smoke (requires multi smoke outputs)
  all       Run all smoke suites
  clean     Remove runs/smoke_* directories
  help      Show this message
EOF
}

target="${1:-help}"

case "$target" in
  single)
    exec bash "$ROOT/scripts/run_exp_single_task_smoke.sh"
    ;;
  multi)
    exec bash "$ROOT/scripts/run_exp_multi_task_smoke.sh"
    ;;
  appendix)
    exec bash "$ROOT/scripts/run_exp_appendix_smoke.sh"
    ;;
  extra)
    exec bash "$ROOT/scripts/run_exp_extra_smoke.sh"
    ;;
  ablation)
    exec bash "$ROOT/scripts/run_exp_ablation_smoke.sh"
    ;;
  all)
    exec bash "$ROOT/scripts/run_exp_all_smoke.sh"
    ;;
  clean)
    find "$ROOT/runs" -maxdepth 1 -mindepth 1 -type d -name 'smoke_*' -print -exec rm -rf {} +
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[ERR] Unknown target: $target" >&2
    usage >&2
    exit 1
    ;;
esac
