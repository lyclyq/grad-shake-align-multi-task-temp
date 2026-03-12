#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE="$ROOT/scripts/pipeline_oneclick.py"
TRIALS="${TRIALS:-96}"
HPO_SEEDS="${HPO_SEEDS:-[2,3]}"
FINAL_SEEDS="${FINAL_SEEDS:-[2,3,5,7,11]}"

run_single_task_protocol() {
  local runs_group="$1"
  local dataset="$2"
  local model="$3"
  local hpo_steps="${4:-100}"
  local final_epochs="${5:-8}"
  local ours_r="${6:-32}"
  local ours_R="${7:-128}"
  shift 7 || true
  local extra=( "$@" )

  "$PYTHON_BIN" "$PIPELINE" \
    --runs_group "$runs_group" \
    --scenario single_task \
    --dataset "$dataset" \
    --model "$model" \
    --trials "$TRIALS" \
    --hpo_baseline_epochs 1 \
    --hpo_grid_epochs 1 \
    --rerank_epochs 1 \
    --hpo_baseline_max_steps "$hpo_steps" \
    --hpo_grid_max_steps "$hpo_steps" \
    --hpo_rerank_max_steps "$hpo_steps" \
    --epochs "$final_epochs" \
    --final_epochs "$final_epochs" \
    --final_seeds "$FINAL_SEEDS" \
    --set "hpo.bandit.refine_seeds=$HPO_SEEDS" \
    --set "hpo.grid.rerank.seeds=$HPO_SEEDS" \
    --ours_r "$ours_r" \
    --ours_R "$ours_R" \
    "${extra[@]}"
}

run_multi_source_protocol() {
  local scenario="$1"
  local runs_group="$2"
  local model="$3"
  local datasets_json="$4"
  local hpo_steps="${5:-80}"
  local final_steps="${6:-800}"
  local virtual_epochs="${7:-8}"
  local ours_r="${8:-32}"
  local ours_R="${9:-128}"
  shift 9 || true
  local extra=( "$@" )

  "$PYTHON_BIN" "$PIPELINE" \
    --runs_group "$runs_group" \
    --scenario "$scenario" \
    --multi_enabled true \
    --multi_datasets "$datasets_json" \
    --model "$model" \
    --trials "$TRIALS" \
    --multi_steps_mode max_steps \
    --max_steps "$final_steps" \
    --epochs "$virtual_epochs" \
    --final_epochs "$virtual_epochs" \
    --final_seeds "$FINAL_SEEDS" \
    --hpo_baseline_epochs 1 \
    --hpo_grid_epochs 1 \
    --rerank_epochs 1 \
    --hpo_baseline_max_steps "$hpo_steps" \
    --hpo_grid_max_steps "$hpo_steps" \
    --hpo_rerank_max_steps "$hpo_steps" \
    --set "hpo.bandit.refine_seeds=$HPO_SEEDS" \
    --set "hpo.grid.rerank.seeds=$HPO_SEEDS" \
    --ours_r "$ours_r" \
    --ours_R "$ours_R" \
    "${extra[@]}"
}

latest_matching_dir() {
  local parent="$1"
  local prefix="$2"
  if [[ ! -d "$parent" ]]; then
    return 1
  fi
  find "$parent" -maxdepth 1 -mindepth 1 -type d -name "${prefix}*" | sort | tail -n 1
}

resolve_runs_group_dir() {
  local runs_group="$1"
  local datasets_json="${2:-}"
  if [[ -z "$datasets_json" ]]; then
    printf '%s\n' "$ROOT/runs/$runs_group"
    return 0
  fi
  local ds_token
  ds_token="$("$PYTHON_BIN" -c 'import json,sys; xs=json.loads(sys.argv[1]); print("+".join(str(x).strip().replace("/", "_") for x in xs if str(x).strip()))' "$datasets_json")"
  if [[ -z "$ds_token" ]]; then
    printf '%s\n' "$ROOT/runs/$runs_group"
    return 0
  fi
  printf '%s\n' "$ROOT/runs/${runs_group}__${ds_token}"
}

run_multi_task_ablation_resume() {
  local runs_group="$1"
  local model="$2"
  local datasets_json="$3"
  local hpo_steps="${4:-80}"
  local final_steps="${5:-800}"
  local virtual_epochs="${6:-8}"
  local ours_r="${7:-32}"
  local ours_R="${8:-128}"
  local resume_debug="${9:-}"
  local resume_final="${10:-}"
  local runs_parent
  runs_parent="$(resolve_runs_group_dir "$runs_group" "$datasets_json")"

  if [[ -z "$resume_debug" ]]; then
    resume_debug="$(latest_matching_dir "$runs_parent" "hpo__" || true)"
  fi
  if [[ -z "$resume_final" ]]; then
    resume_final="$(latest_matching_dir "$runs_parent" "final__" || true)"
  fi

  if [[ -z "$resume_debug" || ! -d "$resume_debug" ]]; then
    echo "[ERR] Missing resume_debug HPO dir for $runs_group" >&2
    return 1
  fi
  if [[ -z "$resume_final" || ! -d "$resume_final" ]]; then
    echo "[ERR] Missing resume_final dir for $runs_group" >&2
    return 1
  fi

  run_multi_source_protocol \
    "multi_task" \
    "$runs_group" \
    "$model" \
    "$datasets_json" \
    "$hpo_steps" \
    "$final_steps" \
    "$virtual_epochs" \
    "$ours_r" \
    "$ours_R" \
    --resume_debug "$resume_debug" \
    --resume_final "$resume_final" \
    --with_ablations
}
