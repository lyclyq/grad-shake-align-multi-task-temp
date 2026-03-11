#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE="$ROOT/scripts/pipeline_oneclick.py"

SMOKE_TRIALS="${SMOKE_TRIALS:-2}"
SMOKE_HPO_STEPS="${SMOKE_HPO_STEPS:-5}"
SMOKE_FINAL_STEPS="${SMOKE_FINAL_STEPS:-5}"
SMOKE_HPO_SEEDS="${SMOKE_HPO_SEEDS:-[1,2]}"
SMOKE_FINAL_SEEDS="${SMOKE_FINAL_SEEDS:-[2]}"

run_single_task_smoke() {
  local runs_group="$1"
  local dataset="$2"
  local model="$3"
  shift 3 || true
  local extra=( "$@" )

  "$PYTHON_BIN" "$PIPELINE" \
    --runs_group "$runs_group" \
    --scenario single_task \
    --dataset "$dataset" \
    --model "$model" \
    --trials "$SMOKE_TRIALS" \
    --epochs 1 \
    --final_epochs 1 \
    --max_steps "$SMOKE_FINAL_STEPS" \
    --hpo_baseline_epochs 1 \
    --hpo_grid_epochs 1 \
    --rerank_epochs 1 \
    --hpo_baseline_max_steps "$SMOKE_HPO_STEPS" \
    --hpo_grid_max_steps "$SMOKE_HPO_STEPS" \
    --hpo_rerank_max_steps "$SMOKE_HPO_STEPS" \
    --final_seeds "$SMOKE_FINAL_SEEDS" \
    --set "hpo.bandit.refine_seeds=$SMOKE_HPO_SEEDS" \
    --set "train.single.steps_mode=max_steps" \
    "${extra[@]}"
}

run_multi_source_smoke() {
  local scenario="$1"
  local runs_group="$2"
  local model="$3"
  local datasets_json="$4"
  shift 4 || true
  local extra=( "$@" )

  "$PYTHON_BIN" "$PIPELINE" \
    --runs_group "$runs_group" \
    --scenario "$scenario" \
    --multi_enabled true \
    --multi_datasets "$datasets_json" \
    --model "$model" \
    --trials "$SMOKE_TRIALS" \
    --multi_steps_mode max_steps \
    --max_steps "$SMOKE_FINAL_STEPS" \
    --epochs 1 \
    --final_epochs 1 \
    --hpo_baseline_epochs 1 \
    --hpo_grid_epochs 1 \
    --rerank_epochs 1 \
    --hpo_baseline_max_steps "$SMOKE_HPO_STEPS" \
    --hpo_grid_max_steps "$SMOKE_HPO_STEPS" \
    --hpo_rerank_max_steps "$SMOKE_HPO_STEPS" \
    --final_seeds "$SMOKE_FINAL_SEEDS" \
    --set "hpo.bandit.refine_seeds=$SMOKE_HPO_SEEDS" \
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

run_multi_task_ablation_smoke_resume() {
  local runs_group="$1"
  local model="$2"
  local datasets_json="$3"
  local resume_debug="${4:-}"
  local resume_final="${5:-}"
  local runs_parent
  runs_parent="$(resolve_runs_group_dir "$runs_group" "$datasets_json")"

  if [[ -z "$resume_debug" ]]; then
    resume_debug="$(latest_matching_dir "$runs_parent" "hpo__" || true)"
  fi
  if [[ -z "$resume_final" ]]; then
    resume_final="$(latest_matching_dir "$runs_parent" "final__" || true)"
  fi

  if [[ -z "$resume_debug" || ! -d "$resume_debug" ]]; then
    echo "[ERR] Missing smoke HPO dir for $runs_group" >&2
    return 1
  fi
  if [[ -z "$resume_final" || ! -d "$resume_final" ]]; then
    echo "[ERR] Missing smoke final dir for $runs_group" >&2
    return 1
  fi

  run_multi_source_smoke \
    "multi_task" \
    "$runs_group" \
    "$model" \
    "$datasets_json" \
    --resume_debug "$resume_debug" \
    --resume_final "$resume_final" \
    --with_ablations
}
