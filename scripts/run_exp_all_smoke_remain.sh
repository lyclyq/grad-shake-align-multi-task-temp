#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_smoke_common.sh"

export SMOKE_TRIALS="${SMOKE_TRIALS:-1}"
export SMOKE_HPO_STEPS="${SMOKE_HPO_STEPS:-1}"
export SMOKE_FINAL_STEPS="${SMOKE_FINAL_STEPS:-1}"
export SMOKE_HPO_SEEDS="${SMOKE_HPO_SEEDS:-[1]}"
export SMOKE_FINAL_SEEDS="${SMOKE_FINAL_SEEDS:-[1]}"
export SMOKE_COORD_TOP_K="${SMOKE_COORD_TOP_K:-0}"
export SMOKE_REFINE_RADIX="${SMOKE_REFINE_RADIX:-1}"
export SMOKE_EVAL_MAX_BATCHES="${SMOKE_EVAL_MAX_BATCHES:-1}"
export SMOKE_COMPUTE_TRAIN_ACC="${SMOKE_COMPUTE_TRAIN_ACC:-false}"
export SMOKE_COMPILE="${SMOKE_COMPILE:-false}"
export SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"

MIX3='["glue/rte","glue/mrpc","glue/cola"]'
DATASETS='["glue/sst2","yelp_polarity","amazon_polarity"]'

has_matching_final_provenance() {
  local runs_parent="$1"
  local with_ablations="$2"
  if [[ ! -d "$runs_parent" ]]; then
    return 1
  fi

  "$PYTHON_BIN" - "$runs_parent" "$with_ablations" <<'PY'
import json
import pathlib
import sys

runs_parent = pathlib.Path(sys.argv[1])
want_abl = sys.argv[2].strip().lower() == "true"
for d in sorted(runs_parent.glob("final__*"), reverse=True):
    prov = d / "final_provenance_from_pipeline.json"
    if not prov.exists():
        continue
    try:
        obj = json.loads(prov.read_text(encoding="utf-8"))
    except Exception:
        continue
    got_abl = bool(((obj.get("spec") or {}).get("with_ablations", False)))
    if got_abl == want_abl:
        print(str(d))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

latest_resume_final() {
  local runs_parent="$1"
  if [[ ! -d "$runs_parent" ]]; then
    return 1
  fi
  find "$runs_parent" -maxdepth 1 -mindepth 1 -type d -name 'final__*' | sort | tail -n 1
}

run_single_task_smoke_resumeable() {
  local runs_group="$1"
  local dataset="$2"
  local model="$3"
  shift 3 || true
  local extra=( "$@" )
  local runs_parent="$ROOT/runs/$runs_group"

  if has_matching_final_provenance "$runs_parent" false >/dev/null; then
    echo "[SKIP] completed single-task smoke: $runs_group"
    return 0
  fi

  local resume_debug=""
  local resume_final=""
  local resume_args=()
  resume_debug="$(latest_matching_dir "$runs_parent" "hpo__" || true)"
  resume_final="$(latest_resume_final "$runs_parent" || true)"

  echo "[RUN] single-task smoke: $runs_group"
  if [[ -n "$resume_debug" || -n "$resume_final" ]]; then
    echo "[INFO] resuming with existing dirs"
  fi
  if [[ -n "$resume_debug" ]]; then
    resume_args+=(--resume_debug "$resume_debug")
  fi
  if [[ -n "$resume_final" ]]; then
    resume_args+=(--resume_final "$resume_final")
  fi

  run_single_task_smoke \
    "$runs_group" \
    "$dataset" \
    "$model" \
    "${extra[@]}" \
    "${resume_args[@]}"
}

run_multi_source_smoke_resumeable() {
  local scenario="$1"
  local runs_group="$2"
  local model="$3"
  local datasets_json="$4"
  shift 4 || true
  local extra=( "$@" )
  local runs_parent
  runs_parent="$(resolve_runs_group_dir "$runs_group" "$datasets_json")"

  if has_matching_final_provenance "$runs_parent" false >/dev/null; then
    echo "[SKIP] completed multi-source smoke: $runs_group"
    return 0
  fi

  local resume_debug=""
  local resume_final=""
  local resume_args=()
  resume_debug="$(latest_matching_dir "$runs_parent" "hpo__" || true)"
  resume_final="$(latest_resume_final "$runs_parent" || true)"

  echo "[RUN] multi-source smoke: $runs_group"
  if [[ -n "$resume_debug" || -n "$resume_final" ]]; then
    echo "[INFO] resuming with existing dirs"
  fi
  if [[ -n "$resume_debug" ]]; then
    resume_args+=(--resume_debug "$resume_debug")
  fi
  if [[ -n "$resume_final" ]]; then
    resume_args+=(--resume_final "$resume_final")
  fi

  run_multi_source_smoke \
    "$scenario" \
    "$runs_group" \
    "$model" \
    "$datasets_json" \
    "${extra[@]}" \
    "${resume_args[@]}"
}

run_multi_task_ablation_smoke_resumeable() {
  local runs_group="$1"
  local model="$2"
  local datasets_json="$3"
  local runs_parent
  runs_parent="$(resolve_runs_group_dir "$runs_group" "$datasets_json")"

  if has_matching_final_provenance "$runs_parent" true >/dev/null; then
    echo "[SKIP] completed ablation smoke: $runs_group"
    return 0
  fi

  echo "[RUN] ablation smoke: $runs_group"
  run_multi_task_ablation_smoke_resume "$runs_group" "$model" "$datasets_json"
}

run_single_task_smoke_resumeable \
  "smoke_suite_main_single_roberta_rte" \
  "glue/rte" \
  "roberta-base"

run_multi_source_smoke_resumeable \
  "multi_task" \
  "smoke_suite_main_multi_roberta_mix3" \
  "roberta-base" \
  "$MIX3"

run_single_task_smoke_resumeable \
  "smoke_suite_appendix_rank_single_deberta_mrpc_r16_R64" \
  "glue/mrpc" \
  "microsoft/deberta-v3-base" \
  --ours_r 16 \
  --ours_R 64

run_multi_source_smoke_resumeable \
  "multi_task" \
  "smoke_suite_appendix_rank_multi_deberta_mix3_r16_R64" \
  "microsoft/deberta-v3-base" \
  "$MIX3" \
  --ours_r 16 \
  --ours_R 64

run_multi_source_smoke_resumeable \
  "multi_dataset" \
  "smoke_suite_extra_multidataset_deberta_sentiment" \
  "microsoft/deberta-v3-base" \
  "$DATASETS"

run_multi_task_ablation_smoke_resumeable \
  "smoke_suite_main_multi_roberta_mix3" \
  "roberta-base" \
  "$MIX3"

echo "[SMOKE] remaining smoke experiments finished"
