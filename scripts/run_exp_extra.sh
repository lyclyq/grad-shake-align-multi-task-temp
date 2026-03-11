#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/exp_protocol_common.sh"

DATASETS='["glue/sst2","yelp_polarity","amazon_polarity"]'

run_multi_source_protocol \
  "multi_dataset" \
  "extra_multidataset_deberta_sentiment" \
  "microsoft/deberta-v3-base" \
  "$DATASETS"
