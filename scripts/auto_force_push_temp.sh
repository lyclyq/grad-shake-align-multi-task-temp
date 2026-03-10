#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/lyclyq/Optimization/grad-shake-align"
REMOTE_NAME="${AUTO_PUSH_REMOTE_NAME:-temp_backup}"
REMOTE_URL="${AUTO_PUSH_REMOTE_URL:-https://github.com/lyclyq/grad-shake-align-multi-task-temp.git}"
BRANCH="${AUTO_PUSH_BRANCH:-main}"

cd "$REPO_DIR"

if ! git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
else
  current_url="$(git remote get-url "$REMOTE_NAME")"
  if [[ "$current_url" != "$REMOTE_URL" ]]; then
    git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
  fi
fi

git add -A

if ! git diff --cached --quiet || ! git diff --quiet; then
  git commit -m "auto sync $(date +%F)"
fi

git push "$REMOTE_NAME" "HEAD:${BRANCH}" --force
