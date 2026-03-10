#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/lyclyq/Optimization/grad-shake-align"
SYSTEMD_DIR="${HOME}/.config/systemd/user"
SERVICE_NAME="grad-shake-align-auto-push.service"
TIMER_NAME="grad-shake-align-auto-push.timer"
SERVICE_PATH="${SYSTEMD_DIR}/${SERVICE_NAME}"
TIMER_PATH="${SYSTEMD_DIR}/${TIMER_NAME}"

mkdir -p "$SYSTEMD_DIR"

cp "${REPO_DIR}/systemd/${SERVICE_NAME}" "$SERVICE_PATH"
cp "${REPO_DIR}/systemd/${TIMER_NAME}" "$TIMER_PATH"

systemctl --user daemon-reload
systemctl --user enable --now "$TIMER_NAME"

printf '%s\n' "Installed ${TIMER_NAME}"
systemctl --user list-timers "$TIMER_NAME" --all --no-pager
