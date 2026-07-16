#!/usr/bin/env bash
set -euo pipefail

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the RoboTwin checkout}"

PYTHON_BIN="${ROBOTWIN_PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROBOTWIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROBOTWIN_ROOT}"

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/replay_multiview_dataset.py" "$@"
