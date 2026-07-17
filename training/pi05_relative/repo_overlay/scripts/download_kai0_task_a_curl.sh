#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-https://hf-mirror.com/datasets/OpenDriveLab-org/Kai0/resolve/main}"
ROOT="${ROOT:-/public/home/chenglongyan/workspace/lerobot/hf_downloads/datasets/OpenDriveLab-org/Kai0}"
LIST="${LIST:-/tmp/kai0_task_a_files_${USER}.txt}"
JOBS="${JOBS:-16}"
LOG_DIR="${ROOT}"
DONE_LOG="${LOG_DIR}/download_task_a_done.log"
ERR_LOG="${LOG_DIR}/download_task_a_errors.log"

mkdir -p "$ROOT"
: > "$DONE_LOG"
: > "$ERR_LOG"

download_one() {
  local rel="$1"
  local out="${ROOT}/${rel}"
  local tmp="${out}.part"
  local url="${BASE_URL}/${rel}"

  mkdir -p "$(dirname "$out")"

  if [[ -s "$out" ]]; then
    printf 'SKIP %s\n' "$rel" >> "$DONE_LOG"
    return 0
  fi

  if curl -fsSL \
      --retry 5 \
      --retry-delay 2 \
      --retry-all-errors \
      --connect-timeout 20 \
      --speed-time 90 \
      --speed-limit 1024 \
      -C - \
      -o "$tmp" \
      "$url"; then
    mv "$tmp" "$out"
    printf 'OK %s\n' "$rel" >> "$DONE_LOG"
  else
    printf '%s\n' "$rel" >> "$ERR_LOG"
    return 1
  fi
}

active=0
status=0

while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue

  download_one "$rel" &
  active=$((active + 1))

  if (( active >= JOBS )); then
    if ! wait -n; then
      status=1
    fi
    active=$((active - 1))
  fi
done < "$LIST"

while (( active > 0 )); do
  if ! wait -n; then
    status=1
  fi
  active=$((active - 1))
done

exit "$status"
