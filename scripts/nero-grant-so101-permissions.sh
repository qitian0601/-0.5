#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DEVICE="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122981-if00"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [DEVICE ...]

Grant the current login user read/write access to SO101 serial device nodes.

Default device:
  ${DEFAULT_DEVICE}

Examples:
  $(basename "$0")
  $(basename "$0") /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C122981-if00
  $(basename "$0") /dev/ttyACM0 /dev/ttyACM1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET_USER="${TARGET_USER:-${SUDO_USER:-$(logname 2>/dev/null || id -un)}}"
export TARGET_USER

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo --preserve-env=TARGET_USER "$0" "$@"
fi

if ! id "${TARGET_USER}" >/dev/null 2>&1; then
  echo "User '${TARGET_USER}' does not exist." >&2
  exit 1
fi

devices=("$@")
if [[ "${#devices[@]}" -eq 0 ]]; then
  devices=("${DEFAULT_DEVICE}")
fi

for device in "${devices[@]}"; do
  if [[ ! -e "${device}" ]]; then
    echo "Device not found: ${device}" >&2
    exit 1
  fi

  real_device="$(readlink -f "${device}")"
  if [[ ! -e "${real_device}" ]]; then
    echo "Resolved device not found: ${device} -> ${real_device}" >&2
    exit 1
  fi

  setfacl -m "u:${TARGET_USER}:rw" "${real_device}"
  echo "Granted ${TARGET_USER} read/write access: ${device} -> ${real_device}"
  getfacl -p "${real_device}" | sed -n '1,8p'
done

echo
echo "For a permanent fix, add the user to dialout and log in again:"
echo "  sudo usermod -aG dialout ${TARGET_USER}"
