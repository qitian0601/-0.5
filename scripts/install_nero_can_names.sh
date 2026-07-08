#!/usr/bin/env bash
set -euo pipefail

LEFT_SERIAL="003100414148570C20343133"
RIGHT_SERIAL="003400464148570A20343133"
LEFT_NAME="nero_left"
RIGHT_NAME="nero_right"
BITRATE="${NERO_CAN_BITRATE:-1000000}"
APPLY_NOW=false

if [[ "${1:-}" == "--apply-now" ]]; then
  APPLY_NOW=true
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0 [--apply-now]" >&2
  exit 1
fi

install -d -m 0755 /etc/systemd/network

cat >/etc/systemd/network/10-nero-left-can.link <<EOF
[Match]
Driver=gs_usb
Property=ID_SERIAL_SHORT=${LEFT_SERIAL}

[Link]
NamePolicy=
Name=${LEFT_NAME}
EOF

cat >/etc/systemd/network/10-nero-right-can.link <<EOF
[Match]
Driver=gs_usb
Property=ID_SERIAL_SHORT=${RIGHT_SERIAL}

[Link]
NamePolicy=
Name=${RIGHT_NAME}
EOF

udevadm control --reload

find_interface_by_serial() {
  local serial="$1"
  local iface
  for path in /sys/class/net/*; do
    iface="$(basename "${path}")"
    [[ -e "${path}/device" ]] || continue
    if udevadm info -q property -p "${path}" | grep -qx "ID_SERIAL_SHORT=${serial}"; then
      printf '%s\n' "${iface}"
      return 0
    fi
  done
  return 1
}

rename_if_needed() {
  local serial="$1"
  local target="$2"
  local iface
  iface="$(find_interface_by_serial "${serial}" || true)"
  if [[ -z "${iface}" ]]; then
    echo "Warning: serial ${serial} is not currently connected." >&2
    return 0
  fi
  if [[ "${iface}" == "${target}" ]]; then
    ip link set "${target}" down || true
    ip link set "${target}" type can bitrate "${BITRATE}"
    ip link set "${target}" up
    return 0
  fi
  ip link set "${iface}" down || true
  ip link set "${iface}" name "${target}"
  ip link set "${target}" type can bitrate "${BITRATE}"
  ip link set "${target}" up
}

if [[ "${APPLY_NOW}" == true ]]; then
  rename_if_needed "${LEFT_SERIAL}" "${LEFT_NAME}"
  rename_if_needed "${RIGHT_SERIAL}" "${RIGHT_NAME}"
fi

echo "Installed Nero CAN interface names:"
echo "  ${LEFT_NAME}:  serial ${LEFT_SERIAL}"
echo "  ${RIGHT_NAME}: serial ${RIGHT_SERIAL}"
echo
if [[ "${APPLY_NOW}" != true ]]; then
  echo "Reconnect the USB-CAN adapters or reboot, then bring them up at ${BITRATE} bitrate."
fi
