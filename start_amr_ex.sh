#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start_amr.sh
#
# Launches the robot Docker container with:
#  • X11 display forwarding
#  • ALSA → PulseAudio routing (so PyAudio sees the Bluetooth mic)
#  • PulseAudio socket share  ← required for Bluetooth microphone
#  • Serial port access
#
# Workflow:
#   1.  ./src/robot_controller/scripts/use_bluetooth_mic.sh
#   2.  ./start_amr.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── PulseAudio socket (host → container) ─────────────────────────────────────
# Bluetooth audio is a PulseAudio virtual source — not a raw ALSA device.
# The container connects to the host's PulseAudio daemon via its Unix socket.
PULSE_SOCKET="/run/user/$(id -u)/pulse/native"
PULSE_COOKIE="$HOME/.config/pulse/cookie"

if [[ ! -S "$PULSE_SOCKET" ]]; then
    echo "[WARN]  PulseAudio socket not found at $PULSE_SOCKET"
    echo "[WARN]  Audio (including Bluetooth mic) will NOT work inside the container."
    echo "[WARN]  Make sure PulseAudio is running: pulseaudio --start"
else
    echo "[OK]    PulseAudio socket: $PULSE_SOCKET"
fi

# ── Bluetooth mic prerequisite check ─────────────────────────────────────────
BT_SOURCE_NAME=$(pactl info 2>/dev/null | grep "Default Source:" | awk '{print $NF}' || true)
if echo "$BT_SOURCE_NAME" | grep -qi "bluez"; then
    echo "[OK]    BT mic is default source: $BT_SOURCE_NAME"
else
    echo "[WARN]  Default audio source is NOT Bluetooth: ${BT_SOURCE_NAME:-none}"
    echo "[WARN]  Run:  ${SCRIPT_DIR}/use_bluetooth_mic.sh   before starting."
    echo "[WARN]  Continuing anyway …"
fi

# ── ALSA → PulseAudio config file ────────────────────────────────────────────
# PyAudio uses ALSA directly; without this config it cannot see PulseAudio
# virtual devices (like the BT headset). docker_asoundrc routes ALSA "default"
# through the PulseAudio plugin (requires libasound2-plugins in the container).
ASOUNDRC="${SCRIPT_DIR}/docker_asoundrc"
if [[ ! -f "$ASOUNDRC" ]]; then
    echo "[WARN]  docker_asoundrc not found at $ASOUNDRC — BT mic may not appear in PyAudio"
fi

# ── X11 display access ────────────────────────────────────────────────────────
echo "[INFO]  Granting X11 display access for Docker …"
xhost +local:root

# ── Build optional mount flags ────────────────────────────────────────────────
COOKIE_MOUNT=""
[[ -f "$PULSE_COOKIE" ]] && COOKIE_MOUNT="-v ${PULSE_COOKIE}:/root/.config/pulse/cookie:ro"

ASOUNDRC_MOUNT=""
[[ -f "$ASOUNDRC" ]] && ASOUNDRC_MOUNT="-v ${ASOUNDRC}:/root/.asoundrc:ro"

# ── Launch container ──────────────────────────────────────────────────────────
echo "[INFO]  Starting AMR container …"

docker run -it --rm \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/video1:/dev/video1 \
  --device=/dev/snd:/dev/snd \
  -v "${PULSE_SOCKET}:/run/user/1000/pulse/native" \
  -e PULSE_SERVER="unix:/run/user/1000/pulse/native" \
  -e PULSE_RUNTIME_PATH="/run/user/1000/pulse" \
  ${COOKIE_MOUNT} \
  ${ASOUNDRC_MOUNT} \
  --device=/dev/ttyACM0:/dev/ttyACM0 \
  --device=/dev/ttyACM1:/dev/ttyACM1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY="$DISPLAY" \
  -v /home/manh/ros2_workspace:/ros2_ws \
  -w /ros2_ws \
  my_robot_env \
  bash -c '
    # Ensure ALSA PulseAudio plugin is present (needed for PyAudio to see BT mic)
    if ! dpkg -l libasound2-plugins 2>/dev/null | grep -q "^ii"; then
        echo "[SETUP] Installing libasound2-plugins for ALSA→PulseAudio routing …"
        apt-get install -qq -y libasound2-plugins 2>&1 | tail -3
    fi
    source install/setup.bash
    ros2 launch robot_controller bringup.launch.py \
  '

