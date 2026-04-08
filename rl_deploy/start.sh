#!/bin/bash
# ============================================================
#  CRATER MISSION CONTROL — One-Command Launcher
#  Team Crater | Tulane University | CMPS 4020 | Spring 2026
# ============================================================
#
#  Usage:  ssh jetson@100.68.244.40
#          cd ~/rl_deploy && ./start.sh
#
#  Or one-liner from your laptop:
#    ssh jetson@100.68.244.40 'cd ~/rl_deploy && ./start.sh'
#
# ============================================================

set -e
cd "$(dirname "$0")"

PORT=5050
TAILSCALE_IP="100.68.244.40"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        🚀 CRATER MISSION CONTROL — LAUNCHER            ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Cleaning up old processes...                           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# --- Kill any existing instances ---
pkill -9 -f "python3 mission_control.py" 2>/dev/null || true
pkill -9 -f "python3 rl_deploy.py" 2>/dev/null || true
# Also kill any python3 holding the camera
fuser -k /dev/video0 2>/dev/null || true
fuser -k /dev/video1 2>/dev/null || true
echo "  Waiting for camera/serial to release..."
sleep 5

# --- Clean stale bytecode (prevents garbled camera from old code) ---
find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# --- Ensure frame directory exists ---
mkdir -p /tmp/rover_frames

# --- Launch ---
echo "Starting Mission Control on port ${PORT}..."
echo ""
python3 mission_control.py --port ${PORT}
