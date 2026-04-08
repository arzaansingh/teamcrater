#!/usr/bin/env python3
"""
Mission Control — Team Crater
================================
Space-mission-themed web control panel for the lunar rover.
Tulane University | CMPS 4020 Capstone | Spring 2026

Modules:
  mission_control.py  — This file: Web UI + SocketIO server (THE PANEL)
  rover_control.py    — Hardware abstraction (THE CONTROLS)
  rl_deploy.py        — RL navigation autonomy (THE MODEL, runs as subprocess)

Usage:
    python3 mission_control.py                    # Full mode with hardware
    python3 mission_control.py --no-hardware      # UI only, no rover connection
    python3 mission_control.py --port 5050        # Custom port
    python3 mission_control.py --segmentation     # Enable terrain segmentation view

Access: http://100.68.244.40:5050  (via Tailscale)
"""

import sys
import os
import time
import signal
import argparse
import subprocess
import threading
import json as _json

from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO, emit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rover_control import RoverHardware, DummyRoverHardware, CameraManager

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crater-mission-control'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# Global state — set in main()
rover = None           # RoverHardware or DummyRoverHardware
camera_mgr = None      # CameraManager
proc_mgr = None        # SubprocessManager
_app_config = {}       # CLI config passed to template


# ---------------------------------------------------------------------------
# SubprocessManager — launches rl_deploy.py and streams its logs
# ---------------------------------------------------------------------------

class SubprocessManager:
    """Manages rl_deploy.py as a child process with log streaming."""

    def __init__(self, sio):
        self.sio = sio
        self.proc = None
        self._log_thread = None

    def start(self, config):
        """Build CLI from config dict, launch rl_deploy.py, start log reader."""
        if self.is_running:
            return False

        cmd = self._build_command(config)
        print(f"[PROC] Launching: {' '.join(cmd)}")

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                preexec_fn=os.setsid,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            )
            self._log_thread = threading.Thread(target=self._read_logs, daemon=True)
            self._log_thread.start()
            return True
        except Exception as e:
            print(f"[PROC] Failed to launch: {e}")
            self.sio.emit('log_line', {'text': f'[ERROR] Launch failed: {e}', 'ts': time.time()})
            return False

    def stop(self):
        """Graceful stop: SIGINT then SIGKILL after 3s."""
        if not self.is_running:
            return
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        except Exception:
            pass
        self.proc = None
        print("[PROC] Stopped")

    def e_stop(self):
        """Immediate kill — no grace period."""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        self.proc = None
        print("[PROC] E-STOP killed")

    @property
    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def _build_command(self, cfg):
        cmd = [sys.executable, 'rl_deploy.py']
        if cfg.get('model'):
            # Expand ~ and resolve path so subprocess finds the model
            model_path = os.path.expanduser(str(cfg['model']))
            cmd += ['--model', model_path]
        if cfg.get('demo'):
            cmd.append('--demo')
        if cfg.get('dry_run'):
            cmd.append('--dry-run')
        if cfg.get('max_speed') is not None:
            cmd += ['--max-speed', str(cfg['max_speed'])]
        if cfg.get('duration'):
            cmd += ['--duration', str(cfg['duration'])]
        if cfg.get('distance'):
            cmd += ['--distance', str(cfg['distance'])]
        if cfg.get('port'):
            cmd += ['--port', str(cfg['port'])]
        if not cfg.get('enable_detect', True):
            cmd.append('--no-detect')
        if not cfg.get('enable_gimbal', True):
            cmd.append('--no-gimbal')
        if cfg.get('target_mode') in ('face', 'red'):
            cmd += ['--target', cfg['target_mode']]
        if cfg.get('face_distance'):
            cmd += ['--face-distance', str(cfg['face_distance'])]
        if cfg.get('segmentation'):
            cmd.append('--segmentation')
        if cfg.get('seg_model'):
            cmd += ['--seg-model', str(cfg['seg_model'])]
        if cfg.get('log_rewards'):
            cmd.append('--log-rewards')
        # Always stream frames for web UI during autonomous
        cmd += ['--frame-dir', '/tmp/rover_frames']
        return cmd

    def _read_logs(self):
        """Stream subprocess stdout to web UI."""
        try:
            for line in self.proc.stdout:
                text = line.rstrip()
                if text:
                    self.sio.emit('log_line', {'text': text, 'ts': time.time()})
                    self._parse_telemetry(text)
        except Exception:
            pass
        # Process exited
        code = self.proc.returncode if self.proc else -1
        self.sio.emit('mission_ended', {'code': code})
        self.sio.emit('log_line', {
            'text': f'[SYSTEM] Process exited with code {code}',
            'ts': time.time()
        })

    def _parse_telemetry(self, line):
        """Extract telemetry values from structured rl_deploy.py log lines."""
        # Example line: [42s] raw=[+0.15,-0.30] -> lin=+0.150 ang=-0.450 depth=0.12m ...
        try:
            telem = {}
            if 'depth=' in line:
                import re
                m = re.search(r'depth=([\d.]+)m', line)
                if m:
                    telem['min_depth'] = float(m.group(1))
                m = re.search(r'L=([\d.]+)', line)
                if m:
                    telem['left_depth'] = float(m.group(1))
                m = re.search(r'R=([\d.]+)', line)
                if m:
                    telem['right_depth'] = float(m.group(1))
                m = re.search(r'lin=([+-]?[\d.]+)', line)
                if m:
                    telem['speed'] = abs(float(m.group(1)))
                m = re.search(r'hdg=([+-]?\d+)', line)
                if m:
                    telem['heading'] = float(m.group(1))
                m = re.search(r'dist=([\d.]+)m', line)
                if m:
                    telem['distance'] = float(m.group(1))
            if 'SEARCH' in line or 'NAVIGATE' in line or 'LOST' in line or 'ARRIVED' in line:
                for state in ['SEARCH_SPIN', 'SEARCH_DRIVE', 'NAVIGATE', 'LOST_HOLD', 'LOST_SPIN', 'ARRIVED']:
                    if state in line:
                        telem['face_state'] = state
                        break
            for mode in ['EMERGENCY', 'BACKUP', 'AVOID', 'SLOW', 'CLEAR']:
                if f'[{mode}]' in line:
                    telem['mode'] = mode
                    break
            if telem:
                self.sio.emit('telemetry', telem)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mode transition helpers
# ---------------------------------------------------------------------------

def _transition_to_autonomous(config):
    """Release camera + serial, launch subprocess."""
    global camera_mgr, rover, proc_mgr
    camera_mgr.stop()
    rover.release_serial()
    time.sleep(1.0)
    ok = proc_mgr.start(config)
    if not ok:
        # Failed — restore monitoring mode
        time.sleep(0.5)
        rover.reacquire_serial()
        camera_mgr.start()
    return ok


def _transition_to_monitoring():
    """Kill subprocess, restore camera + serial."""
    global camera_mgr, rover, proc_mgr
    time.sleep(1.0)
    rover.reacquire_serial()
    time.sleep(0.5)
    camera_mgr.start()


# ---------------------------------------------------------------------------
# MJPEG stream generators
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, config=_app_config)


def _mjpeg_generator(feed_name, target_fps=10):
    """Yields MJPEG frames from camera or fallback files. Streams until client disconnects."""
    interval = 1.0 / target_fps
    while True:
        data = None
        if feed_name == 'rgb':
            data = camera_mgr.get_rgb_jpeg()
        elif feed_name == 'depth':
            data = camera_mgr.get_depth_jpeg()
        elif feed_name == 'seg':
            data = camera_mgr.get_seg_jpeg()

        # Fallback to files written by rl_deploy.py subprocess
        if not data:
            try:
                with open(f'/tmp/rover_frames/{feed_name}.jpg', 'rb') as f:
                    data = f.read()
            except Exception:
                pass

        if data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n'
                   + data + b'\r\n')
        time.sleep(interval)


@app.route('/stream/rgb')
def stream_rgb():
    """MJPEG stream — persistent connection, low latency, ~10fps."""
    return Response(_mjpeg_generator('rgb', target_fps=10),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-store'})


@app.route('/stream/depth')
def stream_depth():
    return Response(_mjpeg_generator('depth', target_fps=8),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-store'})


@app.route('/stream/seg')
def stream_seg():
    return Response(_mjpeg_generator('seg', target_fps=5),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-store'})


# Snapshot endpoints kept as fallback for browsers that don't support MJPEG
@app.route('/snapshot/rgb')
def snap_rgb():
    data = camera_mgr.get_rgb_jpeg()
    if not data:
        try:
            with open('/tmp/rover_frames/rgb.jpg', 'rb') as f:
                data = f.read()
        except Exception:
            return '', 204
    return Response(data, mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


@app.route('/snapshot/depth')
def snap_depth():
    data = camera_mgr.get_depth_jpeg()
    if not data:
        try:
            with open('/tmp/rover_frames/depth.jpg', 'rb') as f:
                data = f.read()
        except Exception:
            return '', 204
    return Response(data, mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


@app.route('/snapshot/seg')
def snap_seg():
    data = camera_mgr.get_seg_jpeg()
    if not data:
        try:
            with open('/tmp/rover_frames/seg.jpg', 'rb') as f:
                data = f.read()
        except Exception:
            return '', 204
    return Response(data, mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


@app.route('/api/status')
def api_status():
    telem = camera_mgr.get_telemetry()
    return _json.dumps({
        'camera_active': camera_mgr.is_active,
        'autonomous': proc_mgr.is_running if proc_mgr else False,
        'hardware': rover.is_connected if hasattr(rover, 'is_connected') else False,
        **telem,
    }), 200, {'Content-Type': 'application/json'}


# ---------------------------------------------------------------------------
# SocketIO event handlers
# ---------------------------------------------------------------------------

@socketio.on('connect')
def on_connect():
    emit('mission_state', {
        'running': proc_mgr.is_running if proc_mgr else False,
        'hardware': rover.is_connected if hasattr(rover, 'is_connected') else False,
    })


@socketio.on('start_mission')
def on_start_mission(config):
    # Force-clean any stale subprocess (race condition after E-STOP)
    if proc_mgr.is_running:
        emit('log_line', {'text': '[SYSTEM] Cleaning up previous mission...', 'ts': time.time()})
        proc_mgr.e_stop()
        time.sleep(1.0)
    emit('log_line', {'text': '[SYSTEM] Starting autonomous mission...', 'ts': time.time()})
    ok = _transition_to_autonomous(config)
    emit('mission_state', {'running': ok})
    if not ok:
        emit('log_line', {'text': '[ERROR] Failed to start mission', 'ts': time.time()})


@socketio.on('stop_mission')
def on_stop_mission():
    if not proc_mgr.is_running:
        return
    emit('log_line', {'text': '[SYSTEM] Stopping mission (graceful)...', 'ts': time.time()})
    proc_mgr.stop()
    _transition_to_monitoring()
    emit('mission_state', {'running': False})
    emit('log_line', {'text': '[SYSTEM] Mission stopped. Camera restored.', 'ts': time.time()})


@socketio.on('e_stop')
def on_e_stop():
    emit('log_line', {'text': '[EMERGENCY] E-STOP ACTIVATED', 'ts': time.time()})
    proc_mgr.e_stop()
    time.sleep(0.5)
    rover.emergency_stop()
    _transition_to_monitoring()
    emit('mission_state', {'running': False})
    emit('log_line', {'text': '[SYSTEM] E-STOP complete. All systems halted.', 'ts': time.time()})


@socketio.on('manual_drive')
def on_manual_drive(data):
    if proc_mgr and proc_mgr.is_running:
        return  # blocked during autonomous
    rover.drive(data.get('linear', 0), data.get('angular', 0))


@socketio.on('manual_stop')
def on_manual_stop():
    if proc_mgr and proc_mgr.is_running:
        return
    rover.stop()


@socketio.on('gimbal_move')
def on_gimbal_move(data):
    if proc_mgr and proc_mgr.is_running:
        return
    rover.gimbal_set(data.get('pan', 0), data.get('tilt', 0))


@socketio.on('lights_set')
def on_lights_set(data):
    rover.lights_set(data.get('a', 0), data.get('b', 0))


@socketio.on('set_detection')
def on_set_detection(data):
    camera_mgr.set_detection(data.get('enabled', False))


@socketio.on('reset_all')
def on_reset_all():
    """Kill everything, restart camera + serial from scratch."""
    global camera_mgr, rover, proc_mgr
    emit('log_line', {'text': '[SYSTEM] FULL RESET — killing processes, restarting hardware...', 'ts': time.time()})
    # Kill any running subprocess
    proc_mgr.e_stop()
    time.sleep(0.5)
    # Stop camera
    camera_mgr.stop()
    time.sleep(0.5)
    # Emergency stop rover
    rover.emergency_stop()
    time.sleep(0.5)
    # Reacquire serial
    rover.reacquire_serial()
    time.sleep(0.5)
    # Restart camera
    camera_mgr.start()
    emit('mission_state', {'running': False})
    emit('log_line', {'text': '[SYSTEM] RESET COMPLETE — all systems restarted.', 'ts': time.time()})


# ---------------------------------------------------------------------------
# HTML Template — Space Mission Control Panel
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CRATER MISSION CONTROL</title>
<script src="https://cdn.socket.io/4.7.4/socket.io.min.js"></script>
<style>
/* ===== RESET & BASE ===== */
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --bg: #1B3A2D; --panel: #0D2818; --border: #006747;
    --accent: #87CEEB; --green: #006747; --green-light: #00A676;
    --text: #e0e0e0; --text-dim: #9BC4A8; --text-mono: #00ff88;
    --red: #FF0000; --red-glow: #FF6B6B; --amber: #FFB800;
    --log-bg: #0a1a12;
}
body {
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    height: 100vh; overflow: hidden;
    display: grid;
    grid-template-rows: 36px 1fr 140px;
    grid-template-columns: 190px 1fr 230px;
    grid-template-areas:
        "header  header  header"
        "left    center  right"
        "log     log     log";
}
@media (max-width: 1100px) {
    body {
        grid-template-columns: 1fr;
        grid-template-rows: auto auto auto auto auto;
        grid-template-areas: "header" "center" "left" "right" "log";
    }
}
@media (max-width: 700px) {
    .feeds { grid-template-columns: 1fr !important; }
    .feeds.presenter { grid-template-columns: 1fr !important; }
    .feed-box.primary { grid-row: auto; grid-column: auto; }
    .telemetry { grid-template-columns: repeat(3, 1fr) !important; }
}

/* ===== HEADER ===== */
.header {
    grid-area: header;
    background: linear-gradient(135deg, var(--green) 0%, var(--bg) 100%);
    border-bottom: 2px solid var(--accent);
    padding: 4px 16px;
    display: flex; align-items: center; justify-content: space-between;
}
.header-title { display: flex; align-items: center; gap: 10px; }
.header-title h1 { color: var(--accent); font-size: 1em; letter-spacing: 2px; font-weight: 700; }
.header-title .subtitle { display: none; }
.header-status { display: flex; align-items: center; gap: 10px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.status-dot.on { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
.status-dot.off { background: #ff4444; }
.mode-badge {
    background: var(--panel); border: 1px solid var(--accent);
    color: var(--accent); padding: 2px 8px; border-radius: 4px;
    font-size: 0.65em; font-weight: 700; letter-spacing: 1px;
}
.mode-badge.autonomous { border-color: var(--amber); color: var(--amber); }
.tulane-badge { color: var(--text-dim); font-size: 0.5em; letter-spacing: 0.5px; }

/* ===== PANELS ===== */
.panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; margin: 2px; overflow: hidden;
}
.panel-header {
    background: var(--green); color: var(--accent);
    padding: 3px 8px; font-size: 0.65em; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase;
}

/* ===== LEFT: MANUAL CONTROL ===== */
.left-panel {
    grid-area: left; display: flex; flex-direction: column; gap: 2px;
    padding: 2px; overflow-y: auto;
}
.left-panel.disabled { opacity: 0.4; pointer-events: none; }

/* WASD Keys */
.wasd { display: grid; grid-template-columns: repeat(3, 38px); gap: 2px; justify-content: center; padding: 4px; }
.wasd-key {
    width: 38px; height: 32px; background: #1a3a28; border: 1px solid var(--border);
    border-radius: 4px; display: flex; align-items: center; justify-content: center;
    color: var(--accent); font-weight: 700; font-size: 0.85em;
    cursor: pointer; user-select: none; transition: all 0.1s;
}
.wasd-key:hover { background: var(--green); }
.wasd-key.active { background: var(--accent); color: var(--bg); box-shadow: 0 0 10px var(--accent); }
.wasd-key.blank { visibility: hidden; }

/* Sliders */
.ctrl-group { padding: 2px 6px; }
.ctrl-label { color: var(--text-dim); font-size: 0.62em; margin-bottom: 1px; display: flex; justify-content: space-between; }
.ctrl-label span { color: var(--accent); font-weight: 700; }
input[type="range"] {
    width: 100%; -webkit-appearance: none; height: 6px;
    background: #1a3a28; border-radius: 3px; outline: none;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px;
    background: var(--accent); border-radius: 50%; cursor: pointer;
}

/* ===== CENTER ===== */
.center-panel { grid-area: center; display: flex; flex-direction: column; gap: 3px; padding: 3px; overflow-y: auto; }

/* Camera feeds — Presenter mode (click to zoom) */
.feeds {
    display: grid; gap: 6px;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: auto;
    transition: all 0.3s ease;
}
.feeds.presenter {
    grid-template-columns: 1fr 280px;
    grid-template-rows: auto auto;
}
.feed-box {
    background: #000; border: 2px solid var(--border); border-radius: 6px;
    overflow: hidden; cursor: pointer; transition: all 0.3s ease;
}
.feed-box:hover { border-color: var(--accent); }
.feed-box.primary {
    grid-row: 1 / 3; grid-column: 1;
    border-color: var(--accent); box-shadow: 0 0 12px rgba(135,206,235,0.3);
}
.feed-box .feed-label {
    background: var(--green); color: var(--accent);
    text-align: center; font-size: 0.55em; padding: 2px; font-weight: 700;
}
.feed-box img { width: 100%; height: auto; display: block; }
.feed-info {
    padding: 2px 5px; background: rgba(0,0,0,0.8);
    font-family: 'Courier New', monospace; font-size: 0.55em;
    color: var(--text-dim); line-height: 1.2;
}
.feed-info .info-val { color: var(--accent); font-weight: 700; }
.feed-hint { display: none; }

/* Autonomous overlay */
.auto-overlay {
    display: none; flex: 1; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.7); border: 2px solid var(--amber);
    border-radius: 8px; text-align: center; padding: 20px;
    min-height: 200px;
}
.auto-overlay.active { display: flex; }
.auto-overlay h2 { color: var(--amber); font-size: 1.2em; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

/* Telemetry gauges */
.telemetry { display: grid; grid-template-columns: repeat(6, 1fr); gap: 3px; padding: 3px; }
.gauge {
    background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
    text-align: center; padding: 3px 2px;
}
.gauge-label { color: var(--text-dim); font-size: 0.5em; text-transform: uppercase; letter-spacing: 0.5px; }
.gauge-value { color: var(--accent); font-size: 1em; font-weight: 700; font-family: 'Courier New', monospace; margin: 1px 0; }
.gauge-unit { color: var(--text-dim); font-size: 0.45em; }

/* ===== RIGHT: MISSION CONFIG ===== */
.right-panel { grid-area: right; display: flex; flex-direction: column; gap: 2px; padding: 2px; overflow-y: auto; }

.config-form { padding: 3px 6px; display: flex; flex-direction: column; gap: 3px; }
.config-form label { color: var(--text-dim); font-size: 0.62em; display: flex; flex-direction: column; gap: 1px; }
.config-form label.row { flex-direction: row; align-items: center; gap: 6px; }
.config-form input[type="text"], .config-form input[type="number"] {
    background: #1a3a28; border: 1px solid var(--border); color: var(--text);
    padding: 2px 5px; border-radius: 3px; font-size: 0.72em; width: 100%;
}
.config-form input[type="checkbox"] { accent-color: var(--accent); }
.config-form select {
    background: #1a3a28; border: 1px solid var(--border); color: var(--text);
    padding: 4px 8px; border-radius: 4px; font-size: 0.8em;
}
.face-opts { display: none; padding-left: 10px; border-left: 2px solid var(--border); }
.face-opts.show { display: block; }

/* Buttons */
.btn-group { display: flex; flex-direction: column; gap: 3px; padding: 3px 6px; }
.btn {
    padding: 6px; border: none; border-radius: 4px; font-weight: 700;
    font-size: 0.75em; cursor: pointer; letter-spacing: 1px; transition: all 0.2s;
}
.btn-start { background: var(--green); color: var(--accent); }
.btn-start:hover { background: var(--green-light); }
.btn-start:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-stop { background: var(--amber); color: #000; display: none; }
.btn-stop:hover { background: #ffcc33; }
.btn-estop {
    background: var(--red); color: #fff; font-size: 1em;
    box-shadow: 0 0 15px rgba(255,0,0,0.3);
    animation: estop-glow 2s infinite;
}
.btn-estop:hover { box-shadow: 0 0 25px rgba(255,0,0,0.6); }
@keyframes estop-glow { 0%,100% { box-shadow: 0 0 15px rgba(255,0,0,0.3); } 50% { box-shadow: 0 0 25px rgba(255,0,0,0.5); } }

/* ===== LOG STREAM ===== */
.log-panel {
    grid-area: log; margin: 0 3px 3px;
    background: var(--log-bg); border: 1px solid var(--border); border-radius: 6px;
    display: flex; flex-direction: column; overflow: hidden;
}
.log-header {
    background: var(--green); color: var(--accent);
    padding: 3px 8px; font-size: 0.65em; font-weight: 700;
    display: flex; justify-content: space-between; align-items: center;
}
.log-header .thinking { display: none; }
.log-header .thinking.active { display: inline; animation: pulse 1.5s infinite; }
.log-body {
    flex: 1; overflow-y: auto; padding: 3px 8px;
    font-family: 'Courier New', monospace; font-size: 0.62em;
    color: var(--text-mono); line-height: 1.3;
}
.log-body .log-error { color: #ff6b6b; }
.log-body .log-warn { color: #ffb800; }
.log-body .log-ok { color: #00ff88; }
.log-body .log-info { color: var(--accent); }
.log-body .log-system { color: #b088f9; font-weight: 700; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<!-- ===== HEADER ===== -->
<div class="header">
    <div class="header-title">
        <div>
            <h1>CRATER MISSION CONTROL</h1>
            <div class="subtitle">Team Crater | Autonomous Lunar Rover Navigation</div>
        </div>
    </div>
    <div class="header-status">
        <div class="tulane-badge">TULANE UNIVERSITY | CMPS 4020 | SPRING 2026</div>
        <div><span class="status-dot" id="hw-dot"></span> <small style="color:var(--text-dim)">HW</small></div>
        <div><span class="status-dot" id="cam-dot"></span> <small style="color:var(--text-dim)">CAM</small></div>
        <div class="mode-badge" id="mode-badge">MONITORING</div>
    </div>
</div>

<!-- ===== LEFT: MANUAL CONTROL ===== -->
<div class="left-panel" id="manual-panel">
    <div class="panel">
        <div class="panel-header">Manual Control</div>
        <div style="padding:2px; text-align:center;">
            <div class="wasd">
                <div class="wasd-key blank"></div>
                <div class="wasd-key" data-key="w" id="key-w">W</div>
                <div class="wasd-key blank"></div>
                <div class="wasd-key" data-key="a" id="key-a">A</div>
                <div class="wasd-key" data-key="s" id="key-s">S</div>
                <div class="wasd-key" data-key="d" id="key-d">D</div>
            </div>
            <div style="color:var(--text-dim);font-size:0.5em;margin-top:1px;">WASD or click</div>
        </div>
        <div class="ctrl-group">
            <div class="ctrl-label">Speed <span id="speed-val">0.15</span> m/s</div>
            <input type="range" id="speed-slider" min="0.05" max="0.30" step="0.01" value="0.15">
        </div>
        <div class="ctrl-group">
            <div class="ctrl-label">Turn <span id="turn-val">1.00</span> rad/s</div>
            <input type="range" id="turn-slider" min="0.1" max="2.0" step="0.05" value="1.00">
        </div>
        <div class="ctrl-group">
            <div class="ctrl-label">Comm Delay <span id="delay-val">0</span> ms</div>
            <input type="range" id="delay-slider" min="0" max="2000" step="50" value="0">
            <div style="color:var(--text-dim);font-size:0.55em;margin-top:2px;">Simulates space communication latency</div>
        </div>
    </div>
    <div class="panel">
        <div class="panel-header">Gimbal</div>
        <div class="ctrl-group">
            <div class="ctrl-label">Pan <span id="pan-val">0</span>&deg;</div>
            <input type="range" id="pan-slider" min="-90" max="90" step="1" value="0">
        </div>
        <div class="ctrl-group">
            <div class="ctrl-label">Tilt <span id="tilt-val">0</span>&deg;</div>
            <input type="range" id="tilt-slider" min="-30" max="60" step="1" value="0">
        </div>
        <button class="btn btn-start" style="margin:6px 10px;padding:6px;font-size:0.7em;" onclick="centerGimbal()">CENTER</button>
    </div>
    <div class="panel">
        <div class="panel-header">Lights</div>
        <div class="ctrl-group">
            <div class="ctrl-label">Base LEDs <span id="led-a-val">0</span></div>
            <input type="range" id="led-a" min="0" max="255" step="1" value="0">
        </div>
        <div class="ctrl-group">
            <div class="ctrl-label">Head LEDs <span id="led-b-val">0</span></div>
            <input type="range" id="led-b" min="0" max="255" step="1" value="0">
        </div>
    </div>
    <div class="panel">
        <div class="panel-header">Detection Overlay</div>
        <div class="ctrl-group">
            <label class="row" style="color:var(--text-dim);font-size:0.75em;display:flex;align-items:center;gap:8px;cursor:pointer;">
                <input type="checkbox" id="detect-toggle" style="accent-color:var(--accent);" onchange="toggleDetection()">
                Object Detection (MobileNet SSD)
            </label>
            <div style="color:var(--text-dim);font-size:0.6em;margin-top:4px;">Draws bounding boxes on RGB feed</div>
        </div>
    </div>
</div>

<!-- ===== CENTER ===== -->
<div class="center-panel">
    <div class="feed-hint">Click a feed to enlarge it (presenter mode)</div>
    <div class="feeds" id="camera-feeds">
        <div class="feed-box" id="feed-rgb" onclick="togglePresenter('feed-rgb')">
            <div class="feed-label">RGB + DETECTIONS</div>
            <img id="img-rgb" alt="RGB" />
            <div class="feed-info" id="info-rgb">
                FPS: <span class="info-val" id="fi-fps">--</span> |
                Status: <span class="info-val" id="fi-status">Loading...</span>
            </div>
        </div>
        <div class="feed-box" id="feed-depth" onclick="togglePresenter('feed-depth')">
            <div class="feed-label">DEPTH MAP</div>
            <img id="img-depth" alt="Depth" />
            <div class="feed-info" id="info-depth">
                Min: <span class="info-val" id="fi-min-d">--</span>m |
                L: <span class="info-val" id="fi-left-d">--</span>m |
                R: <span class="info-val" id="fi-right-d">--</span>m
            </div>
        </div>
        <div class="feed-box" id="feed-seg" onclick="togglePresenter('feed-seg')">
            <div class="feed-label">TERRAIN SEGMENTATION</div>
            <img id="img-seg" alt="Segmentation" />
            <div class="feed-info" id="info-seg">
                Terrain analysis <span class="info-val" id="fi-seg-status">{{ 'Active' if config.segmentation else 'Off' }}</span>
            </div>
        </div>
    </div>
    <div class="auto-overlay" id="auto-overlay">
        <div>
            <h2>AUTONOMOUS MODE ACTIVE</h2>
            <p style="color:var(--text-dim);margin-top:8px;font-size:0.9em;">Camera in use by RL controller. Monitor progress in logs below.</p>
        </div>
    </div>
    <div class="telemetry">
        <div class="gauge">
            <div class="gauge-label">Speed</div>
            <div class="gauge-value" id="g-speed">0.00</div>
            <div class="gauge-unit">m/s</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Heading</div>
            <div class="gauge-value" id="g-heading">0</div>
            <div class="gauge-unit">deg</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Min Depth</div>
            <div class="gauge-value" id="g-depth">--</div>
            <div class="gauge-unit">m</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Camera FPS</div>
            <div class="gauge-value" id="g-fps">--</div>
            <div class="gauge-unit">fps</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Mode</div>
            <div class="gauge-value" id="g-mode" style="font-size:0.9em;">IDLE</div>
            <div class="gauge-unit">&nbsp;</div>
        </div>
        <div class="gauge">
            <div class="gauge-label">Face State</div>
            <div class="gauge-value" id="g-face" style="font-size:0.8em;">--</div>
            <div class="gauge-unit">&nbsp;</div>
        </div>
    </div>
</div>

<!-- ===== RIGHT: MISSION CONFIG ===== -->
<div class="right-panel">
    <div class="panel" style="flex:1;">
        <div class="panel-header">Mission Configuration</div>
        <div class="config-form" id="config-form">
            <label>Model Path
                <input type="text" id="cfg-model" value="~/srb-waypoint_navigation_visual.zip">
            </label>
            <label class="row">
                <input type="checkbox" id="cfg-demo"> Demo (no camera)
            </label>
            <label class="row">
                <input type="checkbox" id="cfg-dry-run"> Dry Run (no motors)
            </label>
            <label>Max Speed (m/s)
                <div style="display:flex;align-items:center;gap:8px;">
                    <input type="range" id="cfg-speed" min="0.05" max="0.30" step="0.01" value="0.15" style="flex:1;">
                    <span id="cfg-speed-val" style="color:var(--accent);font-weight:700;min-width:35px;">0.15</span>
                </div>
            </label>
            <label>Duration (seconds, blank=unlimited)
                <input type="number" id="cfg-duration" placeholder="e.g. 120" min="1">
            </label>
            <label>Target Distance (meters, blank=none)
                <input type="number" id="cfg-distance" placeholder="e.g. 5" min="0.1" step="0.1">
            </label>
            <label>Serial Port
                <input type="text" id="cfg-port" value="/dev/ttyTHS1">
            </label>
            <label class="row">
                <input type="checkbox" id="cfg-detect" checked> Object Detection
            </label>
            <label class="row">
                <input type="checkbox" id="cfg-gimbal" checked> Gimbal Scanning
            </label>
            <label>Target Mode
                <select id="cfg-target" onchange="toggleFaceOpts()">
                    <option value="distance">Distance Navigation</option>
                    <option value="face">Face Navigation</option>
                    <option value="red">Red Flag Target</option>
                </select>
            </label>
            <div class="face-opts" id="face-opts">
                <label>Stop Distance (m)
                    <input type="number" id="cfg-face-dist" value="0.8" min="0.05" step="0.05">
                </label>
            </div>
            <label class="row">
                <input type="checkbox" id="cfg-seg"> Terrain Segmentation
            </label>
            <label>Seg Model Path
                <input type="text" id="cfg-seg-model" value="mark_model/unet_lunar_segmentation.pth" style="font-size:0.7em;">
            </label>
            <label class="row">
                <input type="checkbox" id="cfg-rewards"> Log Rewards
            </label>
        </div>
        <div class="btn-group">
            <button class="btn btn-start" id="btn-start" onclick="startMission()">START MISSION</button>
            <button class="btn btn-stop" id="btn-stop" onclick="stopMission()">STOP MISSION</button>
            <button class="btn btn-estop" onclick="eStop()">E-STOP</button>
            <button class="btn" style="background:#334;color:var(--text-dim);font-size:0.65em;margin-top:4px;" onclick="resetAll()">🔄 RESET ALL</button>
        </div>
    </div>
</div>

<!-- ===== LOG STREAM ===== -->
<div class="log-panel">
    <div class="log-header">
        <div>ROVER LOG <span class="thinking" id="thinking-indicator">&#x1F9E0; THINKING...</span></div>
        <div>
            <label style="font-size:0.8em;cursor:pointer;">
                <input type="checkbox" id="log-autoscroll" checked style="accent-color:var(--accent);"> Auto-scroll
            </label>
        </div>
    </div>
    <div class="log-body" id="log-body">
        <div class="log-system">[SYSTEM] Mission Control initialized. Ready.</div>
    </div>
</div>

<!-- ===== JAVASCRIPT ===== -->
<script>
const socket = io();
let autonomousRunning = false;

// ===== CAMERA FEEDS — MJPEG streams with auto-reconnect =====
// MJPEG streams are persistent HTTP connections — much lower latency than
// snapshot polling. If the stream dies (mode transition), we auto-reconnect.
function startMjpeg(imgId, streamUrl) {
    const img = document.getElementById(imgId);
    if (!img) return;
    // Set the src to the MJPEG stream — browser handles everything
    img.src = streamUrl;
    // Auto-reconnect on error (stream dies during mode transitions)
    img.onerror = () => {
        setTimeout(() => { img.src = streamUrl + '?t=' + Date.now(); }, 1000);
    };
}
startMjpeg('img-rgb', '/stream/rgb');
startMjpeg('img-depth', '/stream/depth');
// Seg uses snapshot polling — MJPEG would need a 3rd persistent thread
// and Werkzeug's thread pool can't handle it alongside rgb+depth+socketio.
// Seg only updates every ~2s anyway (1.7s PyTorch inference).
function pollSeg() {
    const img = document.getElementById('img-seg');
    if (!img) return;
    let loading = false;
    setInterval(() => {
        if (loading) return;
        loading = true;
        const tmp = new Image();
        tmp.onload = () => { img.src = tmp.src; loading = false; };
        tmp.onerror = () => { loading = false; };
        tmp.src = '/snapshot/seg?t=' + Date.now();
    }, 2000);  // poll every 2s (matches inference rate)
}
pollSeg();

// ===== CONNECTION STATUS =====
socket.on('connect', () => {
    addLog('[SYSTEM] Connected to mission control server', 'system');
});
socket.on('disconnect', () => {
    addLog('[SYSTEM] Disconnected from server', 'error');
});

// ===== MISSION STATE =====
socket.on('mission_state', (data) => {
    autonomousRunning = data.running;
    updateUI();
});
socket.on('mission_ended', (data) => {
    autonomousRunning = false;
    updateUI();
    addLog(`[SYSTEM] Mission ended (exit code: ${data.code})`, 'system');
});

function updateUI() {
    const badge = document.getElementById('mode-badge');
    const feeds = document.getElementById('camera-feeds');
    const overlay = document.getElementById('auto-overlay');
    const manual = document.getElementById('manual-panel');
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const thinking = document.getElementById('thinking-indicator');

    if (autonomousRunning) {
        badge.textContent = 'AUTONOMOUS';
        badge.classList.add('autonomous');
        feeds.style.display = 'grid';
        overlay.classList.remove('active');
        manual.classList.add('disabled');
        btnStart.style.display = 'none';
        btnStop.style.display = 'block';
        thinking.classList.add('active');
    } else {
        badge.textContent = 'MONITORING';
        badge.classList.remove('autonomous');
        feeds.style.display = 'grid';
        overlay.classList.remove('active');
        manual.classList.remove('disabled');
        btnStart.style.display = 'block';
        btnStop.style.display = 'none';
        thinking.classList.remove('active');
    }
}

// ===== LOG STREAM =====
socket.on('log_line', (data) => {
    let cls = '';
    const t = data.text;
    if (t.includes('[ERROR]') || t.includes('[EMERGENCY]')) cls = 'log-error';
    else if (t.includes('[WARN]')) cls = 'log-warn';
    else if (t.includes('[OK]')) cls = 'log-ok';
    else if (t.includes('[INFO]') || t.includes('[NAV]')) cls = 'log-info';
    else if (t.includes('[SYSTEM]')) cls = 'log-system';
    addLog(t, cls);
});

function addLog(text, cls) {
    const body = document.getElementById('log-body');
    const div = document.createElement('div');
    if (cls) div.className = cls.startsWith('log-') ? cls : 'log-' + cls;
    div.textContent = text;
    body.appendChild(div);
    // Keep log size manageable
    while (body.children.length > 500) body.removeChild(body.firstChild);
    if (document.getElementById('log-autoscroll').checked) {
        body.scrollTop = body.scrollHeight;
    }
}

// ===== TELEMETRY =====
socket.on('telemetry', (data) => {
    if (data.speed !== undefined) document.getElementById('g-speed').textContent = data.speed.toFixed(2);
    if (data.heading !== undefined) document.getElementById('g-heading').textContent = Math.round(data.heading);
    if (data.min_depth !== undefined) document.getElementById('g-depth').textContent = data.min_depth.toFixed(2);
    if (data.distance !== undefined) document.getElementById('g-fps').textContent = data.distance.toFixed(1);
    if (data.mode) document.getElementById('g-mode').textContent = data.mode;
    if (data.face_state) document.getElementById('g-face').textContent = data.face_state;
});

// Poll camera telemetry in monitoring mode
setInterval(async () => {
    try {
        const r = await fetch('/api/status');
        const d = await r.json();
        document.getElementById('hw-dot').className = 'status-dot ' + (d.hardware ? 'on' : 'off');
        document.getElementById('cam-dot').className = 'status-dot ' + (d.camera_active ? 'on' : 'off');
        if (!autonomousRunning) {
            if (d.min_depth !== undefined) document.getElementById('g-depth').textContent = d.min_depth.toFixed(2);
            document.getElementById('g-speed').textContent = '0.00';
            document.getElementById('g-fps').textContent = d.fps || '--';
            // Update feed info panels
            document.getElementById('fi-fps').textContent = d.fps || '--';
            document.getElementById('fi-status').textContent = d.camera_active ? 'Streaming' : 'Off';
            if (d.min_depth !== undefined) document.getElementById('fi-min-d').textContent = d.min_depth.toFixed(2);
            if (d.left_depth !== undefined) document.getElementById('fi-left-d').textContent = d.left_depth.toFixed(2);
            if (d.right_depth !== undefined) document.getElementById('fi-right-d').textContent = d.right_depth.toFixed(2);
        }
    } catch(e) {}
}, 2000);

// ===== MANUAL DRIVE (WASD) =====
const keysDown = new Set();
let driveInterval = null;

document.addEventListener('keydown', (e) => {
    if (autonomousRunning) return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    const k = e.key.toLowerCase();
    if ('wasd'.includes(k) && k.length === 1) {
        e.preventDefault();
        keysDown.add(k);
        document.getElementById('key-' + k)?.classList.add('active');
        if (!driveInterval) driveInterval = setInterval(sendDrive, 50);
    }
});

document.addEventListener('keyup', (e) => {
    const k = e.key.toLowerCase();
    keysDown.delete(k);
    document.getElementById('key-' + k)?.classList.remove('active');
    if (keysDown.size === 0 && driveInterval) {
        clearInterval(driveInterval);
        driveInterval = null;
        socket.emit('manual_stop');
    }
});

// Mouse/touch on WASD keys
document.querySelectorAll('.wasd-key[data-key]').forEach(el => {
    const k = el.dataset.key;
    el.addEventListener('mousedown', (e) => { e.preventDefault(); keysDown.add(k); el.classList.add('active'); if(!driveInterval) driveInterval = setInterval(sendDrive,50); });
    el.addEventListener('mouseup', () => { keysDown.delete(k); el.classList.remove('active'); if(keysDown.size===0){clearInterval(driveInterval);driveInterval=null;socket.emit('manual_stop');} });
    el.addEventListener('mouseleave', () => { keysDown.delete(k); el.classList.remove('active'); if(keysDown.size===0&&driveInterval){clearInterval(driveInterval);driveInterval=null;socket.emit('manual_stop');} });
});

function sendDrive() {
    let lin = 0, ang = 0;
    const spd = parseFloat(document.getElementById('speed-slider').value);
    const turnSpd = parseFloat(document.getElementById('turn-slider').value);
    const commDelay = parseInt(document.getElementById('delay-slider').value);
    if (keysDown.has('w')) lin = spd;
    if (keysDown.has('s')) lin = -spd;
    if (keysDown.has('a')) ang = turnSpd;
    if (keysDown.has('d')) ang = -turnSpd;
    const cmd = {linear: lin, angular: ang};
    if (commDelay > 0) {
        setTimeout(() => socket.emit('manual_drive', cmd), commDelay);
    } else {
        socket.emit('manual_drive', cmd);
    }
}

// Speed slider display
document.getElementById('speed-slider').addEventListener('input', (e) => {
    document.getElementById('speed-val').textContent = parseFloat(e.target.value).toFixed(2);
});
// Turn slider display
document.getElementById('turn-slider').addEventListener('input', (e) => {
    document.getElementById('turn-val').textContent = parseFloat(e.target.value).toFixed(2);
});
// Comm delay slider display
document.getElementById('delay-slider').addEventListener('input', (e) => {
    document.getElementById('delay-val').textContent = e.target.value;
});

// ===== GIMBAL =====
document.getElementById('pan-slider').addEventListener('input', (e) => {
    document.getElementById('pan-val').textContent = e.target.value;
    socket.emit('gimbal_move', {pan: parseInt(e.target.value), tilt: parseInt(document.getElementById('tilt-slider').value)});
});
document.getElementById('tilt-slider').addEventListener('input', (e) => {
    document.getElementById('tilt-val').textContent = e.target.value;
    socket.emit('gimbal_move', {pan: parseInt(document.getElementById('pan-slider').value), tilt: parseInt(e.target.value)});
});
function centerGimbal() {
    document.getElementById('pan-slider').value = 0;
    document.getElementById('tilt-slider').value = 0;
    document.getElementById('pan-val').textContent = '0';
    document.getElementById('tilt-val').textContent = '0';
    socket.emit('gimbal_move', {pan: 0, tilt: 0});
}

// ===== LIGHTS =====
document.getElementById('led-a').addEventListener('input', (e) => {
    document.getElementById('led-a-val').textContent = e.target.value;
    socket.emit('lights_set', {a: parseInt(e.target.value), b: parseInt(document.getElementById('led-b').value)});
});
document.getElementById('led-b').addEventListener('input', (e) => {
    document.getElementById('led-b-val').textContent = e.target.value;
    socket.emit('lights_set', {a: parseInt(document.getElementById('led-a').value), b: parseInt(e.target.value)});
});

// ===== CONFIG =====
document.getElementById('cfg-speed').addEventListener('input', (e) => {
    document.getElementById('cfg-speed-val').textContent = parseFloat(e.target.value).toFixed(2);
});

function toggleFaceOpts() {
    const mode = document.getElementById('cfg-target').value;
    const show = mode === 'face' || mode === 'red';
    document.getElementById('face-opts').classList.toggle('show', show);
}

function getConfig() {
    return {
        model: document.getElementById('cfg-model').value,
        demo: document.getElementById('cfg-demo').checked,
        dry_run: document.getElementById('cfg-dry-run').checked,
        max_speed: parseFloat(document.getElementById('cfg-speed').value),
        duration: document.getElementById('cfg-duration').value ? parseFloat(document.getElementById('cfg-duration').value) : null,
        distance: document.getElementById('cfg-distance').value ? parseFloat(document.getElementById('cfg-distance').value) : null,
        port: document.getElementById('cfg-port').value,
        enable_detect: document.getElementById('cfg-detect').checked,
        enable_gimbal: document.getElementById('cfg-gimbal').checked,
        target_mode: document.getElementById('cfg-target').value,
        face_distance: parseFloat(document.getElementById('cfg-face-dist').value),
        segmentation: document.getElementById('cfg-seg').checked,
        seg_model: document.getElementById('cfg-seg-model').value,
        log_rewards: document.getElementById('cfg-rewards').checked,
    };
}

function startMission() {
    if (autonomousRunning) return;
    const cfg = getConfig();
    addLog('[SYSTEM] Requesting mission start...', 'system');
    addLog('[SYSTEM] Config: ' + JSON.stringify(cfg, null, 0), 'info');
    socket.emit('start_mission', cfg);
}

function stopMission() {
    addLog('[SYSTEM] Requesting mission stop...', 'system');
    socket.emit('stop_mission');
}

function eStop() {
    addLog('[EMERGENCY] E-STOP ACTIVATED!', 'error');
    socket.emit('e_stop');
}

function resetAll() {
    if (!confirm('Reset everything? This kills all processes and restarts cameras.')) return;
    addLog('[SYSTEM] Full reset requested...', 'system');
    socket.emit('reset_all');
}

// ===== PRESENTER MODE (click feed to zoom) =====
let activePrimary = null;

function togglePresenter(feedId) {
    const container = document.getElementById('camera-feeds');
    const boxes = container.querySelectorAll('.feed-box');
    const clicked = document.getElementById(feedId);

    if (activePrimary === feedId) {
        // Click the same one again — exit presenter mode
        activePrimary = null;
        container.classList.remove('presenter');
        boxes.forEach(b => {
            b.classList.remove('primary');
            b.style.order = '';
        });
    } else {
        // Enter presenter mode with this feed as primary
        activePrimary = feedId;
        container.classList.add('presenter');
        boxes.forEach(b => {
            b.classList.remove('primary');
            b.style.order = '';
        });
        clicked.classList.add('primary');
        clicked.style.order = '-1';  // ensure primary appears first in grid
    }
}

// ===== DETECTION TOGGLE =====
function toggleDetection() {
    const enabled = document.getElementById('detect-toggle').checked;
    socket.emit('set_detection', {enabled: enabled});
    addLog('[SYSTEM] Detection overlay ' + (enabled ? 'ON' : 'OFF'), 'system');
}

// E-STOP on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') eStop();
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Crater Mission Control Panel')
    parser.add_argument('--port', type=int, default=5050,
                        help='Web server port (default: 5050)')
    parser.add_argument('--no-hardware', action='store_true',
                        help='Run without rover hardware (UI testing)')
    parser.add_argument('--segmentation', action='store_true', default=True,
                        help='Enable terrain segmentation in camera view (default: ON)')
    parser.add_argument('--no-segmentation', action='store_true',
                        help='Disable terrain segmentation')
    parser.add_argument('--seg-model', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'mark_model', 'unet_lunar_segmentation.pth'),
                        help='Path to UNet segmentation model')
    parser.add_argument('--no-camera', action='store_true',
                        help='Run without camera (UI + manual control only)')
    args = parser.parse_args()

    # --no-segmentation overrides --segmentation default
    if args.no_segmentation:
        args.segmentation = False

    global rover, camera_mgr, proc_mgr, _app_config

    _app_config = {
        'segmentation': args.segmentation,
        'port': args.port,
    }

    # Initialize hardware
    if args.no_hardware:
        rover = DummyRoverHardware()
        print("[INIT] No-hardware mode — manual control disabled")
    else:
        rover = RoverHardware()

    # Initialize camera
    camera_mgr = CameraManager(
        enable_seg=args.segmentation,
        seg_model_path=args.seg_model,
    )
    if not args.no_camera and not args.no_hardware:
        camera_mgr.start()

    # Initialize subprocess manager
    proc_mgr = SubprocessManager(socketio)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║   CRATER MISSION CONTROL                                      ║
║   Team Crater | Tulane University | CMPS 4020                 ║
║                                                               ║
║   Web Panel:  http://0.0.0.0:{args.port:<5}                        ║
║   Tailscale:  http://100.68.244.40:{args.port:<5}                   ║
║   Hardware:   {'Connected' if rover.is_connected else 'Not connected':44}║
║   Camera:     {'Starting...' if camera_mgr.is_active else 'Off':44}║
║   Segmentation: {'ON' if args.segmentation else 'OFF':41}║
║                                                               ║
║   Press Ctrl+C to shut down                                   ║
╚═══════════════════════════════════════════════════════════════╝
""")

    try:
        socketio.run(app, host='0.0.0.0', port=args.port,
                     allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping...")
    finally:
        if proc_mgr and proc_mgr.is_running:
            proc_mgr.e_stop()
        if camera_mgr:
            camera_mgr.stop()
        if hasattr(rover, 'stop'):
            rover.stop()
        print("[SHUTDOWN] Done.")


if __name__ == '__main__':
    main()
