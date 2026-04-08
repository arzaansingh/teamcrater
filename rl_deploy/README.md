# Team Crater — Autonomous Lunar Rover Navigation

> **CMPS 4020 Senior Capstone** | Tulane University | Spring 2026
>
> Autonomous point-to-point navigation using reinforcement learning and computer vision, trained in simulation and deployed on a physical rover.

**Team:** Mary Ella Scroggie (Simulation), Arzaan Singh (RL & Deployment), Mark Zhai (Computer Vision)
**Faculty Advisors:** Dr. Zhengming Ding & Dr. Zizhan Zheng, Department of Computer Science

---

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware](#hardware)
- [Software Architecture](#software-architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Mission Control Web UI](#mission-control-web-ui)
- [RL Autonomous Navigation](#rl-autonomous-navigation)
- [Face Navigation Mode](#face-navigation-mode)
- [Three-Tier Safety System](#three-tier-safety-system)
- [Computer Vision — Terrain Segmentation](#computer-vision--terrain-segmentation)
- [3D Coordinate Tracking](#3d-coordinate-tracking)
- [Simulation — Space Robotics Bench](#simulation--space-robotics-bench)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

Team Crater builds an autonomous navigation system for a six-wheeled rover that drives from point A to point B while avoiding obstacles — no human control required.

**Three integrated components:**

| Component | Lead | Description |
|-----------|------|-------------|
| **Simulation Environment** | Mary Ella | NVIDIA Space Robotics Bench (Isaac Sim) + custom Unity environments with procedural lunar terrain |
| **Reinforcement Learning** | Arzaan | PPO policy trained on 120M timesteps in simulation, deployed on physical rover at 5 Hz |
| **Computer Vision** | Mark | MobileNet-V2 UNet terrain segmentation trained on 9,766 lunar renders |

**Key Innovation:** The RL model is trained entirely in simulation and transferred (sim-to-real) onto the physical rover's Jetson Orin Nano using depth-only observations, bridging the gap between virtual training and real-world autonomy.

---

## Hardware

| Component | Details |
|-----------|---------|
| **Rover** | Waveshare UGV Rover PT — 6-wheel differential drive |
| **Compute** | NVIDIA Jetson Orin Nano (8 GB), Ubuntu 22.04 |
| **Lower Controller** | ESP32 — motors, encoders, IMU, voltage sensing |
| **Camera** | OAK-D Lite — stereo depth + RGB (128x128 depth, 640x480 RGB) |
| **Gimbal** | Pan-tilt servo mount for camera scanning and active tracking |
| **Connection** | Tailscale VPN (IP: `100.68.244.40`), SSH user/pass: `jetson/jetson` |
| **Serial** | UART `/dev/ttyTHS0` at 115200 baud, JSON + newline protocol |

**Key specs:** Max speed 0.3 m/s (software-limited), 5 Hz control loop, heartbeat auto-stop if no command for ~3 seconds.

---

## Software Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  mission_control.py — Web UI (Flask + SocketIO, port 5000)  │
│  Live camera feeds, manual WASD control, mission launcher   │
└──────────────────────────┬──────────────────────────────────┘
                           │ launches as subprocess
┌──────────────────────────▼──────────────────────────────────┐
│  rl_deploy.py — Autonomous Navigation Engine                │
│                                                             │
│  ┌─ RL Model (PPO) ────── Trained policy from SRB          │
│  ├─ Face/Red Tracker ──── 3D coordinate tracking + gimbal   │
│  ├─ Safety System ──────── EMERGENCY→BACKUP→AVOID→SLOW     │
│  ├─ Course Correction ─── Heading-based recovery            │
│  ├─ Obstacle Memory ───── World-frame position tracking     │
│  └─ Odometry ──────────── Encoder-based pose estimation     │
└──────────────────────────┬──────────────────────────────────┘
                           │ JSON over UART
┌──────────────────────────▼──────────────────────────────────┐
│  rover_control.py — Hardware Abstraction Layer              │
│  CameraManager (MJPEG streams), background segmentation     │
│  RoverHardware wrapper for BaseController                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  ESP32 Lower Controller — Motors / Encoders / IMU / Voltage │
└─────────────────────────────────────────────────────────────┘
```

**Design principle:** Each module is separate but integrated. Changing the web UI doesn't touch hardware code. Changing hardware abstraction doesn't touch RL logic. The RL model runs as a subprocess — completely isolated.

---

## Repository Structure

```
teamcrater/
├── capstone_rl/                  # RL training code for SRB
│   ├── algos/                    # PPO training and evaluation
│   ├── envs/                     # Environment wrappers
│   ├── scripts/                  # Plotting utilities
│   └── logs/                     # Training logs
│
├── ComputerVisionModelHPC/       # Mark's CV model training (LONI HPC)
│   ├── train.py                  # UNet training script
│   ├── dataset.py                # Lunar landscape dataset loader
│   └── organize_dataset.py       # Dataset preparation
│
├── rl_deploy/                    # Deployment on physical rover
│   ├── rl_deploy.py              # Main autonomy engine (2400+ lines)
│   ├── mission_control.py        # Web UI server (Flask + SocketIO)
│   ├── rover_control.py          # Hardware abstraction layer
│   ├── rover_patterns.py         # Demo movement patterns
│   ├── start.sh                  # One-command launcher
│   ├── test_rl_deploy.py         # 117+ unit tests (rl_deploy)
│   ├── test_mission_control.py   # 117 unit tests (web UI)
│   └── mark_model/               # Terrain segmentation model
│       └── unet_lunar_segmentation.pth
│
├── MaskScoringFunctionipynb.ipynb    # CV mask scoring notebook
├── Testingthemodel.ipynb             # CV model testing
└── computervisionbasemodel.ipynb     # CV base model training
```

---

## Quick Start

### One-Command Launch (on Jetson)

```bash
ssh jetson@100.68.244.40
cd ~/rl_deploy
./start.sh
```

This script:
1. Kills any existing server/camera processes
2. Releases the camera device
3. Waits for hardware to settle
4. Cleans Python cache
5. Starts the mission control server on port 5000

Then open **http://100.68.244.40:5000** from any browser on the same Tailscale network.

### Manual Setup

```bash
# Install dependencies
pip3 install pyserial flask flask-socketio opencv-python numpy torch depthai

# Start mission control
sudo -E python3 mission_control.py

# Or run RL autonomy directly
sudo -E python3 rl_deploy.py --distance 5 --max-speed 0.15
```

---

## Mission Control Web UI

The web-based mission control panel provides remote monitoring and control of the rover.

**Features:**
- Live MJPEG camera feeds (RGB, depth heatmap, terrain segmentation)
- WASD keyboard manual driving + speed slider
- Gimbal pan/tilt control
- One-click autonomous mission launch with configurable parameters
- Real-time log streaming from the autonomy engine
- E-STOP button for emergency shutdown
- Reset All for full state cleanup

**Camera Feeds:**
- **RGB + Depth**: MJPEG streams for low-latency viewing
- **Segmentation**: Snapshot polling (2s interval) to avoid thread pool limits
- During autonomous mode: reads frames written by `rl_deploy.py` to `/tmp/rover_frames/`

**Mode Transitions:**
- **Monitoring → Autonomous:** Camera released, serial handed off, subprocess launched
- **Autonomous → Monitoring:** Subprocess stopped, hardware reacquired, feeds resume

---

## RL Autonomous Navigation

The core autonomy engine (`rl_deploy.py`) loads a trained PPO policy and runs a 5 Hz control loop.

### How It Works

Each cycle at 5 Hz:
1. **Capture** 128x128 depth frame from OAK-D Lite
2. **Safety check** — depth thresholds can override everything
3. **Face/target detection** (if enabled) — encode target as state vector
4. **RL inference** — PPO model outputs `[linear, angular]` velocities
5. **Override priority** — Safety > Avoidance > Course correction > RL output
6. **Motor command** — sent to ESP32 via UART

### CLI Reference

```bash
# Full autonomous run — 5 meters, 5 min timeout
sudo -E python3 rl_deploy.py --max-speed 0.15 --distance 5 --duration 300

# Face tracking mode
sudo -E python3 rl_deploy.py --target face --face-distance 1.5 --duration 120

# With terrain segmentation
sudo -E python3 rl_deploy.py --distance 5 --segmentation

# Dry run (prints commands, doesn't move)
sudo -E python3 rl_deploy.py --dry-run --distance 5

# Demo mode (no camera, no hardware)
python3 rl_deploy.py --demo --dry-run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `~/srb-waypoint_navigation_visual.zip` | Trained PPO model path |
| `--max-speed` | `0.3` | Max linear speed (m/s) |
| `--distance` | None | Target distance in meters |
| `--duration` | None | Max runtime in seconds |
| `--target` | `distance` | Target mode: `distance` or `face` |
| `--face-distance` | `1.0` | Stop distance for face mode (m) |
| `--demo` | off | Dummy camera (no hardware) |
| `--dry-run` | off | Print commands only |
| `--no-detect` | off | Disable object detection |
| `--no-gimbal` | off | Disable gimbal scanning |
| `--segmentation` | off | Enable terrain segmentation |
| `--log-rewards` | off | Log per-step rewards to CSV |

---

## Face Navigation Mode

Navigate toward a detected person using `--target face`. The face detector finds a face, computes a 3D angle and distance, and encodes it as the RL model's "waypoint" state vector. The RL model then steers toward the face naturally.

**State machine:**

| State | Behavior |
|-------|----------|
| **SEARCH** | Rotate in place to find a face, gimbal scans |
| **NAVIGATE** | Face detected — RL model steers toward it, gimbal tracks actively |
| **LOST_HOLD** | Face lost — hold direction for 1s, then re-search |
| **ARRIVED** | Within stop distance — motors halt |

**3D tracking:** The system computes gimbal-compensated horizontal and vertical angles using the camera's 73-degree horizontal and 58-degree vertical FOV, then estimates the target's world-frame (x, y, z) position.

---

## Three-Tier Safety System

Depth-based safety tiers that override the RL model when obstacles are close:

| Tier | Distance | Action |
|------|----------|--------|
| **EMERGENCY** | <= 5 cm | Hard stop, 1s backup, committed avoidance turn |
| **BACKUP** | <= 15 cm | Active reverse with turning |
| **AVOID** | <= 25 cm | Committed 3-second turn (auto-extends up to 8s) |
| **SLOW** | <= 45 cm | Reduced speed, RL model continues navigating |

**Stereo dead zone handling:** The OAK-D Lite returns 0 depth for objects closer than ~20 cm. Detected via 85% zero-pixel threshold, which triggers the emergency protocol.

**Additional safety features:**
- Obstacle memory: records obstacle world-frame positions for 15 seconds
- Course correction: proportional heading control after avoidance maneuvers
- Heartbeat: rover auto-stops if no command received for 3 seconds

---

## Computer Vision — Terrain Segmentation

Mark's UNet model (MobileNet-V2 encoder) segments the RGB camera feed into traversable terrain vs. obstacles.

**Model details:**
- Trained on 9,766 realistic lunar renders (Artificial Lunar Rocky Landscape Dataset)
- 4 classes: ground, sky, small rock, big rock
- Input: RGB image resized to 480x480
- Output: per-pixel class mask + directional feasibility scores

**Integration with navigation:**
- Runs in a background thread (0.6 FPS inference, 31 FPS main capture)
- Frame divided into 5 directional sectors, each scored for feasibility
- When the avoidance system faces an ambiguous left-vs-right choice, it prefers the side with fewer rocks
- Segmentation mask displayed live in the mission control web UI

---

## 3D Coordinate Tracking

The system uses gimbal-compensated angles to estimate target positions in 3D world coordinates.

**How it works:**
1. Detect target (face or red marker) in RGB frame
2. Compute horizontal angle from frame center (73-degree HFOV)
3. Compute vertical angle from frame center (58-degree VFOV)
4. Add gimbal pan/tilt offsets for total world-relative angles
5. Sample depth at target location
6. Compute 3D world position: `(x, y, z)` relative to rover

**Active gimbal tracking:** During navigation, the gimbal tilts to keep the target in view, clamped to [-20, +25] degrees, updating when the change exceeds 3 degrees.

---

## Simulation — Space Robotics Bench

### Training Setup

- **Platform:** NVIDIA Isaac Sim via Space Robotics Bench (SRB)
- **Task:** `waypoint_navigation_visual` — navigate to waypoints on lunar terrain
- **Algorithm:** PPO via RL-Games library
- **Training:** ~120 million timesteps, 64 parallel environments
- **Model size:** ~155 MB `.zip` file

### Observation and Action Spaces

```
Observation:
  state:       [4] — relative direction/distance to waypoint
  image_front: [128, 128, 1] — depth map in raw meters (0 = close, 10 = far)

Action:
  [2] — [linear_velocity, angular_velocity] in [-1, 1]
```

### Training Command

```bash
python3 train.py task=waypoint_navigation_visual \
    env.camera_data_types=[depth] \
    env.camera_resolution=[128,128] \
    num_envs=64
```

**Note:** LiDAR has geometry bugs in the current SRB version — use depth camera only.

### Custom Unity Environment (Mary Ella)

A parallel Unity environment was developed with:
- Procedural lunar terrain (craters, elevation variation, realistic lighting)
- 6-wheel rover model matching the physical Waveshare platform
- Integration point for ML segmentation model as RL observation input

---

## Testing

234+ total unit tests across two test suites, all running without hardware:

```bash
# RL deploy tests
python3 -m pytest test_rl_deploy.py -v

# Mission control tests
python3 -m pytest test_mission_control.py -v

# Run all
python3 -m pytest test_rl_deploy.py test_mission_control.py -v
```

**Test coverage:**

| Area | Tests | Covers |
|------|-------|--------|
| Depth processing | 4 | Frame format, normalization, clipping |
| Reactive avoidance | 7 | Committed turns, direction, extension |
| Course correction | 5 | Proportional control, deadband, clamping |
| Obstacle memory | 9 | World-frame tracking, pruning, nearest |
| Safety tiers | 4 | EMERGENCY/BACKUP/AVOID/SLOW thresholds |
| Object detection | 4 | Danger scoring, center weighting |
| Gimbal scanner | 6 | Scan cycle, centering, settle steps |
| Odometry | 9 | Encoder integration, heading, displacement |
| Destination | 5 | Displacement-based goal tracking |
| Face tracker | 6 | Angle computation, depth mapping, 3D coords |
| Face navigation SM | 8 | State transitions, search/navigate/lost/arrived |
| Terrain segmenter | 4 | Feasibility scoring, left/right split |
| 3D tracking | 10 | Gimbal compensation, world position, vertical angles |
| Reward logger | 8 | Reward shaping, CSV output |
| Mission control web | 117 | Routes, MJPEG streams, SocketIO events, config |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied: /dev/ttyTHS0` | Use `sudo -E python3 ...` |
| Camera not releasing | Run `fuser -k /dev/video*` or use `start.sh` |
| RGB feed garbled (pink/green) | Delete `__pycache__/` and restart |
| Segmentation never shows | Ensure `--segmentation` flag or seg model exists |
| Rover spins endlessly | Check gimbal tilt (should be 0), verify safety thresholds |
| SCP/SSH timeout | Check Tailscale: `tailscale status`, ping `100.68.244.40` |
| `No module named 'base_ctrl'` | `ln -sfn ~/ugv_jetson ~/rl_deploy/ugv_jetson` |

---

## Acknowledgments

- **Dr. Zhengming Ding & Dr. Zizhan Zheng** — Faculty advisors
- **Waveshare** — UGV Rover PT hardware and base software
- **NVIDIA** — Space Robotics Bench, Isaac Sim, Jetson platform
- **Luxonis** — OAK-D Lite depth camera and DepthAI SDK
- **RL-Games** — PPO implementation for RL training
- **Tulane University School of Science & Engineering** — Capstone program support
