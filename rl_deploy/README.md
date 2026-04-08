# Team Crater — Autonomous Lunar Rover Navigation

> **CMPS 4010/4020 Senior Capstone** | Tulane University | 2025–2026
>
> Autonomous point-to-point navigation using reinforcement learning and computer vision, trained in simulation and deployed on a physical rover.

**Team Members:** Mary Ella Petersen (Simulation), Arzaan Irani (RL & Deployment), Mark Tikhonov (Computer Vision)
**Advisor:** Dr. Jihun Hamm, Department of Computer Science

---

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware](#hardware)
- [Software Architecture](#software-architecture)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
  - [Jetson Setup](#1-jetson-orin-nano-setup)
  - [Workstation Setup (SRB Training)](#2-workstation-setup-srb-training)
- [Running the Rover](#running-the-rover)
  - [Connecting to the Rover](#connecting-to-the-rover)
  - [RL Autonomous Navigation](#rl-autonomous-navigation-rl_deploypy)
  - [Pattern Demonstrations](#pattern-demonstrations-rover_patternspy)
  - [Web UI Control](#web-ui-control)
  - [ROS 2 Control](#ros-2-control)
- [RL Deployment — Detailed Reference](#rl-deployment--detailed-reference)
  - [Navigation Architecture](#navigation-architecture-7-layers)
  - [Three-Tier Safety System](#three-tier-safety-system)
  - [Configuration Parameters](#configuration-parameters)
  - [Depth Camera Processing](#depth-camera-processing)
  - [Object Detection](#object-detection)
  - [Gimbal Scanning](#gimbal-scanning)
  - [Course Correction & Obstacle Memory](#course-correction--obstacle-memory)
  - [Odometry & Destination Tracking](#odometry--destination-tracking)
  - [Face Navigation Mode](#face-navigation-mode)
  - [Terrain Segmentation (Mark's UNet)](#terrain-segmentation-marks-unet)
  - [Reward Logger](#reward-logger)
- [Full Pipeline Explanation](#full-pipeline-explanation)
- [Testing](#testing)
- [Serial Command Reference](#serial-command-reference)
- [Simulation — Space Robotics Bench (SRB)](#simulation--space-robotics-bench-srb)
- [Troubleshooting](#troubleshooting)
- [Future Plans](#future-plans)

---

## Project Overview

Team Crater builds an autonomous navigation system for a six-wheeled rover that drives from point A to point B while avoiding obstacles — no human control required. We combine:

1. **Reinforcement Learning** — A PPO policy trained in NVIDIA's Space Robotics Bench (SRB) on simulated lunar terrain, learning to navigate toward waypoints using depth images.
2. **Computer Vision** — MobileNet-SSD object detection classifying obstacles in real time on the rover's RGB camera.
3. **Reactive Safety** — A layered avoidance system that overrides the RL model when obstacles are dangerously close.

The RL model is trained entirely in simulation and transferred (sim-to-real) onto the physical rover's Jetson Orin Nano, where it runs inference at 5 Hz and sends velocity commands to the motors via serial.

---

## Hardware

| Component | Details |
|-----------|---------|
| **Rover** | Waveshare UGV Rover PT — 6-wheel differential drive |
| **Compute** | NVIDIA Jetson Orin Nano (8GB), Ubuntu 22.04 |
| **Lower Controller** | ESP32 — handles motors, encoders, IMU, voltage sensing |
| **Depth Camera** | OAK-D Lite (stereo depth + RGB) |
| **Gimbal** | Pan-tilt servo mount for camera scanning |
| **Connection** | Tailscale VPN (static IP: `100.68.244.40`) |
| **Serial** | UART `/dev/ttyTHS1` at 115200 baud, JSON + newline protocol |
| **Battery** | 12V LiPo, monitored via ESP32 voltage ADC |

### Physical Specs
- **Wheel separation:** 0.175m (center to center)
- **Max speed:** 1.0 m/s (hardware limit), 0.3 m/s (software-limited for safety)
- **Heartbeat:** Rover auto-stops if no command received for ~3 seconds

---

## Software Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    rl_deploy.py                           │
│                                                          │
│  Layer 1: RL Model ──────── PPO policy (SRB-trained)     │
│  Layer 2: Object Detection ─ MobileNet-SSD on RGB        │
│  Layer 3: Gimbal Scanner ─── Pan/tilt environment scan    │
│  Layer 4: Reactive Avoidance ─ Committed turns            │
│  Layer 5: Course Correction ── Heading-based recovery     │
│  Layer 6: Odometry ────────── Encoder-based tracking      │
│  Layer 7: Destination ──────── Stop at goal distance      │
│  Layer 8: Reward Logger ───── Per-step reward CSV logging │
│                                                          │
│  Safety: EMERGENCY → BACKUP → AVOID → SLOW → RL Model   │
└──────────────────┬───────────────────────────────────────┘
                   │ JSON over UART
                   ▼
┌──────────────────────────────────────────────────────────┐
│  ESP32 Lower Controller                                  │
│  Motors / Encoders / IMU / Voltage                       │
└──────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Jetson-Rover/
├── rl_deploy.py            # Main RL autonomous navigation (1150+ lines)
├── test_rl_deploy.py       # 103 unit tests for rl_deploy.py (runs without hardware)
├── rover_patterns.py       # Movement pattern demos (circle, square, star, etc.)
├── README.md               # This file
│
├── ugv_jetson/             # Waveshare upper-computer program (Jetson)
│   ├── base_ctrl.py        # BaseController — serial JSON interface to ESP32
│   ├── cv_ctrl.py          # OpenCV vision functions (detection, tracking)
│   ├── app.py              # Flask + SocketIO web UI server (port 5000)
│   ├── config.yaml         # Robot configuration (type, speeds, command codes)
│   ├── audio_ctrl.py       # Audio playback
│   ├── os_info.py          # System info display
│   ├── requirements.txt    # Python dependencies
│   ├── models/             # Pre-trained cascade classifiers (face detection)
│   ├── media/              # Image/video assets
│   ├── sounds/             # Audio files
│   └── templates/          # Flask web UI HTML templates
│
├── ugv_ws/                 # ROS 2 workspace
│   └── src/ugv_main/
│       ├── ugv_base_node/  # ESP32 driver node
│       ├── ugv_bringup/    # Launch configs
│       ├── ugv_description/ # URDF robot model
│       ├── ugv_gazebo/     # Gazebo simulation
│       ├── ugv_nav/        # Navigation params
│       ├── ugv_slam/       # SLAM configs (gmapping, cartographer, rtabmap)
│       ├── ugv_vision/     # Camera driver nodes
│       └── ugv_interface/  # ROS service/action definitions
│
└── jetson/                 # Local Python packages
```

---

## Setup & Installation

### 1. Jetson Orin Nano Setup

**Prerequisites:** Ubuntu 22.04 with JetPack, Python 3.10+

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/jetson-rover.git
cd jetson-rover

# Install Python dependencies
pip3 install pyserial flask flask-socketio opencv-python imutils numpy

# Install RL dependencies
pip3 install stable-baselines3 onnxruntime

# Install OAK-D Lite camera driver (depthai v3)
pip3 install depthai

# Verify serial port access
ls -la /dev/ttyTHS1
# If permission denied:
sudo usermod -aG dialout $USER
# Then reboot
```

**Trained model:** Place the trained SRB model at `~/srb-waypoint_navigation_visual.zip`

**Symlink setup for rl_deploy.py** (if running from `~/rl_deploy/`):
```bash
mkdir -p ~/rl_deploy
cp rl_deploy.py test_rl_deploy.py ~/rl_deploy/
ln -sfn ~/ugv_jetson ~/rl_deploy/ugv_jetson
```

### 2. Workstation Setup (SRB Training)

**Prerequisites:** NVIDIA GPU with CUDA, Isaac Sim, Python 3.10+

```bash
# Install Space Robotics Bench
# Follow: https://github.com/NVlabs/space_robotics_bench

# Install training dependencies
pip3 install stable-baselines3 rl-games

# Train the visual waypoint navigation policy
python3 train.py task=waypoint_navigation_visual \
    env.camera_data_types=[depth] \
    env.camera_resolution=[128,128] \
    env.camera_record=true \
    num_envs=64
```

**Training details:**
- **Algorithm:** PPO (via RL-Games)
- **Timesteps:** ~120 million until convergence
- **Observation space:** `Dict('state': Box(4,), 'image_front': Box(-10, 10, (128, 128, 1)))`
  - `state`: 4-element vector encoding relative waypoint direction
  - `image_front`: 128x128 depth map in raw meters (0 = close, 10 = far)
- **Action space:** `Box(-1, 1, (2,))` → `[linear_velocity, angular_velocity]`
- **Output model:** ~155MB `.zip` file

---

## Running the Rover

### Connecting to the Rover

```bash
# SSH via Tailscale VPN
ssh jetson@100.68.244.40
# Password: jetson

# Or via local network (find IP with router/nmap)
ssh jetson@<local-ip>
```

**Tailscale setup:**
```bash
# On Jetson (one-time)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# On your workstation
tailscale up
# Now you can reach the Jetson at 100.68.244.40
```

### RL Autonomous Navigation (`rl_deploy.py`)

This is the main deployment script. It loads the trained RL model, captures depth frames, runs inference, and sends velocity commands to the rover.

```bash
# Navigate to the deployment directory
cd ~/rl_deploy

# Full autonomous run — 5 meters, 5 min timeout
sudo -E python3 rl_deploy.py --max-speed 0.3 --distance 5 --duration 300

# Slower, more conservative
sudo -E python3 rl_deploy.py --max-speed 0.15 --distance 3 --duration 120

# Dry run (prints commands, doesn't move)
sudo -E python3 rl_deploy.py --dry-run --distance 5

# Demo mode (no camera, no hardware — for testing logic)
python3 rl_deploy.py --demo --dry-run

# Minimal mode (no object detection, no gimbal scanning)
sudo -E python3 rl_deploy.py --no-detect --no-gimbal --distance 5

# Custom model path
sudo -E python3 rl_deploy.py --model /path/to/model.zip --distance 5
```

**Why `sudo -E`?** The serial port `/dev/ttyTHS1` requires root access. `-E` preserves your environment variables (needed for Python path).

**CLI Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `~/srb-waypoint_navigation_visual.zip` | Path to trained PPO model |
| `--max-speed` | `0.3` | Max linear speed in m/s |
| `--distance` | None | Target distance in meters (stop when reached) |
| `--duration` | None | Max runtime in seconds |
| `--port` | `/dev/ttyTHS1` | Serial port |
| `--demo` | off | Use dummy camera (no real hardware) |
| `--dry-run` | off | Print commands instead of sending |
| `--no-detect` | off | Disable MobileNet-SSD object detection |
| `--no-gimbal` | off | Disable gimbal pan/tilt scanning |

**Emergency stop:** Press `Ctrl+C` at any time. The rover stops immediately.

### Pattern Demonstrations (`rover_patterns.py`)

Pre-programmed movement patterns for demos and calibration:

```bash
cd ~/rl_deploy

# Interactive menu — choose a pattern
sudo -E python3 rover_patterns.py

# Run a specific pattern
sudo -E python3 rover_patterns.py circle
sudo -E python3 rover_patterns.py square
sudo -E python3 rover_patterns.py star
sudo -E python3 rover_patterns.py zigzag
sudo -E python3 rover_patterns.py figure_eight
sudo -E python3 rover_patterns.py spiral
sudo -E python3 rover_patterns.py triangle
sudo -E python3 rover_patterns.py back_and_forth
sudo -E python3 rover_patterns.py gwagon_turn

# Preview without moving
sudo -E python3 rover_patterns.py --dry-run circle

# Adjust speed (default: 0.15 m/s)
sudo -E python3 rover_patterns.py --speed 0.2 circle

# List all available patterns
python3 rover_patterns.py --list
```

### Web UI Control

The Waveshare web interface provides manual joystick control, camera feed, and system info:

```bash
cd ~/ugv_jetson
python3 app.py
# Access at http://100.68.244.40:5000
```

### ROS 2 Control

```bash
# Build the workspace (first time)
cd ~/ugv_ws
colcon build --symlink-install

# Source the workspace
source install/setup.bash

# Launch the driver node
ros2 launch ugv_bringup ugv_driver_launch.py

# Send velocity commands
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2}, angular: {z: 0.0}}"

# Keyboard teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

## RL Deployment — Detailed Reference

### Navigation Architecture (7 Layers)

The controller uses a layered architecture where higher layers override lower ones:

| Priority | Layer | Role |
|----------|-------|------|
| 1 (lowest) | **RL Model** | Forward + steering intent toward waypoint |
| 2 | **Object Detection** | MobileNet-SSD identifies obstacle class + position |
| 3 | **Gimbal Scanner** | Periodic pan/tilt to map side obstacles |
| 4 | **Reactive Avoidance** | Committed turns to go around obstacles |
| 5 | **Course Correction** | Steer back toward original heading after avoidance |
| 6 | **Odometry** | Track distance, heading, compare commanded vs actual velocity |
| 7 (highest) | **Destination** | Stop when target distance is reached |

Safety tiers (EMERGENCY → BACKUP → AVOID → SLOW) override everything.

### Three-Tier Safety System

Depth-based safety tiers that override the RL model when obstacles are close:

```
Distance from obstacle:
  ≤ 0.05m  →  EMERGENCY: Hard stop → 1s backup → committed avoidance turn
  ≤ 0.15m  →  BACKUP:    Reverse + turn to create clearance
  ≤ 0.10m  →  AVOID:     Committed 3s turn, auto-extends if path blocked (max 8s)
  ≤ 0.20m  →  SLOW:      Reduced speed, RL model still steering
  > 0.20m  →  RL MODEL:  Full autonomous control
```

The EMERGENCY tier handles the OAK-D Lite stereo dead zone: when objects are within ~20cm, the stereo camera returns 0mm depth. We detect this via the zero-pixel fraction (>85% = something blocking the camera).

### Configuration Parameters

All tunable constants are at the top of `rl_deploy.py`:

**Safety & Speed:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_LINEAR_SPEED` | 0.3 m/s | Software speed cap |
| `MAX_ANGULAR_SPEED` | 0.5 rad/s | Max turn rate |
| `EMERGENCY_STOP_DIST` | 0.05m | Hard stop + backup threshold |
| `BACKUP_DIST` | 0.15m | Active reverse threshold |
| `CONTROL_HZ` | 5 Hz | Inference + command loop rate |

**Obstacle Avoidance:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `OBSTACLE_STOP_DIST` | 0.10m | Full stop + committed turn |
| `OBSTACLE_SLOW_DIST` | 0.20m | Reduce speed + bias steering |
| `OBSTACLE_TURN_SPEED` | 0.45 rad/s | Turn rate during avoidance |
| `COMMITTED_TURN_STEPS` | 15 (3.0s) | Duration of committed turn |
| `MAX_AVOID_STEPS` | 40 (8.0s) | Absolute avoidance cap — prevents infinite spinning |

**Backup Maneuver:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `BACKUP_SPEED_FRAC` | 0.5 | Fraction of max_speed for reverse |
| `BACKUP_TURN_SPEED` | 0.4 rad/s | Turn rate while reversing |
| `BACKUP_DURATION` | 1.0s | How long to reverse before re-assessing |

**Course Correction:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `COURSE_CORRECT_GAIN` | 0.6 | Proportional gain (rad/s per radian error) |
| `COURSE_CORRECT_MAX` | 0.3 rad/s | Maximum angular correction |
| `COURSE_CORRECT_DEADBAND` | 5° | Ignore heading error smaller than this |

**Obstacle Memory:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `OBSTACLE_MEMORY_DURATION` | 15.0s | How long to remember obstacle positions |
| `OBSTACLE_MEMORY_RADIUS` | 0.5m | Proximity threshold for "near" |
| `OBSTACLE_PASSED_BEHIND` | 0.3m | Distance behind rover = obstacle passed |

**Gimbal Scanning:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `GIMBAL_SCAN_ANGLE` | 45° | Pan range left/right from center |
| `GIMBAL_SCAN_SPEED` | 100 | Servo speed (0=fastest) |
| `GIMBAL_SCAN_INTERVAL` | 10 steps (2s) | Time between scan moves |
| `GIMBAL_TILT_DEFAULT` | 0° | Camera tilt (0=level, negative=down) |
| `GIMBAL_SETTLE_STEPS` | 2 | Wait after centering before depth capture |

**Object Detection:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `DETECT_CONFIDENCE` | 0.3 | Minimum detection confidence |
| `DETECT_OBSTACLE_CLASSES` | dict | Class → danger weight mapping |

**Odometry:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `WHEEL_SEPARATION` | 0.175m | Distance between wheel centers |
| `ENCODER_SCALE` | 0.01 | Meters per encoder tick |

**Destination:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `DESTINATION_REACHED_TOL` | 0.10m | "Close enough" threshold |
| `DESTINATION_OFF_COURSE_DEG` | 45° | Heading drift warning threshold |

### Depth Camera Processing

The OAK-D Lite stereo camera has a critical quirk: objects closer than ~20cm fall within the stereo dead zone, returning 0mm depth.

**Processing pipeline:**
1. Capture raw depth frame from stereo node (640x400)
2. Resize to 128x128 (RL model input size)
3. Convert mm → meters, clip to [0, 10]
4. Extract center strip (20%–80% height) for obstacle detection
5. Count zero/near-zero pixels (`< 0.02m`)
6. If >85% zeros → `OBSTACLE_VERY_CLOSE` (0.08m) — something is blocking the camera
7. Otherwise, compute 5th percentile depth as `min_depth`
8. Split left/right halves for directional avoidance
9. Fill zero pixels with `min_depth` before passing to RL model

### Object Detection

MobileNet-SSD (Caffe) runs on the RGB camera feed every 3 control frames:

**Detected obstacle classes and danger weights:**
```
person: 1.0    chair: 0.8    sofa: 0.8     diningtable: 0.7
dog: 0.9       cat: 0.9      car: 1.0      bicycle: 0.8
motorbike: 0.9 bottle: 0.5   pottedplant: 0.5  tvmonitor: 0.6
```

When a high-danger object is detected, avoidance thresholds expand:
- `stop_dist += danger * 0.15`
- `slow_dist += danger * 0.20`

The detector also informs avoidance direction: if the object is off-center, turn away from it.

### Gimbal Scanning

The camera gimbal periodically pans left and right to scan for side obstacles:

```
Cycle: CENTER → SCAN RIGHT (45°) → CENTER → SCAN LEFT (45°) → repeat
```

**Critical rule:** The gimbal MUST be centered when capturing depth for RL inference. The RL model was trained on forward-facing depth only. Scanning happens between depth captures, with `GIMBAL_SETTLE_STEPS` (2 frames) of settling time after centering.

During avoidance, the gimbal auto-centers and stays centered until the maneuver completes.

### Course Correction & Obstacle Memory

**Course Correction:** After completing an avoidance maneuver, the rover uses proportional heading control to steer back toward its original heading:
- Error = target_heading - current_heading (normalized to [-π, π])
- Correction = gain × error, clamped to ±0.3 rad/s
- Deadband: ignore error < 5° to avoid oscillation

**Obstacle Memory:** Obstacles are recorded in world-frame (x, y) coordinates:
- Position computed from rover odometry + depth reading direction
- Positions remembered for 15 seconds, then pruned
- Duplicate positions within 20cm are merged
- Used to inform steering when left/right depths are equal
- `all_passed()` checks if all remembered obstacles are behind the rover

### Odometry & Destination Tracking

**Odometry** reads ESP32 encoder feedback (`odl`, `odr` fields) and computes:
- Path length (total distance traveled)
- Heading (accumulated angular displacement)
- Position (x, y) in the start frame
- Actual linear and angular velocity (for velocity error tracking)
- Displacement (straight-line distance from start)

**Destination tracking** uses displacement (not path length):
- Target: reach N meters of straight-line displacement from start
- `displacement = sqrt(x² + y²)` — unaffected by avoidance detours
- Automatically stops when `remaining ≤ 0.10m`
- Logs "off course" warnings when heading drift > 45°

### Face Navigation Mode

Navigate toward a detected person's face using `--target face`:

```bash
sudo -E python3 rl_deploy.py --target face --face-distance 1.5 --duration 120
```

**How it works:** The face detector finds a face (Haar cascade primary, MobileNet-SSD 'person' fallback), computes the angle and distance to it, and encodes that as the RL model's state vector. The RL model then steers toward the face naturally — the same way it steers toward waypoints in simulation.

**State machine:**

| State | Behavior | RL Model Role |
|-------|----------|---------------|
| **SEARCH** | Rotate in place (±0.25 rad/s, reversing every 3s) to find a face | Overridden — pure rotation |
| **NAVIGATE** | Face detected. State vector = `[cos(angle), sin(angle), 0, dist/10]` | **Full control** — steers toward face via state vector |
| **LOST** | Face disappeared. Hold last state for 1s, then re-search | Drives with last known state, then overridden |
| **ARRIVED** | Face distance < threshold (default 1.0m) | Stopped |

**Key design:** During NAVIGATE, the RL model controls ALL steering and obstacle avoidance. The face tracker only provides the direction — the model does the driving. During SRB training, the model learned to steer toward waypoints at various angles, so encoding the face as a "waypoint" works naturally.

**Face detection details:**
- Primary: Haar cascade (`haarcascade_frontalface_default.xml`) — fast, good for frontal faces
- Fallback: MobileNet-SSD 'person' class — more robust to angles/lighting, less precise
- Angle: Computed from face center position × OAK-D Lite HFOV (73°)
- Distance: Depth sampled from a 7×7 patch at the face center in the depth frame
- RGB (640×480) to depth (128×128) coordinate mapping for depth lookup

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FACE_HFOV_RAD` | 73° | OAK-D Lite horizontal FOV |
| `FACE_STOP_DISTANCE` | 1.0m | Stop when this close to face |
| `FACE_RECHECK_STEPS` | 10 (2s) | Re-detect face interval during NAVIGATE |
| `FACE_SEARCH_SPEED` | 0.25 rad/s | Rotation speed during search |
| `FACE_SEARCH_TIMEOUT` | 60s | Give up searching after this |
| `FACE_LOST_HOLD_TIME` | 1.0s | Hold last state before re-searching |

### Terrain Segmentation (Mark's UNet)

Integrates a 4-class lunar terrain segmentation model to improve avoidance decisions:

```bash
sudo -E python3 rl_deploy.py --distance 5 --segmentation
sudo -E python3 rl_deploy.py --target face --segmentation --duration 120
```

**Model:** UNet with MobileNet-v2 encoder, trained on the Artificial Lunar Rocky Landscape Dataset.
- **Classes:** 0=ground, 1=sky, 2=small rock, 3=big rock
- **Input:** RGB image resized to 720×480
- **Output:** Per-pixel class mask → feasibility score [0-1]
- **Model file:** `mark model/unet_lunar_segmentation.pth` (26MB)

**Integration with avoidance:**
- Runs every 2 seconds (10 control steps)
- Computes separate feasibility scores for left and right halves of the frame
- When the avoidance system needs to choose a turn direction and left/right depths are similar, it prefers the side with higher terrain feasibility (fewer rocks)
- Feasibility score: rock density → inverse sigmoid (1.0 = clear ground, 0.0 = very rocky)

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEG_RUN_INTERVAL` | 10 steps (2s) | How often to run segmentation |
| `SEG_ROCK_PENALTY` | 50 | Penalty scale for rock density |
| `SEG_SIGMOID_WIDTH` | 0.5 | Sigmoid sharpness |

---

### Reward Logger

Optional per-step reward logging records what shaped rewards *would be* assigned at each control step — useful for post-hoc analysis and future online learning research. **No model weights are updated.**

```bash
sudo -E python3 rl_deploy.py --distance 5 --log-rewards
sudo -E python3 rl_deploy.py --target face --log-rewards --reward-log my_run.csv
```

**Output:** CSV file with columns: `step, timestamp, reward, event, linear_vel, angular_vel, min_depth, left_depth, right_depth, mode, face_state, seg_left, seg_right, odom_dist, odom_disp, heading_deg`

**Reward shaping:**

| Event | Reward | Trigger |
|-------|--------|---------|
| `GOAL` | +10.0 | Reached face target or distance goal |
| `FWD` | +0.2 × (v/v_max) | Forward progress (scaled by speed) |
| `EMERGENCY` | -2.0 | Emergency stop (≤5cm) |
| `BACKUP` | -1.0 | Active backup maneuver |
| `CLOSE` | -0.5 | In STOP/AVOID zone |
| `AVOID` | -0.3 | Avoidance turning in place |
| `IDLE` | -0.1 | Near-zero velocity |

---

## Full Pipeline Explanation

This section describes the complete flow from perception to motor command, explaining how each component contributes.

### The Big Picture

The rover's autonomy stack operates at **5 Hz** (200ms per cycle). Each cycle:

```
Depth Camera (128×128) ──→ RL Model (PPO) ──→ [linear, angular] velocity
       │                        ↑
       │                   State Vector
       │                   [cos θ, sin θ, 0, dist]
       │                        ↑
       │                  Face Detector (optional)
       │                  or Default [1,0,0,1] = "straight ahead"
       │
       ├──→ Safety System (depth thresholds) ──→ Can OVERRIDE RL output
       │
       └──→ Terrain Segmenter (optional) ──→ Informs avoidance DIRECTION
```

### Step-by-Step: One Control Cycle

1. **Gimbal Check** — If the gimbal is still moving from a scan, skip this frame (depth would be angled, not forward-facing).

2. **Depth Capture** — The OAK-D Lite provides a 128×128 depth map in meters. Zero pixels (stereo dead zone) are filled with 0.05m (assume very close). The frame also yields `min_depth`, `left_depth`, and `right_depth` aggregates.

3. **Safety Tiers** — Hard-coded depth thresholds that can override everything:
   - **EMERGENCY** (≤5cm): Full stop → backup → committed turn
   - **BACKUP** (≤15cm): Reverse while turning
   - **STOP** (≤10cm): Stop forward motion, committed turn
   - **SLOW** (≤20cm): Reduce speed, bias steering away

4. **Face Detection** (if `--target face`):
   - Haar cascade + MobileNet-SSD fallback detects the largest face in RGB
   - Computes angle from frame center using the camera's 73° HFOV
   - Samples depth at the face location (7×7 median patch)
   - Encodes as state vector: `[cos(angle), sin(angle), 0, distance/10]`
   - This replaces the default `[1, 0, 0, 1]` ("waypoint straight ahead")
   - **Key insight:** The RL model was trained with waypoints at various angles/distances. By encoding the face as a "waypoint," the model naturally steers toward it.

5. **Terrain Segmentation** (if `--segmentation`):
   - Mark's UNet segments the RGB frame into 4 classes: ground, sky, small rock, big rock
   - Computes feasibility scores for left and right halves (rock density → sigmoid)
   - When the avoidance system can't decide left vs. right by depth alone, it picks the side with fewer rocks

6. **RL Inference** — The PPO model receives `{state: [4], image_front: [128,128,1]}` and outputs `[linear, angular]` in [-1, 1]. These are scaled to m/s and rad/s.

7. **Override Priority** — The final command is chosen by priority:
   - Safety override (EMERGENCY/BACKUP) > Face search rotation > Avoidance turn > Slow zone scaling > RL model output

8. **Motor Command** — The velocity is sent to the ESP32 as `{"T":13, "X": linear, "Z": angular}` over UART.

### How Mark's Model Helps

The RL model was trained in simulation with perfect depth data and no real-world rock textures. In the real world:

- **Problem:** Two paths might have similar depth readings, but one is rocky terrain and the other is clear ground. The depth camera can't tell the difference.
- **Solution:** Mark's UNet, trained on lunar rock imagery, classifies each pixel as ground, sky, small rock, or big rock. When the avoidance system faces an ambiguous left-vs-right choice, it checks which side has fewer rocks.
- **Integration:** The segmentation doesn't replace the RL model — it informs the *direction* of avoidance turns. The RL model still handles forward navigation and general steering.

### What Our RL Model Does

The PPO policy was trained in Space Robotics Bench (SRB) on NVIDIA Isaac Sim:

- **Observation:** 128×128 depth map + 4D state vector (direction/distance to waypoint)
- **Action:** Continuous [linear_velocity, angular_velocity] in [-1, 1]
- **Training:** ~120M timesteps across 64 parallel lunar environments with rocks, craters, and slopes
- **Reward:** Progress toward waypoint (positive), collisions (negative), goal bonus
- **Sim-to-Real:** Trained entirely in sim, deployed zero-shot on the physical rover. The depth camera provides a similar observation to what the model saw during training.

The model learned to:
- Navigate toward waypoints at arbitrary angles
- Avoid obstacles using depth perception
- Recover from near-collisions
- Handle uneven terrain and slopes

Our safety layers (reactive avoidance, backup, emergency stop) provide a hard floor — they override the RL model when obstacles are dangerously close, ensuring the rover never crashes even if the RL model makes a mistake.

---

## Testing

103 unit tests covering all subsystems. Tests run without hardware, camera, or trained model:

```bash
cd ~/rl_deploy   # or the repo root

# Run all tests
python3 -m pytest test_rl_deploy.py -v

# Run a specific test class
python3 -m pytest test_rl_deploy.py::TestDepthNormalization -v
python3 -m pytest test_rl_deploy.py::TestReactiveAvoidance -v
python3 -m pytest test_rl_deploy.py::TestSafetyTiers -v

# Run with output
python3 -m pytest test_rl_deploy.py -v -s
```

**Test classes:**

| Class | Tests | What it covers |
|-------|-------|---------------|
| `TestDepthNormalization` | 4 | Depth frame format, raw meter values, clipping |
| `TestReactiveAvoidance` | 7 | Committed turns, turn direction, extension |
| `TestCommittedTurn` | 5 | Turn step counting, clearance exit |
| `TestCourseCorrection` | 5 | Proportional heading control, deadband, clamping |
| `TestObstacleMemory` | 9 | World-frame positions, pruning, nearest-in-front |
| `TestSpeedClamping` | 4 | Velocity limits, negative speed capping |
| `TestDummyDepthCamera` | 6 | Fake camera data ranges and format |
| `TestActionScaling` | 4 | RL action → velocity mapping |
| `TestSafetyTiers` | 4 | EMERGENCY/BACKUP/AVOID/SLOW thresholds |
| `TestObjectDetection` | 4 | Danger scoring, center weighting |
| `TestGimbalScanner` | 6 | Scan cycle, centering, settle steps |
| `TestOdometryTracker` | 9 | Encoder integration, heading, displacement |
| `TestDestinationTracker` | 5 | Displacement-based goal tracking |
| `TestIntegration` | 2 | End-to-end mocked control loop |
| `TestFaceTracker` | 6 | Angle computation, depth mapping, state encoding |
| `TestFaceNavigationSM` | 8 | State machine transitions, search/navigate/lost/arrived |
| `TestTerrainSegmenter` | 4 | Feasibility scoring, rock density, left/right split |
| `TestSegmentationAvoidance` | 3 | Terrain-aware turn preference, depth priority |
| `TestRewardLogger` | 8 | Reward shaping, CSV output, dummy no-op |

---

## Serial Command Reference

Communication with the ESP32 uses JSON over UART (`/dev/ttyTHS1`, 115200 baud). Each command is a JSON object followed by a newline.

| T Value | Command | Example |
|---------|---------|---------|
| 1 | Motor speed (L/R in m/s, range -1 to 1) | `{"T":1,"L":0.2,"R":0.2}` |
| 11 | PWM control (L/R, range -255 to 255) | `{"T":11,"L":128,"R":128}` |
| 13 | Velocity control (X=linear m/s, Z=angular rad/s) | `{"T":13,"X":0.2,"Z":0.0}` |
| 132 | LED lights (IO4/IO5 PWM 0-255) | `{"T":132,"IO4":255,"IO5":255}` |
| 133 | Gimbal control (X=pan, Y=tilt, SPD, ACC) | `{"T":133,"X":0,"Y":0,"SPD":0,"ACC":0}` |
| 137 | Gimbal steady mode | `{"T":137}` |
| 900 | Chassis config (main_type, module_type) | `{"T":900,"main":2,"module":2}` |

**BaseController API** (`ugv_jetson/base_ctrl.py`):
```python
from base_ctrl import BaseController
base = BaseController('/dev/ttyTHS1', 115200)

base.base_json_ctrl({"T":13, "X":0.2, "Z":0.0})  # Send any JSON command
base.base_speed_ctrl(0.2, 0.2)                     # Left/right motor speeds (m/s)
base.gimbal_ctrl(pan, tilt, speed, accel)           # Pan-tilt gimbal
base.lights_ctrl(pwmA, pwmB)                        # LED brightness (0-255)
base.base_oled(line, text)                           # Write to OLED display
data = base.feedback_data()                          # Read encoder/IMU/voltage
# Returns: {'odl': int, 'odr': int, 'ax': float, 'ay': float, ...}
```

---

## Simulation — Space Robotics Bench (SRB)

### Training Environment

- **Platform:** NVIDIA Isaac Sim via Space Robotics Bench (SRB)
- **Task:** `waypoint_navigation_visual` — navigate to sequential waypoints on lunar terrain
- **Terrain:** Simulated lunar surface with rocks, craters, slopes, and uneven ground
- **Algorithm:** PPO via RL-Games library
- **Training:** ~120 million timesteps, ~155MB final model

### Observation Space

```python
Dict({
    'state': Box(shape=(4,), low=-inf, high=inf),
    # 4-element vector encoding relative direction to next waypoint
    # DEFAULT_STATE = [1.0, 0.0, 0.0, 1.0] → waypoint straight ahead

    'image_front': Box(shape=(128, 128, 1), low=-10, high=10),
    # Depth map in RAW METERS (NOT normalized)
    # 0 = very close, 10 = far away
})
```

### Action Space

```python
Box(shape=(2,), low=-1.0, high=1.0)
# action[0] = linear velocity  (×MAX_LINEAR_SPEED → m/s)
# action[1] = angular velocity (×MAX_ANGULAR_SPEED → rad/s)
```

### Training Command

```bash
python3 train.py task=waypoint_navigation_visual \
    env.camera_data_types=[depth] \
    env.camera_resolution=[128,128] \
    num_envs=64
```

**Note:** LiDAR has geometry bugs in the current SRB version — use depth camera only.

### Deploying a Trained Model

```bash
# On your workstation — copy model to Jetson
scp srb-waypoint_navigation_visual.zip jetson@100.68.244.40:~/

# On the Jetson — run it
cd ~/rl_deploy
sudo -E python3 rl_deploy.py --distance 5
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `No module named 'base_ctrl'` | `ugv_jetson/` not in path | `ln -sfn ~/ugv_jetson ~/rl_deploy/ugv_jetson` |
| `module 'depthai' has no attribute 'CameraProperties'` | depthai v3 API change | Code already uses `ColorCameraProperties` and tuple `sensorResolution=(1920, 1080)` |
| `Permission denied: /dev/ttyTHS1` | Need root for serial | Use `sudo -E python3 ...` |
| `unrecognized arguments: --distance` | Running old version from wrong directory | `cd ~/rl_deploy` then run |
| Rover spins in circles, never moves forward | Gimbal tilted down → reading floor as obstacle | Set `GIMBAL_TILT_DEFAULT = 0` (already fixed) |
| Depth reads 999m when obstacle is right in front | Stereo dead zone (<20cm) | Fixed: 85% zero-pixel detection → `OBSTACLE_VERY_CLOSE` |
| Rover freezes at safety stop | Old code did hard stop only | Fixed: now backs up + committed avoidance turn |
| Rover too cautious, barely moves | Avoidance thresholds too high | Lower `OBSTACLE_STOP_DIST` / `OBSTACLE_SLOW_DIST` |
| Avoidance never exits (infinite spinning) | Auto-extend with no cap + floor readings | Fixed: `MAX_AVOID_STEPS = 40` (8s cap) |
| Rover drifts after avoidance | No course correction | Fixed: heading-based proportional control |
| MobileNet model not found | Missing model files | Check `ugv_jetson/models/deploy.prototxt` and `mobilenet_iter_73000.caffemodel` |
| SCP timeout | Jetson unreachable | Check Tailscale: `tailscale status`, ping `100.68.244.40` |

### Diagnostic Commands

```bash
# Check Jetson connection
ping 100.68.244.40
tailscale status

# Check serial port
ls -la /dev/ttyTHS*
# Should show /dev/ttyTHS1

# Check depthai version
python3 -c "import depthai; print(depthai.__version__)"

# Check camera is detected
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"

# Test serial communication manually
python3 -c "
from ugv_jetson.base_ctrl import BaseController
b = BaseController('/dev/ttyTHS1', 115200)
print(b.feedback_data())
"

# Check GPU/system
jetson_stats  # or jtop
nvidia-smi
```

---

## Future Plans

### Expo Demo (April 2026)

1. **Live Face Navigation Demo** — The rover navigates autonomously toward a person's face using RL + face detection, avoiding obstacles in the path. The face target is encoded as the RL model's waypoint state vector.

2. **Terrain Segmentation Showcase** — Demonstrate how Mark's UNet model helps the rover choose clearer paths when multiple routes have similar depth readings.

3. **Reward Analysis Poster Data** — Use the reward logger CSV data from demo runs to create charts showing reward distribution, safety events, and navigation efficiency for the research poster.

### Short-Term Improvements

4. **Online RL Fine-Tuning** — Investigate adding real-world reward signals (negative reward for backup/emergency, positive for forward progress) to fine-tune the policy on the Jetson. Current blocker: PPO requires large rollout buffers (~2048 steps) which is slow at 5Hz; SAC would be more suitable for online learning but requires retraining.

5. **Segmentation as RL Input** — Train a new policy that takes terrain segmentation as an additional observation channel (depth + segmentation mask), allowing the RL model to directly learn terrain preferences rather than using segmentation as a heuristic.

6. **Multi-Waypoint Navigation** — Extend from single-destination to sequential waypoint navigation for longer autonomous missions.

### Long-Term (Future Work)

7. **GPS/RTK Integration** — Add outdoor localization for campus-scale navigation.
8. **Dynamic Obstacle Tracking** — Predict obstacle motion (e.g., walking people) for anticipatory avoidance.
9. **Multi-Rover Coordination** — Extend to collaborative navigation between multiple rovers.

---

## License

This project was developed as part of the CMPS 4010/4020 Senior Capstone at Tulane University. Contact the team for licensing inquiries.

---

## Acknowledgments

- **Dr. Jihun Hamm** — Faculty advisor
- **Waveshare** — UGV Rover PT hardware and `ugv_jetson`/`ugv_ws` base software
- **NVIDIA** — Space Robotics Bench, Isaac Sim, Jetson platform
- **Luxonis** — OAK-D Lite depth camera and depthai SDK
- **Stable-Baselines3** — PPO implementation for RL training
