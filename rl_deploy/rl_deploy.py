#!/usr/bin/env python3
"""
RL Deployment Bridge - Team Crater
====================================
Deploys a trained SRB visual navigation policy onto the Waveshare UGV Rover PT.

The model was trained in Space Robotics Bench (SRB) with:
  - Observation: Dict('state': (4,), 'image_front': (128,128,1)) depth map
  - Action: (2,) [linear_velocity, angular_velocity] in [-1, 1]

Navigation architecture:
  Layer 1 (RL Model):       Forward+steering intent toward waypoint
  Layer 2 (Object Detect):  MobileNet-SSD identifies obstacles (person, chair, etc.)
  Layer 3 (Gimbal Scanner): Periodic pan/tilt to map the environment
  Layer 4 (Reactive Avoid): Committed turns to go AROUND obstacles
  Layer 5 (Recovery):       Steer back toward heading after avoiding
  Layer 6 (Odometry):       Track distance, heading, compare cmd vs actual velocity
  Layer 7 (Destination):    Stop when target distance is reached

IMPORTANT: The RL model was trained on FORWARD-FACING depth. The gimbal MUST
be centered when we capture the depth frame for inference. Scanning happens
between captures.

  Layer 8 (Reward Log):    Optional per-step reward CSV for post-hoc analysis

Usage:
    python3 rl_deploy.py                              # Full deployment
    python3 rl_deploy.py --demo --dry-run             # Test without hardware
    python3 rl_deploy.py --max-speed 0.15 --duration 60
    python3 rl_deploy.py --distance 5                 # Go 5 meters then stop
    python3 rl_deploy.py --no-detect --no-gimbal      # Minimal mode
    python3 rl_deploy.py --distance 5 --log-rewards   # Log shaped rewards to CSV

SAFETY: Press Ctrl+C at any time for emergency stop.
"""

import sys
import os
import time
import signal
import argparse
import math
import csv
import numpy as np

# Add ugv_jetson to path for base_ctrl and cv_ctrl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ugv_jetson'))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 115200
DEFAULT_MODEL_PATH = os.path.expanduser('~/srb-waypoint_navigation_visual.zip')

# Safety limits
MAX_LINEAR_SPEED = 0.3    # m/s
MAX_ANGULAR_SPEED = 0.5   # rad/s
EMERGENCY_STOP_DIST = 0.05   # meters - HARD STOP, then back up (touching/stereo dead zone)
BACKUP_DIST = 0.15           # meters - active backup + turn
CONTROL_HZ = 5               # inference + command rate

# Reactive obstacle avoidance
OBSTACLE_STOP_DIST = 0.10    # meters - full stop + committed turn
OBSTACLE_SLOW_DIST = 0.20    # meters - reduce speed + bias steering
OBSTACLE_TURN_SPEED = 0.45   # rad/s - how fast to turn during avoidance
COMMITTED_TURN_STEPS = 15    # At 5 Hz = 3.0s committed — must be long enough to clear
MAX_AVOID_STEPS = 40         # At 5 Hz = 8s absolute max — prevents infinite spinning

# Backup maneuver
BACKUP_SPEED_FRAC = 0.5      # fraction of max_speed to reverse
BACKUP_TURN_SPEED = 0.4      # rad/s while backing up
BACKUP_DURATION = 1.0        # seconds to back up before re-assessing

# Course correction after avoidance
COURSE_CORRECT_GAIN = 0.6    # rad/s per radian of heading error
COURSE_CORRECT_MAX = 0.3     # max angular correction rad/s
COURSE_CORRECT_DEADBAND = math.radians(5)  # ignore error < 5°

# Obstacle memory
OBSTACLE_MEMORY_DURATION = 15.0   # seconds to remember obstacle locations
OBSTACLE_MEMORY_RADIUS = 0.5     # meters - how close before we consider "near" a remembered obstacle
OBSTACLE_PASSED_BEHIND = 0.3     # meters behind rover = obstacle is passed

# Object detection
DETECT_CONFIDENCE = 0.3
DETECT_OBSTACLE_CLASSES = {
    'person': 1.0, 'chair': 0.8, 'sofa': 0.8, 'diningtable': 0.7,
    'dog': 0.9, 'cat': 0.9, 'car': 1.0, 'bicycle': 0.8,
    'motorbike': 0.9, 'bottle': 0.5, 'pottedplant': 0.5, 'tvmonitor': 0.6,
    'bus': 1.0, 'cow': 0.9, 'horse': 0.9, 'sheep': 0.8,
}

# Gimbal scanning
GIMBAL_SCAN_ANGLE = 45        # degrees left/right from center
GIMBAL_SCAN_SPEED = 100       # servo speed (0=fastest, higher=slower)
GIMBAL_SCAN_ACCEL = 50        # servo acceleration
GIMBAL_SCAN_INTERVAL = 10     # control steps between scan moves (2s at 5Hz)
GIMBAL_TILT_DEFAULT = 0       # level — tilt down was reading the FLOOR as an obstacle
GIMBAL_SETTLE_STEPS = 2       # steps to wait after centering before depth capture

# Odometry
WHEEL_SEPARATION = 0.175      # meters between wheel centers
ENCODER_SCALE = 0.01          # each encoder tick = 0.01 meters

# Destination tracking
DESTINATION_REACHED_TOL = 0.10     # meters - "close enough"
DESTINATION_OFF_COURSE_DEG = 45.0  # degrees drift → warning

# Camera settings
MODEL_INPUT_SIZE = (128, 128)

# State vector: "waypoint is straight ahead"
DEFAULT_STATE = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

# Face navigation
FACE_HFOV_RAD = math.radians(73)       # OAK-D Lite horizontal FOV
FACE_DEPTH_PATCH = 7                    # pixels to sample for depth median
FACE_STOP_DISTANCE = 1.0               # meters — stop when this close to face
FACE_RECHECK_STEPS = 10                # re-detect face every N steps (2s at 5Hz)
FACE_SEARCH_SPEED = 0.25               # rad/s rotation during search
FACE_SEARCH_REVERSE_TIME = 3.0         # seconds before reversing search direction
FACE_SEARCH_TIMEOUT = 60.0             # seconds before giving up search
FACE_LOST_HOLD_TIME = 1.0              # seconds to hold last state when face lost

# Terrain segmentation
SEG_RUN_INTERVAL = 10                   # run segmentation every N steps (2s at 5Hz)
SEG_ROCK_PENALTY = 50                   # penalty scale for rock density
SEG_SIGMOID_WIDTH = 0.5                 # sigmoid sharpness for feasibility score


# ---------------------------------------------------------------------------
# Face Tracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """Detects faces via Haar cascade (primary) or MobileNet-SSD person detection (fallback).
    Estimates angle and distance to the largest detected face."""

    def __init__(self, detector=None):
        import cv2
        haar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'ugv_jetson', 'models', 'haarcascade_frontalface_default.xml')
        self.cascade = None
        if os.path.exists(haar_path):
            self.cascade = cv2.CascadeClassifier(haar_path)
            if self.cascade.empty():
                self.cascade = None
                print("[WARN] Haar cascade failed to load")
            else:
                print("[OK] Haar cascade face detector loaded")
        else:
            print(f"[WARN] Haar cascade not found at {haar_path}")

        self.detector = detector  # MobileNet-SSD fallback (ObjectDetector or None)
        self.last_face = None     # (angle_rad, distance_m, timestamp)

    def detect(self, rgb_frame, depth_frame):
        """Detect largest face and estimate polar coordinates.

        Returns:
            (found, angle_rad, distance_m)
            angle_rad: horizontal angle from camera center (+=right, -=left)
            distance_m: estimated distance from depth frame
        """
        import cv2

        if rgb_frame is None:
            return False, 0.0, 0.0

        h_rgb, w_rgb = rgb_frame.shape[:2]
        best_cx, best_cy, best_area = None, None, 0

        # --- Primary: Haar cascade face detection ---
        if self.cascade is not None:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.2,
                                                   minNeighbors=5, minSize=(30, 30))
            for (x, y, w, fh) in faces:
                area = w * fh
                if area > best_area:
                    best_area = area
                    best_cx = x + w / 2.0
                    best_cy = y + fh / 2.0

        # --- Fallback: MobileNet-SSD 'person' detection ---
        if best_cx is None and self.detector is not None:
            detections = self.detector.detect(rgb_frame)
            for det in detections:
                if det['class'] == 'person' and det['confidence'] > 0.4:
                    bbox = det['bbox']  # (xmin, ymin, xmax, ymax) normalized [0,1]
                    bw = (bbox[2] - bbox[0]) * w_rgb
                    bh = (bbox[3] - bbox[1]) * h_rgb
                    area = bw * bh
                    if area > best_area:
                        best_area = area
                        best_cx = (bbox[0] + bbox[2]) / 2.0 * w_rgb
                        best_cy = (bbox[1] + bbox[3]) / 2.0 * h_rgb

        if best_cx is None:
            return False, 0.0, 0.0

        # --- Compute angle from center_x ---
        center_x_norm = best_cx / w_rgb  # [0, 1]
        angle_rad = (center_x_norm - 0.5) * FACE_HFOV_RAD

        # --- Compute distance from depth frame ---
        h_d, w_d = depth_frame.shape[:2]
        cx_d = int(best_cx * w_d / w_rgb)
        cy_d = int(best_cy * h_d / h_rgb)
        cx_d = max(FACE_DEPTH_PATCH // 2, min(w_d - FACE_DEPTH_PATCH // 2 - 1, cx_d))
        cy_d = max(FACE_DEPTH_PATCH // 2, min(h_d - FACE_DEPTH_PATCH // 2 - 1, cy_d))

        half = FACE_DEPTH_PATCH // 2
        patch = depth_frame[cy_d - half:cy_d + half + 1, cx_d - half:cx_d + half + 1]
        valid = patch[patch > 0.02]
        if len(valid) > 0:
            distance_m = float(np.median(valid))
        else:
            distance_m = 5.0  # can't measure depth — assume moderate distance

        self.last_face = (angle_rad, distance_m, time.time())
        return True, angle_rad, distance_m


class DummyFaceTracker:
    """No-op face tracker for testing/demo mode."""
    def __init__(self):
        self.last_face = None

    def detect(self, rgb_frame, depth_frame):
        return False, 0.0, 0.0


# ---------------------------------------------------------------------------
# Face Navigation State Machine
# ---------------------------------------------------------------------------

class FaceNavigationSM:
    """Navigate toward a detected face using the RL model.

    The face detector updates the state vector fed to the RL model.
    The RL model controls ALL steering and avoidance.
    The only override is SEARCH mode (rotate to find a face).

    States: SEARCH → NAVIGATE → LOST → ARRIVED
    """

    def __init__(self, face_tracker, odom, stop_distance=FACE_STOP_DISTANCE):
        self.tracker = face_tracker
        self.odom = odom
        self.stop_distance = stop_distance
        self.state = 'SEARCH'
        self.reached = False
        self._steps_since_recheck = 0
        self._last_state_vector = DEFAULT_STATE.copy()
        # Search state
        self._search_dir = 1
        self._search_flip_time = time.time()
        self._search_start = time.time()
        # Lost state
        self._lost_time = None

    def update(self, rgb_frame, depth_frame):
        """Compute face navigation output.

        Returns:
            (override, state_vector)
            override: (linear, angular) tuple during SEARCH/ARRIVED, or None when RL drives
            state_vector: 4-element numpy array for RL model observation
        """
        found, angle, distance = self.tracker.detect(rgb_frame, depth_frame)
        now = time.time()

        if self.state == 'SEARCH':
            if found:
                self.state = 'NAVIGATE'
                self._steps_since_recheck = 0
                self._last_state_vector = self._encode_state(angle, distance)
                print(f"  [FACE] Found! angle={math.degrees(angle):.1f}° dist={distance:.2f}m → NAVIGATE")
                if distance < self.stop_distance:
                    self.state = 'ARRIVED'
                    self.reached = True
                    return (0.0, 0.0), DEFAULT_STATE.copy()
                return None, self._last_state_vector
            # Rotate to search — reverse direction periodically
            if now - self._search_flip_time > FACE_SEARCH_REVERSE_TIME:
                self._search_dir *= -1
                self._search_flip_time = now
            # Timeout check
            if now - self._search_start > FACE_SEARCH_TIMEOUT:
                print("  [FACE] Search timeout — no face found")
                self.reached = True  # signal to stop
                return (0.0, 0.0), DEFAULT_STATE.copy()
            return (0.0, self._search_dir * FACE_SEARCH_SPEED), DEFAULT_STATE.copy()

        elif self.state == 'NAVIGATE':
            self._steps_since_recheck += 1
            if self._steps_since_recheck >= FACE_RECHECK_STEPS:
                self._steps_since_recheck = 0
                if found:
                    if distance < self.stop_distance:
                        self.state = 'ARRIVED'
                        self.reached = True
                        print(f"  [FACE] ARRIVED! dist={distance:.2f}m")
                        return (0.0, 0.0), DEFAULT_STATE.copy()
                    self._last_state_vector = self._encode_state(angle, distance)
                    print(f"  [FACE] Tracking: angle={math.degrees(angle):.1f}° dist={distance:.2f}m")
                else:
                    self.state = 'LOST'
                    self._lost_time = now
                    print("  [FACE] Lost target → LOST")
            # RL drives with face-encoded state vector
            return None, self._last_state_vector

        elif self.state == 'LOST':
            if found:
                self.state = 'NAVIGATE'
                self._steps_since_recheck = 0
                self._last_state_vector = self._encode_state(angle, distance)
                print(f"  [FACE] Re-acquired! angle={math.degrees(angle):.1f}° dist={distance:.2f}m")
                return None, self._last_state_vector
            if now - self._lost_time > FACE_LOST_HOLD_TIME:
                self.state = 'SEARCH'
                self._search_start = now
                self._search_flip_time = now
                print("  [FACE] Lost too long → SEARCH")
                return (0.0, self._search_dir * FACE_SEARCH_SPEED), DEFAULT_STATE.copy()
            # Hold last state — RL drives toward last known position
            return None, self._last_state_vector

        elif self.state == 'ARRIVED':
            return (0.0, 0.0), DEFAULT_STATE.copy()

        return None, DEFAULT_STATE.copy()

    @staticmethod
    def _encode_state(angle_rad, distance_m):
        """Encode face position as SRB-compatible state vector.
        [cos(angle), sin(angle), 0, normalized_distance]"""
        return np.array([
            math.cos(angle_rad),
            math.sin(angle_rad),
            0.0,
            min(distance_m / 10.0, 1.0),
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Object Detector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """Detects obstacles using MobileNet-SSD (same model as the rover's web UI)."""

    def __init__(self):
        import cv2
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'ugv_jetson', 'models')
        prototxt = os.path.join(models_dir, 'deploy.prototxt')
        caffemodel = os.path.join(models_dir, 'mobilenet_iter_73000.caffemodel')

        if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
            raise FileNotFoundError(f"MobileNet-SSD model not found in {models_dir}")

        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.class_names = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        print("[OK] MobileNet-SSD object detector loaded")

    def detect(self, rgb_frame):
        """Run detection on an RGB frame. Returns list of detection dicts."""
        import cv2
        blob = cv2.dnn.blobFromImage(
            cv2.resize(rgb_frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        raw = self.net.forward()

        results = []
        for i in range(raw.shape[2]):
            conf = float(raw[0, 0, i, 2])
            if conf > DETECT_CONFIDENCE:
                idx = int(raw[0, 0, i, 1])
                if 0 <= idx < len(self.class_names):
                    box = raw[0, 0, i, 3:7]
                    results.append({
                        'class': self.class_names[idx],
                        'confidence': conf,
                        'bbox': tuple(box),
                        'center_x': float((box[0] + box[2]) / 2.0),
                    })
        return results


class DummyDetector:
    """Placeholder when object detection is disabled."""
    def detect(self, rgb_frame):
        return []


# ---------------------------------------------------------------------------
# Terrain Segmenter (Mark's UNet)
# ---------------------------------------------------------------------------

class TerrainSegmenter:
    """Runs Mark's UNet lunar terrain segmentation model.

    Classes: 0=ground, 1=sky, 2=small rock, 3=big rock
    Returns a feasibility score (0-1) for overall, left, and right halves.
    """

    def __init__(self, model_path):
        import torch
        import segmentation_models_pytorch as smp
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=4,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self._last_result = (0.8, 0.8, 0.8)
        print(f"[OK] Terrain segmenter loaded on {self.device}")

    def analyze(self, rgb_frame):
        """Run segmentation on RGB frame.

        Returns (overall_feasibility, left_feasibility, right_feasibility).
        Scores are 0-1 where 1 = clear ground, 0 = very rocky.
        """
        import cv2
        if rgb_frame is None:
            return self._last_result

        # Resize to model's training resolution (480x720)
        resized = cv2.resize(rgb_frame, (720, 480))
        # Normalize to [0,1] float32, NCHW
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        tensor = self.torch.from_numpy(img).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            output = self.model(tensor)
        mask = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (480, 720) class indices

        overall = self._compute_feasibility(mask)
        w = mask.shape[1]
        left_feas = self._compute_feasibility(mask[:, :w // 2])
        right_feas = self._compute_feasibility(mask[:, w // 2:])

        self._last_result = (overall, left_feas, right_feas)
        return self._last_result

    @staticmethod
    def _compute_feasibility(mask_region):
        """Compute feasibility score from a segmentation mask region.
        Lower rock density → higher feasibility (closer to 1)."""
        counts = np.bincount(mask_region.flatten(), minlength=4)
        ground = max(counts[0], 1)
        small_rock = counts[2]
        big_rock = counts[3]
        rock_ratio = (small_rock + 2 * big_rock) / ground
        # score: 1.0 when no rocks, approaches 0 as rock density increases
        score = 1.0 / (1.0 + rock_ratio * SEG_ROCK_PENALTY)
        # Sigmoid smoothing: maps score [0,1] → feasibility [~0, ~1]
        # score=1 → 1/(1+exp(-5)) ≈ 0.993, score=0 → 1/(1+exp(5)) ≈ 0.007
        return float(1.0 / (1.0 + np.exp(-(score * 10.0 - 5.0))))

    @property
    def last_result(self):
        return self._last_result


class DummySegmenter:
    """No-op when segmentation is disabled."""
    def analyze(self, rgb_frame):
        return (0.8, 0.8, 0.8)

    @property
    def last_result(self):
        return (0.8, 0.8, 0.8)


# ---------------------------------------------------------------------------
# Gimbal Scanner
# ---------------------------------------------------------------------------

class ObstacleMemory:
    """Remembers obstacle positions in world frame so the rover knows when it's past them."""

    def __init__(self):
        self.obstacles = []  # list of (x, y, timestamp, source)

    def add(self, x, y, source="depth"):
        """Record an obstacle at world position (x, y)."""
        now = time.time()
        # Don't add duplicates too close together
        for ox, oy, _, _ in self.obstacles:
            if math.sqrt((ox - x)**2 + (oy - y)**2) < 0.2:
                return
        self.obstacles.append((x, y, now, source))

    def prune(self):
        """Remove old obstacles."""
        now = time.time()
        self.obstacles = [(x, y, t, s) for x, y, t, s in self.obstacles
                          if now - t < OBSTACLE_MEMORY_DURATION]

    def nearest_in_front(self, rover_x, rover_y, rover_heading):
        """Find nearest obstacle that is still in front of the rover.
        Returns (distance, angle_offset) or (None, None) if none in front.
        """
        self.prune()
        best_dist = None
        best_angle = None
        for ox, oy, _, _ in self.obstacles:
            dx = ox - rover_x
            dy = oy - rover_y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 0.05:
                continue
            # Angle from rover heading to obstacle
            angle_to_obs = math.atan2(dy, dx) - rover_heading
            # Normalize to [-pi, pi]
            angle_to_obs = (angle_to_obs + math.pi) % (2 * math.pi) - math.pi
            # Only consider obstacles in front (within ±90°)
            if abs(angle_to_obs) < math.pi / 2:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_angle = angle_to_obs
        return best_dist, best_angle

    def all_passed(self, rover_x, rover_y, rover_heading):
        """True if all remembered obstacles are behind the rover."""
        self.prune()
        for ox, oy, _, _ in self.obstacles:
            dx = ox - rover_x
            dy = oy - rover_y
            # Project onto rover's forward axis
            forward_dist = dx * math.cos(rover_heading) + dy * math.sin(rover_heading)
            if forward_dist > -OBSTACLE_PASSED_BEHIND:
                return False  # This obstacle is still ahead or beside us
        return True

    def count(self):
        self.prune()
        return len(self.obstacles)

    def clear(self):
        self.obstacles = []


class GimbalScanner:
    """Periodically pans the gimbal to scan the environment.

    Cycle: center → right → center → left → center → ...
    Camera MUST be centered when depth is captured for the RL model.
    Also captures depth readings during side scans to populate obstacle memory.
    """

    def __init__(self, base):
        self.base = base
        self._scan_step = 0           # counts control steps
        self._current_target = 0      # current pan angle target
        self._scan_phase = 0          # 0=center, 1=right, 2=center, 3=left
        self._centering_countdown = 0 # steps remaining before camera is settled
        self._last_scan_depth = {}    # {'left': depth, 'right': depth}

    def is_centered(self):
        """True if camera is at forward-facing position and settled."""
        return self._current_target == 0 and self._centering_countdown <= 0

    def center(self):
        """Immediately return gimbal to forward position."""
        self._current_target = 0
        self._centering_countdown = 0
        self._scan_phase = 0
        if self.base:
            self.base.gimbal_ctrl(0, GIMBAL_TILT_DEFAULT, 0, 0)

    def record_scan_depth(self, direction, depth_m):
        """Record the closest depth seen during a side scan."""
        self._last_scan_depth[direction] = depth_m

    def get_scan_depths(self):
        """Return and clear the side scan depth readings."""
        d = dict(self._last_scan_depth)
        self._last_scan_depth = {}
        return d

    def update(self, step, avoid_active):
        """Called every control loop iteration."""
        if self.base is None:
            return None

        # During avoidance, keep camera centered
        if avoid_active:
            if self._current_target != 0:
                self.center()
            return None

        # If we're counting down after centering, just wait
        if self._centering_countdown > 0:
            self._centering_countdown -= 1
            return 'centering'

        # Only change gimbal position every GIMBAL_SCAN_INTERVAL steps
        if step % GIMBAL_SCAN_INTERVAL != 0:
            return None

        # Scan cycle: center(0) → right(1) → center(2) → left(3) → center(0)
        self._scan_phase = (self._scan_phase + 1) % 4

        if self._scan_phase == 1:
            self._current_target = -GIMBAL_SCAN_ANGLE
            self.base.gimbal_ctrl(-GIMBAL_SCAN_ANGLE, GIMBAL_TILT_DEFAULT,
                                  GIMBAL_SCAN_SPEED, GIMBAL_SCAN_ACCEL)
            return 'scan_right'
        elif self._scan_phase == 3:
            self._current_target = GIMBAL_SCAN_ANGLE
            self.base.gimbal_ctrl(GIMBAL_SCAN_ANGLE, GIMBAL_TILT_DEFAULT,
                                  GIMBAL_SCAN_SPEED, GIMBAL_SCAN_ACCEL)
            return 'scan_left'
        else:
            self._current_target = 0
            self._centering_countdown = GIMBAL_SETTLE_STEPS
            self.base.gimbal_ctrl(0, GIMBAL_TILT_DEFAULT,
                                  GIMBAL_SCAN_SPEED, GIMBAL_SCAN_ACCEL)
            return 'centering'


# ---------------------------------------------------------------------------
# Odometry Tracker
# ---------------------------------------------------------------------------

class OdometryTracker:
    """Reads ESP32 encoder feedback to track distance, heading, and actual velocity."""

    def __init__(self, base):
        self.base = base
        self._last_odl = None
        self._last_odr = None
        self._last_time = None

        # Accumulated state
        self.total_distance = 0.0      # meters traveled (path length)
        self.heading = 0.0             # radians, 0 = initial forward
        self.x = 0.0                   # meters, position in start frame
        self.y = 0.0                   # meters, position in start frame
        self.actual_linear_vel = 0.0   # m/s (measured from encoders)
        self.actual_angular_vel = 0.0  # rad/s (measured)

    @property
    def displacement(self):
        """Straight-line distance from start position."""
        return math.sqrt(self.x**2 + self.y**2)

    def reset(self):
        """Reset all odometry state."""
        self._last_odl = None
        self._last_odr = None
        self._last_time = None
        self.total_distance = 0.0
        self.heading = 0.0
        self.x = 0.0
        self.y = 0.0
        self.actual_linear_vel = 0.0
        self.actual_angular_vel = 0.0

    def update(self):
        """Poll feedback_data and update odometry. Returns True if data was read."""
        if self.base is None:
            return False

        try:
            data = self.base.feedback_data()
        except Exception:
            return False

        if data is None or not isinstance(data, dict) or 'odl' not in data:
            return False

        now = time.time()
        odl = data['odl'] * ENCODER_SCALE
        odr = data['odr'] * ENCODER_SCALE

        if self._last_odl is not None and self._last_time is not None:
            dt = now - self._last_time
            if dt > 0.001:
                dl = odl - self._last_odl
                dr = odr - self._last_odr

                # Sanity check: discard if delta > 1m in one tick (wraparound)
                if abs(dl) > 1.0 or abs(dr) > 1.0:
                    self._last_odl = odl
                    self._last_odr = odr
                    self._last_time = now
                    return False

                linear_dist = (dl + dr) / 2.0
                angular_dist = (dr - dl) / WHEEL_SEPARATION

                self.actual_linear_vel = linear_dist / dt
                self.actual_angular_vel = angular_dist / dt
                self.total_distance += abs(linear_dist)
                self.heading += angular_dist
                # Update x, y position in start frame
                self.x += linear_dist * math.cos(self.heading)
                self.y += linear_dist * math.sin(self.heading)

        self._last_odl = odl
        self._last_odr = odr
        self._last_time = now
        return True

    def velocity_error(self, commanded_linear, commanded_angular):
        """Return (linear_error, angular_error) between commanded and actual."""
        return (
            commanded_linear - self.actual_linear_vel,
            commanded_angular - self.actual_angular_vel,
        )


# ---------------------------------------------------------------------------
# Destination Tracker
# ---------------------------------------------------------------------------

class DestinationTracker:
    """Monitors progress toward a distance goal using displacement (straight-line)."""

    def __init__(self, odom):
        self.odom = odom
        self.target_distance = None
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_heading = 0.0
        self.reached = False

    def set_destination(self, distance_meters):
        """Set target: reach this displacement from current position."""
        self.target_distance = distance_meters
        self.start_x = self.odom.x
        self.start_y = self.odom.y
        self.start_heading = self.odom.heading
        self.reached = False
        print(f"[NAV] Destination set: {distance_meters:.1f}m displacement")

    def clear(self):
        """Clear destination."""
        self.target_distance = None
        self.reached = False

    @property
    def displacement_from_start(self):
        """Straight-line distance from where destination was set."""
        dx = self.odom.x - self.start_x
        dy = self.odom.y - self.start_y
        return math.sqrt(dx**2 + dy**2)

    def check(self):
        """Check progress toward destination.

        Returns:
            (status, remaining_meters, heading_drift_degrees)
            status: None, 'tracking', 'reached', or 'off_course'
        """
        if self.target_distance is None:
            return None, 0.0, 0.0

        disp = self.displacement_from_start
        remaining = self.target_distance - disp
        heading_drift = math.degrees(abs(self.odom.heading - self.start_heading))

        if remaining <= DESTINATION_REACHED_TOL:
            self.reached = True
            return 'reached', 0.0, heading_drift

        if heading_drift > DESTINATION_OFF_COURSE_DEG:
            return 'off_course', remaining, heading_drift

        return 'tracking', remaining, heading_drift


# ---------------------------------------------------------------------------
# Depth Camera Interface
# ---------------------------------------------------------------------------

class DepthCamera:
    """Captures depth frames AND optional RGB from the OAK-D Lite camera."""

    def __init__(self, enable_rgb=False):
        self.depth_queue = None
        self.rgb_queue = None
        self.enable_rgb = enable_rgb
        self._init_camera()

    def _init_camera(self):
        try:
            import depthai as dai
            print(f"[INFO] depthai version: {dai.__version__}")

            self._pipeline = dai.Pipeline()

            stereo = self._pipeline.create(dai.node.StereoDepth).build(
                autoCreateCameras=True,
                presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
                size=(640, 400)
            )
            self.depth_queue = stereo.depth.createOutputQueue()

            if self.enable_rgb:
                try:
                    cam_rgb = self._pipeline.create(dai.node.Camera).build(
                        dai.CameraBoardSocket.CAM_A,
                        sensorResolution=(1920, 1080),
                    )
                    video_out = cam_rgb.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
                    self.rgb_queue = video_out.createOutputQueue()
                    print("[OK] RGB camera enabled for object detection")
                except Exception as e:
                    print(f"[WARN] Could not set up RGB camera: {e}")
                    self.enable_rgb = False

            self._pipeline.start()
            print("[OK] OAK-D Lite depth camera initialized")

        except ImportError:
            print("[ERROR] depthai not installed. Use --demo mode.")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Could not initialize OAK-D Lite: {e}")
            sys.exit(1)

    def get_depth_frame(self):
        """Returns (depth_meters, min_depth, left_depth, right_depth).
        depth_meters is 128x128 float32 in raw meters [0, 10].

        IMPORTANT: When the stereo camera can't compute depth (object too close,
        no texture, etc.) it returns 0mm. A high fraction of zero pixels means
        something is RIGHT IN FRONT of the camera — we treat that as very close
        (OBSTACLE_VERY_CLOSE), not as "no data" (999).
        """
        import cv2

        OBSTACLE_VERY_CLOSE = 0.08  # meters — "something is touching the camera"
        ZERO_FRACTION_THRESHOLD = 0.85  # if >85% of pixels are zero → very close

        data = self.depth_queue.get()
        depth_frame = data.getCvFrame()
        depth_resized = cv2.resize(depth_frame, MODEL_INPUT_SIZE,
                                   interpolation=cv2.INTER_AREA)
        depth_meters = depth_resized.astype(np.float32) / 1000.0
        depth_meters = np.clip(depth_meters, 0.0, 10.0)

        h, w = depth_meters.shape
        center_strip = depth_meters[int(h*0.2):int(h*0.8), :]
        total_pixels = center_strip.size

        # Count zero/near-zero pixels (stereo failure = object too close)
        zero_pixels = np.sum(center_strip < 0.02)
        zero_fraction = zero_pixels / total_pixels if total_pixels > 0 else 0

        # If most of the center strip is zero → something is blocking the camera
        if zero_fraction > ZERO_FRACTION_THRESHOLD:
            min_depth = OBSTACLE_VERY_CLOSE
            left_zero = np.sum(center_strip[:, :w//2] < 0.02)
            right_zero = np.sum(center_strip[:, w//2:] < 0.02)
            left_total = center_strip[:, :w//2].size
            right_total = center_strip[:, w//2:].size

            left_depth = OBSTACLE_VERY_CLOSE if left_zero / left_total > ZERO_FRACTION_THRESHOLD else 999.0
            right_depth = OBSTACLE_VERY_CLOSE if right_zero / right_total > ZERO_FRACTION_THRESHOLD else 999.0

            # Fill zero pixels in the depth frame so the RL model sees "very close"
            depth_meters[depth_meters < 0.02] = OBSTACLE_VERY_CLOSE
            return depth_meters, min_depth, left_depth, right_depth

        # Normal case: enough valid pixels to compute stats
        valid_center = center_strip[center_strip > 0.02]
        if len(valid_center) > 10:
            min_depth = float(np.percentile(valid_center, 5))
        elif len(valid_center) > 0:
            min_depth = float(np.min(valid_center))
        else:
            # Very few valid pixels but not enough zeros to trigger "blocked" —
            # this is just stereo noise / texture-less area. Use a safe default.
            min_depth = 1.0  # Assume moderate distance, let RL model handle it

        left_half = center_strip[:, :w//2]
        right_half = center_strip[:, w//2:]
        valid_left = left_half[left_half > 0.02]
        valid_right = right_half[right_half > 0.02]
        left_depth = float(np.median(valid_left)) if len(valid_left) > 5 else 999.0
        right_depth = float(np.median(valid_right)) if len(valid_right) > 5 else 999.0

        # Fill zero pixels so RL model sees something sensible
        depth_meters[depth_meters < 0.02] = min_depth

        return depth_meters, min_depth, left_depth, right_depth

    def get_rgb_frame(self):
        if not self.enable_rgb or self.rgb_queue is None:
            return None
        try:
            data = self.rgb_queue.tryGet()
            if data is not None:
                return data.getCvFrame()
        except Exception:
            pass
        return None

    def close(self):
        if hasattr(self, '_pipeline') and self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass


class DummyDepthCamera:
    """Generates fake depth data for demo/testing without a real camera."""

    def __init__(self):
        self._step = 0
        print("[DEMO] Using dummy depth camera (no real camera)")

    def get_depth_frame(self):
        self._step += 1
        depth = np.ones(MODEL_INPUT_SIZE, dtype=np.float32) * 3.0
        depth[:, :30] = 0.5
        depth[:, -30:] = 0.5
        if (self._step // 10) % 3 == 1:
            depth[:, 40:90] = 0.2
        depth += np.random.normal(0, 0.1, MODEL_INPUT_SIZE).astype(np.float32)
        depth = np.clip(depth, 0.0, 10.0)
        min_depth = float(np.percentile(depth[depth > 0.1], 5)) if np.sum(depth > 0.1) > 10 else 999.0
        left_depth = float(np.median(depth[:, :64]))
        right_depth = float(np.median(depth[:, 64:]))
        return depth, min_depth, left_depth, right_depth

    def get_rgb_frame(self):
        return None

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Reward Logger
# ---------------------------------------------------------------------------

# Reward shaping constants (for logging — NOT used for weight updates)
REWARD_FORWARD_PROGRESS = 0.2     # per step with positive linear vel
REWARD_WAYPOINT_REACHED = 10.0    # goal reached (face target or distance)
PENALTY_BACKUP = -1.0             # triggered backup maneuver
PENALTY_EMERGENCY = -2.0          # emergency stop
PENALTY_CLOSE_OBSTACLE = -0.5     # in STOP/AVOID zone
PENALTY_SPINNING = -0.3           # avoidance step (turning in place)
PENALTY_IDLE = -0.1               # near-zero velocity


class RewardLogger:
    """Logs per-step simulated rewards to a CSV file for post-hoc analysis.

    This does NOT update model weights — it records what rewards *would be*
    assigned if we were training online. Useful for research analysis and
    poster data.

    Columns:
        step, timestamp, reward, event, linear_vel, angular_vel,
        min_depth, left_depth, right_depth, mode, face_state,
        seg_left, seg_right, odom_dist, odom_disp, heading_deg
    """

    COLUMNS = [
        'step', 'timestamp', 'reward', 'event', 'linear_vel', 'angular_vel',
        'min_depth', 'left_depth', 'right_depth', 'mode', 'face_state',
        'seg_left', 'seg_right', 'odom_dist', 'odom_disp', 'heading_deg',
    ]

    def __init__(self, log_path=None):
        if log_path is None:
            ts = time.strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f'rewards_{ts}.csv')
        self.log_path = log_path
        self._file = open(log_path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)
        self._total_reward = 0.0
        self._step_count = 0
        print(f"[OK] Reward logger writing to: {log_path}")

    def compute_reward(self, linear_vel, angular_vel, min_depth, mode,
                       face_state=None, reached_goal=False):
        """Compute the shaped reward for the current step.

        Returns:
            (reward, event_str) — the reward value and a human-readable event tag
        """
        reward = 0.0
        events = []

        # Goal reached — big positive
        if reached_goal:
            reward += REWARD_WAYPOINT_REACHED
            events.append('GOAL')
            return reward, '+'.join(events) if events else 'STEP'

        # Emergency stop
        if min_depth <= EMERGENCY_STOP_DIST:
            reward += PENALTY_EMERGENCY
            events.append('EMERGENCY')

        # Backup zone
        elif min_depth < BACKUP_DIST:
            reward += PENALTY_BACKUP
            events.append('BACKUP')

        # Close obstacle (STOP/AVOID zone)
        elif min_depth < OBSTACLE_STOP_DIST:
            reward += PENALTY_CLOSE_OBSTACLE
            events.append('CLOSE')

        # Spinning / avoidance
        if mode and mode.startswith('AVOID'):
            reward += PENALTY_SPINNING
            events.append('AVOID')

        # Forward progress
        if linear_vel > 0.02:
            reward += REWARD_FORWARD_PROGRESS * (linear_vel / MAX_LINEAR_SPEED)
            events.append('FWD')

        # Near-idle
        elif abs(linear_vel) < 0.01 and abs(angular_vel) < 0.05:
            reward += PENALTY_IDLE
            events.append('IDLE')

        return reward, '+'.join(events) if events else 'STEP'

    def log(self, step, reward, event, linear_vel, angular_vel,
            min_depth, left_depth, right_depth, mode,
            face_state, seg_left, seg_right,
            odom_dist, odom_disp, heading_deg):
        """Write one row to the CSV."""
        self._total_reward += reward
        self._step_count += 1
        self._writer.writerow([
            step, f'{time.time():.3f}', f'{reward:.4f}', event,
            f'{linear_vel:.4f}', f'{angular_vel:.4f}',
            f'{min_depth:.3f}', f'{left_depth:.3f}', f'{right_depth:.3f}',
            mode or '', face_state or '',
            f'{seg_left:.3f}', f'{seg_right:.3f}',
            f'{odom_dist:.3f}', f'{odom_disp:.3f}', f'{heading_deg:.1f}',
        ])
        # Flush periodically so data isn't lost on crash
        if self._step_count % 25 == 0:
            self._file.flush()

    def close(self):
        """Flush and close the CSV file. Print summary."""
        self._file.flush()
        self._file.close()
        avg = self._total_reward / max(self._step_count, 1)
        print(f"[REWARD] Logged {self._step_count} steps to {self.log_path}")
        print(f"[REWARD] Total reward: {self._total_reward:.2f}, "
              f"Average: {avg:.4f}/step")


class DummyRewardLogger:
    """No-op reward logger when --log-rewards is not enabled."""
    def compute_reward(self, *a, **kw):
        return 0.0, ''
    def log(self, *a, **kw):
        pass
    def close(self):
        pass


# ---------------------------------------------------------------------------
# RL Deployment Controller
# ---------------------------------------------------------------------------

class RLDeployController:
    """Main deployment controller: camera -> model -> rover."""

    def __init__(self, model_path, demo=False, max_speed=MAX_LINEAR_SPEED,
                 dry_run=False, port=SERIAL_PORT, enable_detect=True,
                 enable_gimbal=True, target_mode='distance',
                 enable_segmentation=False, seg_model_path=None,
                 face_stop_distance=FACE_STOP_DISTANCE,
                 log_rewards=False, reward_log_path=None):
        self.max_speed = min(abs(max_speed), MAX_LINEAR_SPEED)
        self.dry_run = dry_run
        self.running = False
        self.base = None
        self.target_mode = target_mode

        # Avoidance state machine
        self._avoid_turn_dir = 0
        self._avoid_steps_left = 0
        self._avoid_total_steps = 0   # total steps in current avoidance (for cap)
        self._last_detected_obj = ""

        # Course correction state
        self._target_heading = 0.0  # the heading we WANT to maintain (radians)
        self._avoiding = False      # True while in avoidance maneuver

        # Obstacle memory
        self.obstacle_map = ObstacleMemory()

        # --- Load the trained model ---
        print(f"[INFO] Loading model from: {model_path}")
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path, device='cpu')
            print(f"[OK] Model loaded successfully")
            print(f"     Observation space: {self.model.observation_space}")
            print(f"     Action space: {self.model.action_space}")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            sys.exit(1)

        # --- Initialize camera ---
        # Face mode needs RGB even if detection is off
        needs_rgb = enable_detect or target_mode == 'face' or enable_segmentation
        if demo:
            self.camera = DummyDepthCamera()
        else:
            self.camera = DepthCamera(enable_rgb=needs_rgb)

        # --- Initialize object detector ---
        if enable_detect and not demo:
            try:
                self.detector = ObjectDetector()
            except Exception as e:
                print(f"[WARN] Object detection unavailable: {e}")
                self.detector = DummyDetector()
        else:
            self.detector = DummyDetector()

        # --- Initialize rover connection ---
        if not dry_run:
            try:
                from base_ctrl import BaseController
                self.base = BaseController(port, BAUD_RATE)
                time.sleep(0.5)
                print(f"[OK] Connected to rover on {port}")
            except Exception as e:
                print(f"[ERROR] Could not connect to rover: {e}")
                sys.exit(1)
        else:
            print("[DRY RUN] Commands will be printed, not sent")

        # --- Initialize gimbal scanner ---
        base_for_peripherals = self.base if not dry_run else None
        if enable_gimbal:
            self.gimbal = GimbalScanner(base_for_peripherals)
            if base_for_peripherals:
                # Force gimbal to level position on startup
                base_for_peripherals.gimbal_ctrl(0, 0, 0, 0)
                time.sleep(0.5)
                self.gimbal.center()
                time.sleep(0.3)
                print(f"[OK] Gimbal centered (pan=0, tilt={GIMBAL_TILT_DEFAULT})")
        else:
            self.gimbal = GimbalScanner(None)  # no-op

        # --- Initialize odometry ---
        self.odom = OdometryTracker(base_for_peripherals)

        # --- Initialize destination tracker ---
        self.destination = DestinationTracker(self.odom)

        # --- Initialize face navigation ---
        self.face_tracker = None
        self.face_nav = None
        if target_mode == 'face':
            if demo:
                self.face_tracker = DummyFaceTracker()
            else:
                self.face_tracker = FaceTracker(detector=self.detector)
            self.face_nav = FaceNavigationSM(self.face_tracker, self.odom, face_stop_distance)
            # Disable gimbal scanning in face mode — camera stays centered
            self.gimbal = GimbalScanner(None)
            print(f"[OK] Face navigation mode (stop at {face_stop_distance:.1f}m)")

        # --- Initialize terrain segmenter ---
        if enable_segmentation and seg_model_path:
            try:
                self.segmenter = TerrainSegmenter(seg_model_path)
            except Exception as e:
                print(f"[WARN] Terrain segmentation unavailable: {e}")
                self.segmenter = DummySegmenter()
        else:
            self.segmenter = DummySegmenter()

        # --- Initialize reward logger ---
        if log_rewards:
            self.reward_logger = RewardLogger(reward_log_path)
        else:
            self.reward_logger = DummyRewardLogger()

        # --- Signal handlers ---
        signal.signal(signal.SIGINT, self._emergency_stop)
        signal.signal(signal.SIGTERM, self._emergency_stop)

    def _emergency_stop(self, signum, frame):
        print("\n[EMERGENCY STOP] Stopping rover immediately!")
        self.running = False
        self._send_velocity(0.0, 0.0)
        self.gimbal.center()
        self.camera.close()
        print(f"[NAV] Path: {self.odom.total_distance:.2f}m, Displacement: {self.odom.displacement:.2f}m, "
              f"Heading: {math.degrees(self.odom.heading):.0f}°")
        sys.exit(0)

    def _send_velocity(self, linear, angular):
        linear = max(-self.max_speed, min(self.max_speed, linear))
        angular = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, angular))
        if self.dry_run:
            print(f"  [CMD] linear={linear:+.3f} m/s  angular={angular:+.3f} rad/s")
        else:
            self.base.base_json_ctrl({"T": 13, "X": round(linear, 3), "Z": round(angular, 3)})

    def _stop_rover(self):
        self._send_velocity(0.0, 0.0)
        if not self.dry_run and self.base:
            time.sleep(0.05)
            self.base.base_json_ctrl({"T": 13, "X": 0, "Z": 0})

    def _get_detection_info(self):
        """Run object detection on RGB frame. Returns (detected, class, center_x, conf)."""
        rgb = self.camera.get_rgb_frame()
        if rgb is None:
            return False, "", 0.5, 0.0

        detections = self.detector.detect(rgb)

        best = None
        best_danger = 0
        for det in detections:
            cls = det['class']
            if cls in DETECT_OBSTACLE_CLASSES:
                danger = DETECT_OBSTACLE_CLASSES[cls] * det['confidence']
                center_dist = abs(det['center_x'] - 0.5)
                danger *= (1.0 - center_dist) if center_dist < 0.4 else 0.3
                if danger > best_danger:
                    best_danger = danger
                    best = det

        if best is not None:
            return True, best['class'], best['center_x'], best['confidence']
        return False, "", 0.5, 0.0

    def _record_obstacle_ahead(self, min_depth):
        """Record an obstacle in the obstacle memory at current position + depth forward."""
        if min_depth < 2.0 and min_depth > EMERGENCY_STOP_DIST:
            obs_x = self.odom.x + min_depth * math.cos(self.odom.heading)
            obs_y = self.odom.y + min_depth * math.sin(self.odom.heading)
            self.obstacle_map.add(obs_x, obs_y, source="depth")

    def _compute_course_correction(self):
        """Compute angular correction to steer back toward target heading.
        Returns angular velocity adjustment (rad/s), or 0 if on course.
        """
        heading_error = self._target_heading - self.odom.heading
        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        if abs(heading_error) < COURSE_CORRECT_DEADBAND:
            return 0.0

        correction = COURSE_CORRECT_GAIN * heading_error
        return max(-COURSE_CORRECT_MAX, min(COURSE_CORRECT_MAX, correction))

    def _compute_avoidance(self, min_depth, left_depth, right_depth,
                           obj_detected, obj_class, obj_center_x, obj_conf,
                           seg_left_feas=0.8, seg_right_feas=0.8):
        """Determine avoidance action with committed turning.
        Returns (override, mode_str).
        seg_left_feas/seg_right_feas: terrain feasibility [0-1] from segmentation.
        """
        stop_dist = OBSTACLE_STOP_DIST
        slow_dist = OBSTACLE_SLOW_DIST
        if obj_detected:
            danger = DETECT_OBSTACLE_CLASSES.get(obj_class, 0.5) * obj_conf
            stop_dist += danger * 0.15
            slow_dist += danger * 0.20
            self._last_detected_obj = f"{obj_class}({obj_conf:.0%})"

        # Committed turn in progress — keep turning until path is REALLY clear
        if self._avoid_steps_left > 0:
            self._avoid_steps_left -= 1
            self._avoid_total_steps += 1
            self._avoiding = True
            # Hit the absolute max? Force exit avoidance — we're probably reading the floor
            if self._avoid_total_steps >= MAX_AVOID_STEPS:
                self._avoiding = False
                self._avoid_steps_left = 0
                self._avoid_total_steps = 0
                return None, "MAX_AVOID"
            # Only clear if we have good clearance AND steps are done
            if self._avoid_steps_left == 0 and min_depth > slow_dist + 0.1:
                self._avoiding = False
                self._avoid_total_steps = 0
                return None, "CLEAR"
            elif self._avoid_steps_left == 0 and min_depth <= slow_dist + 0.1:
                # Path still not clear — extend the turn (but cap will catch infinite loops)
                self._avoid_steps_left = COMMITTED_TURN_STEPS // 2
            # Move forward slowly while turning if there's some room, else pure rotation
            fwd = self.max_speed * 0.15 if min_depth > stop_dist else 0.0
            return (fwd, self._avoid_turn_dir * OBSTACLE_TURN_SPEED), "AVOID"

        # New avoidance trigger
        if min_depth < stop_dist:
            # Remember the obstacle location
            self._record_obstacle_ahead(min_depth)

            if obj_detected and abs(obj_center_x - 0.5) > 0.1:
                self._avoid_turn_dir = +1 if obj_center_x > 0.5 else -1
            elif left_depth > right_depth + 0.05:
                self._avoid_turn_dir = +1
            elif right_depth > left_depth + 0.05:
                self._avoid_turn_dir = -1
            elif abs(left_depth - right_depth) < 0.1:
                # Depths similar — use segmentation to prefer clearer terrain
                if seg_left_feas > seg_right_feas + 0.1:
                    self._avoid_turn_dir = +1  # go left (clearer terrain)
                elif seg_right_feas > seg_left_feas + 0.1:
                    self._avoid_turn_dir = -1  # go right (clearer terrain)
                elif self._avoid_turn_dir == 0:
                    self._avoid_turn_dir = -1
            else:
                if self._avoid_turn_dir == 0:
                    self._avoid_turn_dir = -1

            self._avoid_steps_left = COMMITTED_TURN_STEPS
            self._avoid_total_steps = 0  # reset total counter for new avoidance
            self._avoiding = True
            fwd = self.max_speed * 0.15 if min_depth > BACKUP_DIST else 0.0
            return (fwd, self._avoid_turn_dir * OBSTACLE_TURN_SPEED), "AVOID"

        elif min_depth < slow_dist:
            # Remember the obstacle and slow down with steering bias
            self._record_obstacle_ahead(min_depth)
            return 'slow', "SLOW"

        # Clear path — check if we need course correction
        self._avoiding = False
        self._avoid_turn_dir = 0
        self._last_detected_obj = ""

        # Apply heading-based course correction
        correction = self._compute_course_correction()
        if abs(correction) > 0.01:
            return 'course_correct', "CORRECT"

        return None, ""

    def run(self, duration=None, target_distance=None):
        """Main control loop."""
        self.running = True
        step = 0
        start_time = time.time()
        interval = 1.0 / CONTROL_HZ
        detect_interval = 3
        last_detect_result = (False, "", 0.5, 0.0)
        last_cmd_linear = 0.0
        last_cmd_angular = 0.0

        if target_distance is not None:
            self.destination.set_destination(target_distance)

        print("\n" + "=" * 60)
        print("  RL NAVIGATION ACTIVE")
        print(f"  Max speed: {self.max_speed:.2f} m/s")
        print(f"  Control rate: {CONTROL_HZ} Hz")
        print(f"  Avoidance: stop={OBSTACLE_STOP_DIST:.2f}m slow={OBSTACLE_SLOW_DIST:.2f}m")
        print(f"  Object detection: {'ON' if not isinstance(self.detector, DummyDetector) else 'OFF'}")
        print(f"  Gimbal scanning: {'ON' if self.gimbal.base else 'OFF'}")
        print(f"  Target mode: {self.target_mode}")
        if self.face_nav:
            print(f"  Face stop distance: {self.face_nav.stop_distance:.1f}m")
        print(f"  Segmentation: {'ON' if not isinstance(self.segmenter, DummySegmenter) else 'OFF'}")
        if target_distance:
            print(f"  Target distance: {target_distance:.1f}m")
        if duration:
            print(f"  Duration: {duration:.0f}s")
        print("  Press Ctrl+C for emergency stop")
        print("=" * 60 + "\n")

        try:
            while self.running:
                loop_start = time.time()

                if duration and (time.time() - start_time) >= duration:
                    print("[INFO] Duration reached, stopping.")
                    break

                # --- 1. Ensure gimbal is centered for depth capture ---
                if not self.gimbal.is_centered():
                    # Skip this frame — gimbal is moving, depth would be wrong
                    self.gimbal.update(step, self._avoid_steps_left > 0)
                    step += 1
                    elapsed_loop = time.time() - loop_start
                    if interval - elapsed_loop > 0:
                        time.sleep(interval - elapsed_loop)
                    continue

                # --- 2. Get depth frame (camera is forward-facing) ---
                depth_frame, min_depth, left_depth, right_depth = self.camera.get_depth_frame()

                # --- 3. Update odometry ---
                self.odom.update()

                # --- 4. Safety tiers: emergency stop → backup → avoidance ---

                # Tier 1: EMERGENCY STOP — object at ≤5cm or stereo dead zone
                if min_depth <= EMERGENCY_STOP_DIST:
                    self._record_obstacle_ahead(min_depth)
                    self._stop_rover()
                    print(f"  [EMERGENCY] Obstacle at {min_depth:.2f}m! Stopped. Backing up...")
                    time.sleep(0.3)
                    # Back up aggressively for BACKUP_DURATION seconds
                    backup_turn = BACKUP_TURN_SPEED if left_depth > right_depth else -BACKUP_TURN_SPEED
                    backup_end = time.time() + BACKUP_DURATION
                    while time.time() < backup_end and self.running:
                        self._send_velocity(-self.max_speed * BACKUP_SPEED_FRAC, backup_turn)
                        time.sleep(interval)
                    self._stop_rover()
                    # Now do a committed turn to get away
                    self._avoid_turn_dir = +1 if left_depth > right_depth else -1
                    self._avoid_steps_left = COMMITTED_TURN_STEPS
                    self._avoiding = True
                    step += 1
                    continue

                # Tier 2: ACTIVE BACKUP — object at ≤15cm, back up while turning
                if min_depth < BACKUP_DIST:
                    self._record_obstacle_ahead(min_depth)
                    if step % CONTROL_HZ == 0:
                        print(f"  [BACKUP] Obstacle at {min_depth:.2f}m! Backing up...")
                    backup_turn = BACKUP_TURN_SPEED if left_depth > right_depth else -BACKUP_TURN_SPEED
                    self._send_velocity(-self.max_speed * BACKUP_SPEED_FRAC, backup_turn)
                    # Set up committed turn for when we back up far enough
                    if self._avoid_steps_left == 0:
                        self._avoid_turn_dir = +1 if left_depth > right_depth else -1
                        self._avoid_steps_left = COMMITTED_TURN_STEPS
                        self._avoiding = True
                    step += 1
                    time.sleep(interval)
                    continue

                # --- 5. Object detection (every N frames) ---
                if step % detect_interval == 0:
                    last_detect_result = self._get_detection_info()
                obj_detected, obj_class, obj_center_x, obj_conf = last_detect_result

                # --- 5b. Terrain segmentation (periodic) ---
                seg_left, seg_right = 0.8, 0.8
                if step % SEG_RUN_INTERVAL == 0:
                    seg_rgb = self.camera.get_rgb_frame()
                    if seg_rgb is not None:
                        self.segmenter.analyze(seg_rgb)
                _, seg_left, seg_right = self.segmenter.last_result

                # --- 5c. Face navigation (before RL inference) ---
                face_override = None
                state_vector = DEFAULT_STATE.copy()
                if self.face_nav is not None:
                    face_rgb = self.camera.get_rgb_frame()
                    face_override, state_vector = self.face_nav.update(face_rgb, depth_frame)
                    if self.face_nav.reached:
                        print(f"\n[NAV] FACE TARGET {'REACHED' if self.face_nav.state == 'ARRIVED' else 'SEARCH TIMEOUT'}!")
                        if self.face_nav.state == 'ARRIVED':
                            r, e = self.reward_logger.compute_reward(
                                0, 0, min_depth, mode, face_state='ARRIVED', reached_goal=True)
                            self.reward_logger.log(
                                step, r, e, 0, 0, min_depth, left_depth, right_depth,
                                'ARRIVED', 'ARRIVED', seg_left, seg_right,
                                self.odom.total_distance, self.odom.displacement,
                                math.degrees(self.odom.heading))
                        break

                # --- 6. Compute avoidance ---
                override, mode = self._compute_avoidance(
                    min_depth, left_depth, right_depth,
                    obj_detected, obj_class, obj_center_x, obj_conf,
                    seg_left_feas=seg_left, seg_right_feas=seg_right)

                # --- 7. RL inference ---
                obs = {
                    'state': state_vector,
                    'image_front': depth_frame.reshape(128, 128, 1),
                }
                action, _ = self.model.predict(obs, deterministic=True)
                raw_linear = float(action[0])
                raw_angular = float(action[1])

                linear_vel = raw_linear * self.max_speed
                angular_vel = raw_angular * MAX_ANGULAR_SPEED
                if linear_vel < 0:
                    linear_vel = max(linear_vel, -self.max_speed * 0.3)

                # --- 8. Apply overrides (priority: safety > face search > avoidance > RL) ---
                if face_override is not None:
                    # Face SM is in SEARCH or ARRIVED — it controls the rover
                    linear_vel, angular_vel = face_override
                elif override is not None and isinstance(override, tuple):
                    linear_vel, angular_vel = override
                elif override == 'slow':
                    speed_scale = (min_depth - OBSTACLE_STOP_DIST) / (OBSTACLE_SLOW_DIST - OBSTACLE_STOP_DIST)
                    speed_scale = max(0.05, min(0.7, speed_scale))
                    linear_vel *= speed_scale
                    # Stronger steering bias away from the closer side
                    steer_strength = OBSTACLE_TURN_SPEED * 0.5 * (1 - speed_scale)
                    if left_depth > right_depth + 0.05:
                        angular_vel += steer_strength
                    elif right_depth > left_depth + 0.05:
                        angular_vel -= steer_strength
                    else:
                        # Equal depth — steer away from whichever side has a remembered obstacle
                        dist_front, angle_front = self.obstacle_map.nearest_in_front(
                            self.odom.x, self.odom.y, self.odom.heading)
                        if dist_front is not None and angle_front is not None:
                            angular_vel -= 0.2 * (1 if angle_front > 0 else -1)
                elif override == 'course_correct':
                    # Blend RL output with heading correction to get back on course
                    correction = self._compute_course_correction()
                    angular_vel += correction

                # --- 9. Send command ---
                self._send_velocity(linear_vel, angular_vel)
                last_cmd_linear = linear_vel
                last_cmd_angular = angular_vel

                # --- 9b. Reward logging ---
                face_state_str = self.face_nav.state if self.face_nav else None
                reward, event = self.reward_logger.compute_reward(
                    linear_vel, angular_vel, min_depth, mode,
                    face_state=face_state_str, reached_goal=False)
                self.reward_logger.log(
                    step, reward, event, linear_vel, angular_vel,
                    min_depth, left_depth, right_depth, mode,
                    face_state_str, seg_left, seg_right,
                    self.odom.total_distance, self.odom.displacement,
                    math.degrees(self.odom.heading))

                # --- 10. Check destination ---
                if self.destination.target_distance is not None:
                    status, remaining, drift = self.destination.check()
                    if status == 'reached':
                        print(f"\n[NAV] DESTINATION REACHED! "
                              f"Displacement: {self.destination.displacement_from_start:.2f}m, "
                              f"Path length: {self.odom.total_distance:.2f}m, "
                              f"Heading drift: {drift:.0f}")
                        r, e = self.reward_logger.compute_reward(
                            0, 0, min_depth, mode, reached_goal=True)
                        self.reward_logger.log(
                            step, r, e, 0, 0, min_depth, left_depth, right_depth,
                            'REACHED', None, seg_left, seg_right,
                            self.odom.total_distance, self.odom.displacement,
                            math.degrees(self.odom.heading))
                        break
                    elif status == 'off_course' and step % CONTROL_HZ == 0:
                        print(f"  [NAV] Off course! Drift: {drift:.0f}, "
                              f"remaining: {remaining:.2f}m")

                # --- 11. Log ---
                step += 1
                if step % CONTROL_HZ == 0:
                    elapsed = time.time() - start_time
                    mode_str = f" [{mode}]" if mode else ""
                    obj_str = f" obj={self._last_detected_obj}" if self._last_detected_obj else ""

                    # Odometry info
                    odom_str = f" dist={self.odom.total_distance:.2f}m disp={self.odom.displacement:.2f}m"
                    odom_str += f" hdg={math.degrees(self.odom.heading):.0f}"
                    if self.odom.actual_linear_vel != 0:
                        lin_err, ang_err = self.odom.velocity_error(last_cmd_linear, last_cmd_angular)
                        odom_str += f" v_err={lin_err:+.3f}"

                    # Destination info
                    dest_str = ""
                    if self.destination.target_distance is not None:
                        _, rem, _ = self.destination.check()
                        dest_str = f" rem={rem:.2f}m"

                    # Obstacle memory
                    obs_count = self.obstacle_map.count()
                    obs_str = f" obs_mem={obs_count}" if obs_count > 0 else ""

                    # Course correction info
                    cc_str = ""
                    if mode == "CORRECT":
                        cc = self._compute_course_correction()
                        cc_str = f" cc={cc:+.3f}"

                    # Face navigation info
                    face_str = ""
                    if self.face_nav is not None:
                        face_str = f" face={self.face_nav.state}"
                        if self.face_tracker and self.face_tracker.last_face:
                            a, d, _ = self.face_tracker.last_face
                            face_str += f"({math.degrees(a):.0f}°,{d:.1f}m)"

                    # Segmentation info
                    seg_str = ""
                    if not isinstance(self.segmenter, DummySegmenter):
                        seg_str = f" seg=({seg_left:.2f}L/{seg_right:.2f}R)"

                    print(f"  [{elapsed:.0f}s]{mode_str}"
                          f" raw=[{raw_linear:+.3f},{raw_angular:+.3f}]"
                          f" → lin={linear_vel:+.3f} ang={angular_vel:+.3f}"
                          f" depth={min_depth:.2f}m (L={left_depth:.2f} R={right_depth:.2f})"
                          f"{obj_str}{odom_str}{dest_str}{obs_str}{cc_str}{face_str}{seg_str}")

                # --- 12. Kick off gimbal scan (after depth was captured) ---
                self.gimbal.update(step, self._avoid_steps_left > 0)

                # --- 13. Maintain control rate ---
                elapsed_loop = time.time() - loop_start
                sleep_time = interval - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            print("\n[INFO] Stopping rover...")
            self._stop_rover()
            self.gimbal.center()
            self.camera.close()
            self.reward_logger.close()
            print(f"[NAV] Path: {self.odom.total_distance:.2f}m, Displacement: {self.odom.displacement:.2f}m, "
                  f"Final heading: {math.degrees(self.odom.heading):.0f}°")
            print(f"[NAV] Obstacles remembered: {self.obstacle_map.count()}")
            print("[INFO] Shutdown complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='RL Deployment Bridge - Team Crater',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 rl_deploy.py --demo --dry-run           # Test without hardware
  python3 rl_deploy.py --max-speed 0.15 --duration 60
  python3 rl_deploy.py --distance 5               # Go 5 meters then stop
  python3 rl_deploy.py --no-detect --no-gimbal     # Minimal mode
  python3 rl_deploy.py --target face --duration 120    # Navigate to a face
  python3 rl_deploy.py --target face --face-distance 1.5  # Stop 1.5m from face
  python3 rl_deploy.py --distance 5 --segmentation    # With terrain segmentation
  python3 rl_deploy.py --distance 5 --log-rewards     # Log rewards to CSV
        """)

    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to trained model .zip (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--demo', action='store_true',
                        help='Use dummy depth camera (no real camera needed)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands instead of sending to rover')
    parser.add_argument('--max-speed', type=float, default=MAX_LINEAR_SPEED,
                        help=f'Max linear speed in m/s (default: {MAX_LINEAR_SPEED})')
    parser.add_argument('--duration', type=float, default=None,
                        help='Run for N seconds then stop')
    parser.add_argument('--distance', type=float, default=None,
                        help='Target distance in meters (stop when reached)')
    parser.add_argument('--port', type=str, default=SERIAL_PORT,
                        help=f'Serial port (default: {SERIAL_PORT})')
    parser.add_argument('--no-detect', action='store_true',
                        help='Disable MobileNet-SSD object detection')
    parser.add_argument('--no-gimbal', action='store_true',
                        help='Disable gimbal pan/tilt scanning')
    parser.add_argument('--target', choices=['distance', 'face'], default='distance',
                        help='Navigation mode: distance-based or face-tracking (default: distance)')
    parser.add_argument('--face-distance', type=float, default=FACE_STOP_DISTANCE,
                        help=f'Stop distance for face target in meters (default: {FACE_STOP_DISTANCE})')
    parser.add_argument('--segmentation', action='store_true',
                        help='Enable UNet terrain segmentation for avoidance')
    parser.add_argument('--seg-model', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'mark model', 'unet_lunar_segmentation.pth'),
                        help='Path to UNet segmentation model')
    parser.add_argument('--log-rewards', action='store_true',
                        help='Log per-step shaped rewards to CSV (no weight updates)')
    parser.add_argument('--reward-log', type=str, default=None,
                        help='Path for reward CSV log (default: rewards_YYYYMMDD_HHMMSS.csv)')

    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════╗
║   Team Crater - RL Deployment Bridge                         ║
║   SRB Visual Nav + Detection + Gimbal + Odometry → Rover     ║
╚════════════════════════════════════════════════════════════╝
""")

    if not args.dry_run and not args.demo:
        print("SAFETY CHECKLIST:")
        print("  [!] Rover is on the ground with clearance")
        print("  [!] You are nearby to press Ctrl+C")
        print(f"  [!] Max speed: {args.max_speed:.2f} m/s")
        print(f"  [!] Avoidance distances: stop={OBSTACLE_STOP_DIST}m slow={OBSTACLE_SLOW_DIST}m")
        print()
        print("Starting in 3 seconds... (Ctrl+C to abort)")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

    controller = RLDeployController(
        model_path=args.model,
        demo=args.demo,
        max_speed=args.max_speed,
        dry_run=args.dry_run,
        port=args.port,
        enable_detect=not args.no_detect,
        enable_gimbal=not args.no_gimbal,
        target_mode=args.target,
        enable_segmentation=args.segmentation,
        seg_model_path=args.seg_model,
        face_stop_distance=args.face_distance,
        log_rewards=args.log_rewards,
        reward_log_path=args.reward_log,
    )

    controller.run(duration=args.duration, target_distance=args.distance)


if __name__ == '__main__':
    main()
