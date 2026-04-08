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
GIMBAL_SCAN_ANGLE = 70        # degrees left/right from center (wider sweep)
GIMBAL_SCAN_SPEED = 60        # servo speed (0=fastest, lower=faster scan)
GIMBAL_SCAN_ACCEL = 40        # servo acceleration
GIMBAL_SCAN_INTERVAL = 8      # control steps between scan moves (1.6s at 5Hz)
GIMBAL_TILT_DEFAULT = 0       # level — tilt down was reading the FLOOR as an obstacle
GIMBAL_TILT_UP = 15           # degrees up for face scanning (people are taller than rover)
GIMBAL_TILT_DOWN = -10        # degrees down for close-range scanning
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
FACE_VFOV_RAD = math.radians(58)       # OAK-D Lite vertical FOV (~58° for 4:3 sensor)
FACE_DEPTH_PATCH = 7                    # pixels to sample for depth median
FACE_STOP_DISTANCE = 0.8                 # meters — comfortable stop distance from person
FACE_RECHECK_STEPS = 2                 # re-detect face every N steps (0.4s at 5Hz) — fast!
FACE_SEARCH_SPIN_SPEED = 0.35          # rad/s rotation during scan (smoother for detection)
FACE_SEARCH_SPIN_DURATION = 8.0        # seconds for a scan rotation
FACE_SEARCH_DRIVE_DURATION = 6.0       # seconds of RL-driven exploration between scans
FACE_LOST_HOLD_TIME = 2.0              # seconds to hold last state when face lost (drive to estimate)
FACE_LOST_SPIN_TIME = 5.0              # seconds to spin searching after losing face
FACE_MIN_NAVIGATE_STEPS = 10           # min steps (2s) before ARRIVED can trigger
FACE_APPROACH_GAIN = 1.5               # angular gain for steering toward face (proportional control)
FACE_APPROACH_MAX_ANG = 0.4            # max angular vel during face approach (rad/s)

# Red target detection (HSV-based — very robust, no ML needed)
RED_HSV_LOWER1 = np.array([0, 120, 70])    # red wraps around 0° in HSV
RED_HSV_UPPER1 = np.array([10, 255, 255])
RED_HSV_LOWER2 = np.array([170, 120, 70])
RED_HSV_UPPER2 = np.array([180, 255, 255])
RED_MIN_AREA_FRAC = 0.002              # minimum fraction of frame area to count as target
RED_STOP_DISTANCE = 0.10               # meters — stop RIGHT next to the target (10cm)

# Terrain segmentation
SEG_RUN_INTERVAL = 10                   # run segmentation every N steps (2s at 5Hz)
SEG_ROCK_PENALTY = 50                   # penalty scale for rock density
SEG_SIGMOID_WIDTH = 0.5                 # sigmoid sharpness for feasibility score


# ---------------------------------------------------------------------------
# Face Tracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """Detects faces/people using a cascade of detectors (best first):
      1. DNN face detector (OpenCV Caffe model — very reliable)
      2. Haar cascade (fast but brittle)
      3. MobileNet-SSD 'person' class (full-body fallback)
    Estimates angle and distance to the largest detection."""

    FACE_DNN_CONFIDENCE = 0.35     # DNN face detection threshold
    PERSON_CONFIDENCE = 0.4        # MobileNet person detection threshold

    def __init__(self, detector=None):
        import cv2

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'ugv_jetson', 'models')

        # --- DNN face detector (best) ---
        self.dnn_net = None
        dnn_proto = os.path.join(models_dir, 'face_deploy.prototxt')
        dnn_model = os.path.join(models_dir, 'face_res10.caffemodel')
        if os.path.exists(dnn_proto) and os.path.exists(dnn_model):
            try:
                self.dnn_net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
                print("[OK] DNN face detector loaded (res10 SSD)")
            except Exception as e:
                print(f"[WARN] DNN face detector failed: {e}")
        else:
            print(f"[WARN] DNN face model not found at {models_dir}")

        # --- Haar cascade (backup) ---
        self.cascade = None
        haar_path = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
        if os.path.exists(haar_path):
            self.cascade = cv2.CascadeClassifier(haar_path)
            if self.cascade.empty():
                self.cascade = None
            else:
                print("[OK] Haar cascade face detector loaded")

        # --- MobileNet-SSD person fallback ---
        self.detector = detector
        self.last_face = None     # (angle_rad, distance_m, timestamp)
        self._detect_count = 0
        self._found_count = 0
        self._last_method = ''
        self.last_area_frac = 0.0
        self._last_vert_angle = 0.0  # vertical angle for 3D tracking

    def detect(self, rgb_frame, depth_frame):
        """Detect largest face/person and estimate polar coordinates.

        Detection cascade:
          1. DNN face detector (best for real faces, works in varied lighting)
          2. Haar cascade (works for frontal faces in good lighting)
          3. MobileNet-SSD 'person' (detects full body — good when face is obscured)

        Returns:
            (found, angle_rad, distance_m)
        """
        import cv2

        if rgb_frame is None:
            return False, 0.0, 0.0

        self._detect_count += 1
        h_rgb, w_rgb = rgb_frame.shape[:2]
        best_cx, best_cy, best_area = None, None, 0
        method = ''

        # --- 1. DNN face detector (most reliable) ---
        if best_cx is None and self.dnn_net is not None:
            blob = cv2.dnn.blobFromImage(rgb_frame, 1.0, (300, 300),
                                          (104.0, 177.0, 123.0))
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > self.FACE_DNN_CONFIDENCE:
                    box = detections[0, 0, i, 3:7]
                    x1 = max(0, int(box[0] * w_rgb))
                    y1 = max(0, int(box[1] * h_rgb))
                    x2 = min(w_rgb, int(box[2] * w_rgb))
                    y2 = min(h_rgb, int(box[3] * h_rgb))
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_cx = (x1 + x2) / 2.0
                        best_cy = (y1 + y2) / 2.0
                        method = f'DNN({conf:.2f})'

        # --- 2. Haar cascade (fast, frontal faces only) ---
        if best_cx is None and self.cascade is not None:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            # Equalize histogram for better detection in dark conditions
            gray = cv2.equalizeHist(gray)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                   minNeighbors=3, minSize=(20, 20))
            for (x, y, w, fh) in faces:
                area = w * fh
                if area > best_area:
                    best_area = area
                    best_cx = x + w / 2.0
                    best_cy = y + fh / 2.0
                    method = 'Haar'

        # --- 3. MobileNet-SSD 'person' detection (full body) ---
        if best_cx is None and self.detector is not None:
            detections = self.detector.detect(rgb_frame)
            for det in detections:
                cls = det['class']
                conf = det['confidence']
                if cls == 'person' and conf > self.PERSON_CONFIDENCE:
                    bbox = det['bbox']
                    bw = (bbox[2] - bbox[0]) * w_rgb
                    bh = (bbox[3] - bbox[1]) * h_rgb
                    area = bw * bh
                    if area > best_area:
                        best_area = area
                        best_cx = (bbox[0] + bbox[2]) / 2.0 * w_rgb
                        # Aim at upper third of person bbox (where face likely is)
                        best_cy = (bbox[1] * 0.7 + bbox[3] * 0.3) * h_rgb
                        method = f'Person({conf:.2f})'

        if best_cx is None:
            return False, 0.0, 0.0

        self._found_count += 1
        self._last_method = method

        # --- Compute horizontal + vertical angle from frame center ---
        center_x_norm = best_cx / w_rgb  # [0, 1]
        center_y_norm = best_cy / h_rgb  # [0, 1]
        angle_rad = (center_x_norm - 0.5) * FACE_HFOV_RAD   # horizontal
        vert_angle_rad = -(center_y_norm - 0.5) * FACE_VFOV_RAD  # vertical (neg = up in frame)

        # Store for 3D tracking (gimbal offset added by FaceNavigationSM)
        self._last_vert_angle = vert_angle_rad
        self.last_area_frac = best_area / (h_rgb * w_rgb)

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
        self._detect_count = 0
        self._found_count = 0
        self._last_method = ''
        self.last_area_frac = 0.0
        self._last_vert_angle = 0.0

    def detect(self, rgb_frame, depth_frame):
        return False, 0.0, 0.0


# ---------------------------------------------------------------------------
# Red Target Tracker (HSV color detection — very robust)
# ---------------------------------------------------------------------------

class RedTargetTracker:
    """Detects a red flag/square using HSV color thresholding + ML confirmation.

    Two-stage detection:
      1. HSV color masking (fast, finds red regions)
      2. Shape analysis (aspect ratio, solidity check — is it flag-shaped?)

    Much more reliable than face detection — works at any angle, distance,
    lighting condition. HSV handles the heavy lifting, shape analysis
    filters false positives (red shirt, red wall, etc.).
    """

    def __init__(self):
        self.last_face = None  # (angle_rad, distance_m, timestamp)
        self._detect_count = 0
        self._found_count = 0
        self._last_method = 'Red-HSV'
        # Confidence boost from consecutive detections
        self._consecutive = 0
        self._last_cx = None
        self._last_cy = None
        self.last_area_frac = 0.0
        self._consecutive_area = 0  # consecutive frames with area > threshold

    def detect(self, rgb_frame, depth_frame):
        """Detect largest red region and estimate polar coordinates.

        Returns: (found, angle_rad, distance_m) — same interface as FaceTracker.
        """
        import cv2

        if rgb_frame is None:
            return False, 0.0, 0.0

        self._detect_count += 1
        h_rgb, w_rgb = rgb_frame.shape[:2]

        # Convert to HSV for robust color detection
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)

        # Red wraps around 0° in HSV — need two ranges
        mask1 = cv2.inRange(hsv, RED_HSV_LOWER1, RED_HSV_UPPER1)
        mask2 = cv2.inRange(hsv, RED_HSV_LOWER2, RED_HSV_UPPER2)
        red_mask = mask1 | mask2

        # Morphological cleanup — remove noise, fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._consecutive = 0
            self._consecutive_area = 0
            return False, 0.0, 0.0

        # --- Garble detection: if red pixels are spread everywhere, frame is bad ---
        # A garbled frame has red artifacts scattered across the whole image.
        # A real red target is LOCALIZED (single region, not everywhere).
        total_red_pixels = np.count_nonzero(red_mask)
        frame_area = h_rgb * w_rgb
        total_red_frac = total_red_pixels / frame_area

        # If >30% of entire frame is "red", it's garbled/noise, not a real target
        if total_red_frac > 0.30:
            self._consecutive = 0
            self._consecutive_area = 0
            print(f"  [RED] Garble detected: {total_red_frac:.1%} of frame is red — skipping")
            return False, 0.0, 0.0

        # Also check if red is spread across too many separate regions (garble = many blobs)
        if len(contours) > 15:
            self._consecutive = 0
            self._consecutive_area = 0
            print(f"  [RED] Too many red blobs ({len(contours)}) — likely garbled frame")
            return False, 0.0, 0.0

        # Find largest contour that meets minimum area + shape checks
        min_area = frame_area * RED_MIN_AREA_FRAC
        best_contour = None
        best_area = 0
        best_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            # Shape analysis: prefer compact, roughly rectangular shapes
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area > 0 else 0
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)

            # Reject contours that span nearly the entire frame (garble artifact)
            if bw > w_rgb * 0.7 and bh > h_rgb * 0.7:
                continue  # real target won't fill 70%+ of both dimensions

            # Score: bigger area + compact shape + reasonable aspect ratio
            shape_score = solidity * (1.0 if aspect < 4 else 0.5)
            total_score = area * shape_score

            if total_score > best_score:
                best_score = total_score
                best_area = area
                best_contour = c

        if best_contour is None:
            self._consecutive = 0
            self._consecutive_area = 0
            return False, 0.0, 0.0

        self._found_count += 1
        self._consecutive += 1

        # Compute centroid
        M = cv2.moments(best_contour)
        if M['m00'] == 0:
            return False, 0.0, 0.0
        best_cx = M['m10'] / M['m00']
        best_cy = M['m01'] / M['m00']

        # Consistency check: if centroid jumped far from last frame, lower confidence
        if self._last_cx is not None:
            jump = math.sqrt((best_cx - self._last_cx)**2 + (best_cy - self._last_cy)**2)
            if jump > w_rgb * 0.3:
                self._consecutive = 1  # reset streak on big jump
        self._last_cx = best_cx
        self._last_cy = best_cy

        area_frac = best_area / frame_area
        self._last_method = f'Red-HSV({area_frac:.1%},str={self._consecutive})'
        self.last_area_frac = area_frac  # expose for proximity check

        # Track consecutive frames with high area (for robust ARRIVED)
        if area_frac > 0.12:
            self._consecutive_area += 1
        else:
            self._consecutive_area = 0

        # Compute horizontal + vertical angle from frame center
        center_x_norm = best_cx / w_rgb
        center_y_norm = best_cy / h_rgb
        angle_rad = (center_x_norm - 0.5) * FACE_HFOV_RAD       # horizontal
        vert_angle_rad = -(center_y_norm - 0.5) * FACE_VFOV_RAD  # vertical (neg = up)
        self._last_vert_angle = vert_angle_rad  # expose for 3D tracking

        # Compute distance from depth frame — BUT depth is unreliable at close range
        # (stereo camera fails below ~25cm). Use area fraction as a backup proxy.
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
            distance_m = 5.0

        # At close range, depth sensor gives garbage (jumps to 5-8m).
        # Use area fraction as a more reliable distance proxy:
        # ~1% area ≈ 2m, ~5% ≈ 0.8m, ~10% ≈ 0.4m, ~20% ≈ 0.2m, ~40% ≈ 0.1m
        if area_frac > 0.03 and distance_m > 2.0:
            # Depth is clearly wrong — object fills 3%+ of frame but depth says >2m
            # Estimate distance from area instead
            distance_m = max(0.08, 0.5 / math.sqrt(area_frac / 0.03))
            self._last_method += '(area-est)'

        self.last_face = (angle_rad, distance_m, time.time())
        return True, angle_rad, distance_m


# ---------------------------------------------------------------------------
# Face Navigation State Machine
# ---------------------------------------------------------------------------

class FaceNavigationSM:
    """Navigate toward a detected face/target using DIRECT proportional control.

    Instead of relying on the RL model to interpret an encoded state vector
    (which it wasn't trained for), this SM uses proportional control:
    - Steer toward the target using angle from camera center
    - Drive forward at a speed proportional to distance
    - The RL model is only used during SEARCH_DRIVE for exploration

    States:
        SEARCH_SPIN  → slow rotation scanning for target
        SEARCH_DRIVE → RL drives forward exploring, checking for target
        NAVIGATE     → target found, DIRECT proportional steering toward it
        LOST_HOLD    → target just lost, drive toward estimated world position
        LOST_SPIN    → target lost for a while, spin to re-find
        ARRIVED      → within stop distance, mission complete

    Key improvement: When target is found, we estimate its WORLD POSITION
    (using odometry + angle + depth). If we lose the target, we drive toward
    that estimated position rather than immediately spinning.
    """

    def __init__(self, face_tracker, odom, stop_distance=FACE_STOP_DISTANCE,
                 max_speed=0.15, gimbal_base=None):
        self.tracker = face_tracker
        self.odom = odom
        self.stop_distance = stop_distance
        self.max_speed = max_speed
        self.gimbal_base = gimbal_base  # for active gimbal tracking
        self.state = 'SEARCH_SPIN'
        self.reached = False
        self._steps_since_recheck = 0
        self._navigate_steps = 0
        # Search state
        self._search_dir = 1
        self._phase_start = time.time()
        self._search_count = 0
        self._explored_headings = []      # track which headings we've looked at
        # Lost state
        self._lost_time = None
        # 3D target world position estimate (x, y, z)
        self._target_world_x = None
        self._target_world_y = None
        self._target_world_z = None       # height above rover's ground plane
        self._last_angle = 0.0
        self._last_vert_angle = 0.0       # vertical angle to target
        self._last_distance = 5.0
        self._consecutive_found = 0       # consecutive detections (for confidence)
        # Gimbal tracking state
        self._gimbal_pan = 0              # current gimbal pan angle (degrees)
        self._gimbal_tilt = 0             # current gimbal tilt angle (degrees)
        # Random exploration waypoint for SEARCH_DRIVE
        self._explore_state = self._random_explore_state()

    @staticmethod
    def _random_explore_state():
        angle = np.random.uniform(-0.5, 0.5)
        return np.array([
            math.cos(angle), math.sin(angle), 0.0, 1.0
        ], dtype=np.float32)

    def _get_gimbal_compensated_angles(self, frame_h_angle, frame_v_angle):
        """Adjust detection angles by current gimbal pan/tilt offset.

        The camera frame angles are relative to where the gimbal is pointing.
        Add gimbal offsets to get angles relative to the rover's forward axis.
        """
        # Gimbal pan/tilt are in degrees, convert to radians
        total_h_angle = frame_h_angle + math.radians(self._gimbal_pan)
        total_v_angle = frame_v_angle + math.radians(self._gimbal_tilt)
        return total_h_angle, total_v_angle

    def _update_target_world_pos(self, angle_rad, distance_m):
        """Estimate target's 3D world position using odometry + detection + gimbal.

        Uses the vertical angle (from camera + gimbal tilt) to compute the
        target's height above the ground plane, and the horizontal distance
        for the 2D position on the ground.
        """
        # Get vertical angle (from frame detection + gimbal tilt)
        vert_angle = getattr(self.tracker, '_last_vert_angle', 0.0)
        _, total_vert = self._get_gimbal_compensated_angles(angle_rad, vert_angle)
        total_h, _ = self._get_gimbal_compensated_angles(angle_rad, vert_angle)

        # 3D decomposition: distance is slant range from camera to target
        # horizontal_dist = distance * cos(vertical_angle)  (ground plane distance)
        # height = distance * sin(vertical_angle)  (above/below camera level)
        horizontal_dist = distance_m * math.cos(total_vert)
        target_height = distance_m * math.sin(total_vert)

        # 2D world position (ground plane)
        world_angle = self.odom.heading + total_h
        self._target_world_x = self.odom.x + horizontal_dist * math.cos(world_angle)
        self._target_world_y = self.odom.y + horizontal_dist * math.sin(world_angle)
        # Z = height relative to rover camera (positive = above camera)
        self._target_world_z = target_height

    def _angle_to_target_world(self):
        """Compute horizontal angle from rover to estimated target world position."""
        if self._target_world_x is None:
            return 0.0
        dx = self._target_world_x - self.odom.x
        dy = self._target_world_y - self.odom.y
        target_angle = math.atan2(dy, dx)
        error = target_angle - self.odom.heading
        return (error + math.pi) % (2 * math.pi) - math.pi

    def _dist_to_target_world(self):
        """Ground-plane distance from rover to estimated target world position."""
        if self._target_world_x is None:
            return 999.0
        dx = self._target_world_x - self.odom.x
        dy = self._target_world_y - self.odom.y
        return math.sqrt(dx**2 + dy**2)

    def _vert_angle_to_target_world(self):
        """Vertical angle to target from current position (for gimbal aiming)."""
        if self._target_world_z is None:
            return 0.0
        ground_dist = self._dist_to_target_world()
        if ground_dist < 0.1:
            return 0.0
        return math.atan2(self._target_world_z, ground_dist)

    def _update_gimbal_tracking(self, h_angle_rad, v_angle_rad):
        """Actively point gimbal toward target for better tracking.

        During NAVIGATE: tilt gimbal to keep target centered vertically.
        During SEARCH: gimbal does its own scanning (controlled by GimbalScanner).
        """
        if self.gimbal_base is None:
            return

        if self.state == 'NAVIGATE':
            # Compute desired tilt to keep target vertically centered
            # Positive v_angle = target is above camera → tilt up (positive tilt)
            desired_tilt = int(math.degrees(v_angle_rad))
            desired_tilt = max(-20, min(25, desired_tilt))  # clamp for safety

            # Only update gimbal if tilt changed significantly (avoid servo jitter)
            if abs(desired_tilt - self._gimbal_tilt) >= 3:
                self._gimbal_tilt = desired_tilt
                # Pan stays at 0 during NAVIGATE — rover turns its body to aim
                self._gimbal_pan = 0
                self.gimbal_base.gimbal_ctrl(
                    self._gimbal_pan, self._gimbal_tilt,
                    GIMBAL_SCAN_SPEED, GIMBAL_SCAN_ACCEL)

        elif self.state in ('LOST_HOLD',):
            # When target lost, tilt toward estimated world position
            vert = self._vert_angle_to_target_world()
            desired_tilt = int(math.degrees(vert))
            desired_tilt = max(-20, min(25, desired_tilt))
            if abs(desired_tilt - self._gimbal_tilt) >= 3:
                self._gimbal_tilt = desired_tilt
                self._gimbal_pan = 0
                self.gimbal_base.gimbal_ctrl(
                    0, self._gimbal_tilt,
                    GIMBAL_SCAN_SPEED, GIMBAL_SCAN_ACCEL)

    def reset_gimbal(self):
        """Return gimbal to center (called when leaving NAVIGATE state)."""
        if self.gimbal_base is not None:
            self._gimbal_pan = 0
            self._gimbal_tilt = 0
            self.gimbal_base.gimbal_ctrl(0, 0, 0, 0)

    def _proportional_steer(self, angle_rad, distance_m):
        """Direct proportional control: steer toward target and drive forward.

        This is MUCH more reliable than passing an encoded state to the RL model,
        because the RL model was trained for waypoint nav, not face following.
        """
        # Angular: proportional to angle offset
        # Positive angle = target is to the right of center
        # Negative angular velocity = turn right
        angular = -FACE_APPROACH_GAIN * angle_rad
        angular = max(-FACE_APPROACH_MAX_ANG, min(FACE_APPROACH_MAX_ANG, angular))

        # Linear: scale with distance, slow down as we approach
        if distance_m > self.stop_distance * 2:
            linear = self.max_speed * 0.8
        elif distance_m > self.stop_distance:
            frac = (distance_m - self.stop_distance) / self.stop_distance
            linear = self.max_speed * max(0.2, min(0.8, frac * 0.6))
        else:
            linear = 0.0

        # If very off-angle, slow down and focus on turning
        if abs(angle_rad) > math.radians(25):
            linear *= 0.3

        return linear, angular

    def update(self, rgb_frame, depth_frame):
        """Compute face navigation output.

        Returns:
            (override, state_vector)
            override: (linear, angular) tuple when SM controls rover, or None when RL drives
            state_vector: 4-element numpy array for RL model observation
        """
        found, angle, distance = self.tracker.detect(rgb_frame, depth_frame)
        now = time.time()

        # Get vertical angle from tracker (for 3D positioning + gimbal)
        raw_vert_angle = getattr(self.tracker, '_last_vert_angle', 0.0)

        # Compensate detection angles for current gimbal position
        if found:
            comp_h, comp_v = self._get_gimbal_compensated_angles(angle, raw_vert_angle)
        else:
            comp_h, comp_v = angle, raw_vert_angle

        # --- Check area-based ARRIVED (works even when depth is garbage) ---
        # Require 3+ CONSECUTIVE high-area frames to avoid false triggers from garbled frames
        area_frac = getattr(self.tracker, 'last_area_frac', 0.0)
        consec_area = getattr(self.tracker, '_consecutive_area', 0)
        if (found and area_frac > 0.15 and consec_area >= 3
                and self._navigate_steps >= FACE_MIN_NAVIGATE_STEPS):
            self.state = 'ARRIVED'
            self.reached = True
            self.reset_gimbal()
            z_str = f" z={self._target_world_z:.2f}m" if self._target_world_z else ""
            print(f"  [FACE] ARRIVED via area! area={area_frac:.1%} consec={consec_area} "
                  f"dist={distance:.2f}m{z_str} after {self._navigate_steps} steps")
            return (0.0, 0.0), DEFAULT_STATE.copy()

        # --- Any state: if target found, go to NAVIGATE ---
        if found and self.state not in ('NAVIGATE', 'ARRIVED'):
            self.state = 'NAVIGATE'
            if self._navigate_steps == 0:
                self._navigate_steps = 1
            self._steps_since_recheck = 0
            self._last_angle = comp_h   # gimbal-compensated horizontal
            self._last_vert_angle = comp_v
            self._last_distance = distance
            self._consecutive_found = 1
            self._update_target_world_pos(comp_h, distance)
            # Start active gimbal tracking
            self._update_gimbal_tracking(comp_h, comp_v)
            linear, angular = self._proportional_steer(comp_h, distance)
            z_str = f" z={self._target_world_z:.2f}m" if self._target_world_z else ""
            print(f"  [FACE] Found! h={math.degrees(comp_h):.1f}° v={math.degrees(comp_v):.1f}° "
                  f"dist={distance:.2f}m{z_str} area={area_frac:.1%} → NAVIGATE (3D)")
            return (linear, angular), DEFAULT_STATE.copy()

        # --- SEARCH_SPIN: slow rotation scanning ---
        if self.state == 'SEARCH_SPIN':
            elapsed = now - self._phase_start
            if elapsed > 0 and int(elapsed * 2) > len(self._explored_headings):
                self._explored_headings.append(math.degrees(self.odom.heading) % 360)
            if elapsed >= FACE_SEARCH_SPIN_DURATION:
                self.state = 'SEARCH_DRIVE'
                self._phase_start = now
                self._explore_state = self._random_explore_state()
                self._search_count += 1
                print(f"  [FACE] Spin done, exploring (cycle #{self._search_count}, "
                      f"explored {len(self._explored_headings)} directions)")
                return None, self._explore_state
            return (0.0, self._search_dir * FACE_SEARCH_SPIN_SPEED), DEFAULT_STATE.copy()

        # --- SEARCH_DRIVE: RL drives forward exploring ---
        if self.state == 'SEARCH_DRIVE':
            elapsed = now - self._phase_start
            if elapsed >= FACE_SEARCH_DRIVE_DURATION:
                self.state = 'SEARCH_SPIN'
                self._phase_start = now
                self._search_dir *= -1
                print(f"  [FACE] Explore done, spinning again")
                return (0.0, self._search_dir * FACE_SEARCH_SPIN_SPEED), DEFAULT_STATE.copy()
            return None, self._explore_state

        # --- NAVIGATE: DIRECT proportional steering toward target ---
        if self.state == 'NAVIGATE':
            self._navigate_steps += 1
            self._steps_since_recheck += 1

            if self._steps_since_recheck >= FACE_RECHECK_STEPS:
                self._steps_since_recheck = 0
                if found:
                    self._consecutive_found += 1
                    self._last_angle = comp_h
                    self._last_vert_angle = comp_v
                    self._last_distance = distance
                    self._update_target_world_pos(comp_h, distance)
                    # Active gimbal tracking — tilt toward target
                    self._update_gimbal_tracking(comp_h, comp_v)

                    # ARRIVED: depth-based OR area-based
                    if self._navigate_steps >= FACE_MIN_NAVIGATE_STEPS:
                        if distance < self.stop_distance or (area_frac > 0.15 and consec_area >= 3):
                            self.state = 'ARRIVED'
                            self.reached = True
                            self.reset_gimbal()
                            z_str = f" z={self._target_world_z:.2f}m" if self._target_world_z else ""
                            print(f"  [FACE] ARRIVED! dist={distance:.2f}m area={area_frac:.1%}"
                                  f"{z_str} after {self._navigate_steps} steps")
                            return (0.0, 0.0), DEFAULT_STATE.copy()

                    linear, angular = self._proportional_steer(comp_h, distance)
                    z_str = f" z={self._target_world_z:.2f}m" if self._target_world_z else ""
                    print(f"  [FACE] Tracking: h={math.degrees(comp_h):.1f}° v={math.degrees(comp_v):.1f}° "
                          f"dist={distance:.2f}m{z_str} area={area_frac:.1%} "
                          f"(step {self._navigate_steps}, streak={self._consecutive_found})")
                    return (linear, angular), DEFAULT_STATE.copy()
                else:
                    self._consecutive_found = 0
                    self.state = 'LOST_HOLD'
                    self._lost_time = now
                    # Update gimbal to point toward estimated position
                    self._update_gimbal_tracking(
                        self._angle_to_target_world(),
                        self._vert_angle_to_target_world())
                    z_str = f" z={self._target_world_z:.2f}m" if self._target_world_z else ""
                    print(f"  [FACE] Lost target → LOST_HOLD (3D est: "
                          f"dist={self._dist_to_target_world():.1f}m{z_str})")

            # Between rechecks: use proportional steer from last known angle/distance
            linear, angular = self._proportional_steer(self._last_angle, self._last_distance)
            return (linear, angular), DEFAULT_STATE.copy()

        # --- LOST_HOLD: drive toward estimated 3D world position ---
        if self.state == 'LOST_HOLD':
            if found:
                self.state = 'NAVIGATE'
                self._steps_since_recheck = 0
                self._consecutive_found = 1
                self._last_angle = comp_h
                self._last_vert_angle = comp_v
                self._last_distance = distance
                self._update_target_world_pos(comp_h, distance)
                self._update_gimbal_tracking(comp_h, comp_v)
                linear, angular = self._proportional_steer(comp_h, distance)
                print(f"  [FACE] Re-acquired! h={math.degrees(comp_h):.1f}° v={math.degrees(comp_v):.1f}° "
                      f"dist={distance:.2f}m area={area_frac:.1%} (cumulative step {self._navigate_steps})")
                return (linear, angular), DEFAULT_STATE.copy()

            if now - self._lost_time > FACE_LOST_HOLD_TIME:
                self.state = 'LOST_SPIN'
                self._phase_start = now
                self.reset_gimbal()  # center gimbal for search
                print("  [FACE] Still lost → LOST_SPIN")
                return (0.0, self._search_dir * FACE_SEARCH_SPIN_SPEED), DEFAULT_STATE.copy()

            # Drive toward estimated world position using odometry
            angle_to_target = self._angle_to_target_world()
            dist_to_target = self._dist_to_target_world()
            if dist_to_target > 0.3:
                lin = self.max_speed * 0.5
                ang = -FACE_APPROACH_GAIN * angle_to_target
                ang = max(-FACE_APPROACH_MAX_ANG, min(FACE_APPROACH_MAX_ANG, ang))
                return (lin, ang), DEFAULT_STATE.copy()
            else:
                self.state = 'LOST_SPIN'
                self._phase_start = now
                return (0.0, self._search_dir * FACE_SEARCH_SPIN_SPEED), DEFAULT_STATE.copy()

        # --- LOST_SPIN: spin in place trying to re-find ---
        if self.state == 'LOST_SPIN':
            if found:
                self.state = 'NAVIGATE'
                # DON'T reset _navigate_steps — keep accumulating
                self._steps_since_recheck = 0
                self._consecutive_found = 1
                self._last_angle = comp_h
                self._last_vert_angle = comp_v
                self._last_distance = distance
                self._update_target_world_pos(comp_h, distance)
                self._update_gimbal_tracking(comp_h, comp_v)
                linear, angular = self._proportional_steer(comp_h, distance)
                print(f"  [FACE] Re-acquired during spin! h={math.degrees(comp_h):.1f}° "
                      f"v={math.degrees(comp_v):.1f}° dist={distance:.2f}m")
                return (linear, angular), DEFAULT_STATE.copy()
            if now - self._phase_start > FACE_LOST_SPIN_TIME:
                self.state = 'SEARCH_SPIN'
                self._phase_start = now
                self._search_dir *= -1
                print("  [FACE] Lost spin failed → full SEARCH_SPIN")
                return (0.0, self._search_dir * FACE_SEARCH_SPIN_SPEED), DEFAULT_STATE.copy()
            return (0.0, self._search_dir * FACE_SEARCH_SPIN_SPEED), DEFAULT_STATE.copy()

        # --- ARRIVED ---
        if self.state == 'ARRIVED':
            return (0.0, 0.0), DEFAULT_STATE.copy()

        return None, DEFAULT_STATE.copy()


def play_arrival_sound():
    """Play the iconic announcement when the rover reaches its target.
    Uses espeak (text-to-speech) which is available on Jetson Ubuntu."""
    import subprocess

    # "Ladies and gentlemen, we got 'em" — the iconic phrase
    phrase = "Ladies and gentlemen... we got em."

    try:
        # Max volume (-a 200), deep voice, slow dramatic pace
        subprocess.Popen(
            ['espeak', '-a', '200', '-v', 'en+m1', '-s', '110', '-p', '15', phrase],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f'  [SOUND] "{phrase}"')
    except Exception as e:
        print(f"  [SOUND] Could not play arrival sound: {e}")
    # Also try amixer to max out system volume
    try:
        subprocess.Popen(
            ['amixer', 'set', 'Master', '100%'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


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
        self._last_mask = None  # store last segmentation mask for visualization
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
        mask = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (480, 720)
        self._last_mask = mask

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

    Normal mode: center → right → center → left → center → ...
    Face mode:   wider sweep with vertical tilt cycling (up/level/down)
    Camera MUST be centered when depth is captured for the RL model.
    """

    def __init__(self, base, face_mode=False):
        self.base = base
        self.face_mode = face_mode
        self._scan_step = 0           # counts control steps
        self._current_target = 0      # current pan angle target
        self._scan_phase = 0          # 0=center, 1=right, 2=center, 3=left
        self._centering_countdown = 0 # steps remaining before camera is settled
        self._last_scan_depth = {}    # {'left': depth, 'right': depth}
        self._tilt_cycle = 0          # face mode: cycles through tilt angles
        self._tilt_angles = [GIMBAL_TILT_DEFAULT, GIMBAL_TILT_UP, GIMBAL_TILT_DOWN]

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

        # Get current tilt angle (cycle through up/level/down in face mode)
        if self.face_mode:
            tilt = self._tilt_angles[self._tilt_cycle % len(self._tilt_angles)]
        else:
            tilt = GIMBAL_TILT_DEFAULT

        # Scan cycle: center(0) → right(1) → center(2) → left(3) → center(0)
        self._scan_phase = (self._scan_phase + 1) % 4

        if self._scan_phase == 1:
            self._current_target = -GIMBAL_SCAN_ANGLE
            self.base.gimbal_ctrl(-GIMBAL_SCAN_ANGLE, tilt,
                                  GIMBAL_SCAN_SPEED, GIMBAL_SCAN_ACCEL)
            return 'scan_right'
        elif self._scan_phase == 3:
            self._current_target = GIMBAL_SCAN_ANGLE
            self.base.gimbal_ctrl(GIMBAL_SCAN_ANGLE, tilt,
                                  GIMBAL_SCAN_SPEED, GIMBAL_SCAN_ACCEL)
            return 'scan_left'
        else:
            self._current_target = 0
            self._centering_countdown = GIMBAL_SETTLE_STEPS
            # Advance tilt cycle each time we return to center
            if self.face_mode:
                self._tilt_cycle += 1
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
                    video_out = cam_rgb.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i)
                    self.rgb_queue = video_out.createOutputQueue()
                    print("[OK] RGB camera enabled (BGR888i interleaved)")
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
            # Use get() with timeout for a complete, fresh frame.
            # tryGet() can return stale/partial frames causing garbled output.
            data = self.rgb_queue.get(timeout=0.5)
            if data is not None:
                frame = data.getCvFrame()
                if frame is not None:
                    # .copy() is critical — depthai recycles underlying buffers
                    frame = frame.copy()
                    # Basic garble check: a valid BGR frame should have reasonable
                    # channel statistics. Garbled frames often have extreme values.
                    if frame.shape[2] == 3:
                        mean_b, mean_g, mean_r = frame.mean(axis=(0, 1))
                        # Garbled frames often have extreme pink/green — one channel
                        # is wildly different from others
                        channel_spread = max(mean_b, mean_g, mean_r) - min(mean_b, mean_g, mean_r)
                        if channel_spread > 100:
                            return None  # likely garbled, skip this frame
                    return frame
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
                 log_rewards=False, reward_log_path=None,
                 frame_dir=None):
        self.max_speed = min(abs(max_speed), MAX_LINEAR_SPEED)
        self.dry_run = dry_run
        self.running = False
        self.base = None
        self.target_mode = target_mode
        self.frame_dir = frame_dir  # if set, writes camera frames for web UI

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
        # Face/red mode needs RGB even if detection is off
        needs_rgb = enable_detect or target_mode in ('face', 'red') or enable_segmentation
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
            self.gimbal = GimbalScanner(base_for_peripherals, face_mode=(target_mode == 'face'))
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

        # --- Initialize face/red target navigation ---
        self.face_tracker = None
        self.face_nav = None
        if target_mode in ('face', 'red'):
            if demo:
                self.face_tracker = DummyFaceTracker()
            elif target_mode == 'red':
                self.face_tracker = RedTargetTracker()
                print(f"[OK] Red target mode — HSV + MobileNet SSD detection")
            else:
                self.face_tracker = FaceTracker(detector=self.detector)
            self.face_nav = FaceNavigationSM(
                self.face_tracker, self.odom,
                stop_distance=face_stop_distance if target_mode == 'face' else RED_STOP_DISTANCE,
                max_speed=self.max_speed,
                gimbal_base=self.base if not self.dry_run else None,
            )
            print(f"[OK] Target navigation mode: {target_mode} (stop at "
                  f"{self.face_nav.stop_distance:.1f}m)")

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
        """Run object detection on RGB frame. Returns (detected, class, center_x, conf).

        IMPORTANT: In face/red target mode, 'person' is EXCLUDED from obstacle classes
        because the rover is trying to navigate TOWARD a person/target, not away from them.
        """
        rgb = self.camera.get_rgb_frame()
        if rgb is None:
            return False, "", 0.5, 0.0

        detections = self.detector.detect(rgb)

        # In face/red mode, exclude 'person' — we're navigating TO them, not avoiding!
        skip_classes = set()
        if self.target_mode in ('face', 'red'):
            skip_classes.add('person')

        best = None
        best_danger = 0
        for det in detections:
            cls = det['class']
            if cls in skip_classes:
                continue
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
                # When navigating toward a face/target, suppress backup and most avoidance
                # since the "obstacle" IS the target we're trying to reach
                _face_approaching = (self.face_nav is not None
                                     and self.face_nav.state in ('NAVIGATE', 'LOST_HOLD')
                                     and self.face_nav._navigate_steps >= 3)

                # Tier 1: EMERGENCY STOP — object at ≤5cm or stereo dead zone
                # In face/red mode approaching target: consider this ARRIVED instead of backing up
                if min_depth <= EMERGENCY_STOP_DIST and _face_approaching:
                    self._stop_rover()
                    if self.face_nav._last_distance < self.face_nav.stop_distance * 1.5:
                        self.face_nav.state = 'ARRIVED'
                        self.face_nav.reached = True
                        print(f"  [FACE] ARRIVED via proximity! depth={min_depth:.2f}m")
                        break
                    # Not close enough to declared stop distance — just pause
                    print(f"  [FACE] Emergency proximity but target far — pausing")
                    step += 1
                    time.sleep(interval)
                    continue

                if min_depth <= EMERGENCY_STOP_DIST and not _face_approaching:
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
                # Suppressed when approaching face target
                if min_depth < BACKUP_DIST and not _face_approaching:
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

                # --- 5c. Face/target navigation (before RL inference) ---
                face_override = None
                state_vector = DEFAULT_STATE.copy()
                if self.face_nav is not None:
                    face_rgb = self.camera.get_rgb_frame()
                    face_override, state_vector = self.face_nav.update(face_rgb, depth_frame)
                    if self.face_nav.reached:
                        print(f"\n[NAV] TARGET REACHED!")
                        if self.face_nav.state == 'ARRIVED':
                            r, e = self.reward_logger.compute_reward(
                                0, 0, min_depth, 'ARRIVED', face_state='ARRIVED', reached_goal=True)
                            self.reward_logger.log(
                                step, r, e, 0, 0, min_depth, left_depth, right_depth,
                                'ARRIVED', 'ARRIVED', seg_left, seg_right,
                                self.odom.total_distance, self.odom.displacement,
                                math.degrees(self.odom.heading))
                            if not self.dry_run:
                                play_arrival_sound()
                        break

                # --- 6. Compute avoidance ---
                # Skip avoidance when face/target navigation is actively steering
                # (the "obstacle" IS the target — person, flag, etc.)
                if face_override is not None and _face_approaching:
                    override, mode = None, ""
                else:
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

                # --- 8. Apply overrides (priority: face/target > avoidance > RL) ---
                # Face override takes HIGHEST priority — it uses direct proportional
                # control which is more reliable than RL for target following
                if face_override is not None:
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

                # --- 10b. Write frames for web UI (EVERY step for smooth feeds) ---
                if self.frame_dir:
                    try:
                        import cv2
                        _rgb = self.camera.get_rgb_frame()
                        if _rgb is not None:
                            cv2.imwrite(os.path.join(self.frame_dir, 'rgb.jpg'), _rgb,
                                        [cv2.IMWRITE_JPEG_QUALITY, 40])
                        # Depth heatmap — always available since we just captured it
                        _dv = np.clip(depth_frame / 5.0, 0, 1)
                        _dv = (_dv * 255).astype(np.uint8)
                        _dc = cv2.applyColorMap(_dv, cv2.COLORMAP_JET)
                        _dc = cv2.resize(_dc, (640, 480))
                        cv2.imwrite(os.path.join(self.frame_dir, 'depth.jpg'), _dc,
                                    [cv2.IMWRITE_JPEG_QUALITY, 40])
                        # Segmentation overlay — write when available
                        if (hasattr(self, 'segmenter') and self.segmenter
                                and hasattr(self.segmenter, '_last_mask')
                                and self.segmenter._last_mask is not None):
                            _seg_colors = np.array([
                                [50, 50, 50], [255, 80, 80],
                                [0, 200, 0], [0, 0, 255]], dtype=np.uint8)
                            _mask = self.segmenter._last_mask
                            _seg = _seg_colors[_mask]
                            _seg_rgb = _rgb if _rgb is not None else np.zeros(
                                (_mask.shape[0], _mask.shape[1], 3), dtype=np.uint8)
                            _seg_rgb_sm = cv2.resize(_seg_rgb,
                                                     (_mask.shape[1], _mask.shape[0]))
                            _seg = cv2.addWeighted(_seg_rgb_sm, 0.4, _seg, 0.6, 0)
                            _seg = cv2.resize(_seg, (640, 480))
                            cv2.putText(_seg,
                                        f"Feasibility: L={seg_left:.2f} R={seg_right:.2f}",
                                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 1)
                            cv2.imwrite(os.path.join(self.frame_dir, 'seg.jpg'), _seg,
                                        [cv2.IMWRITE_JPEG_QUALITY, 40])
                    except Exception:
                        pass

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
                        if hasattr(self.face_tracker, '_last_method') and self.face_tracker._last_method:
                            face_str += f"[{self.face_tracker._last_method}]"
                        if hasattr(self.face_tracker, '_detect_count') and self.face_tracker._detect_count > 0:
                            rate = self.face_tracker._found_count / self.face_tracker._detect_count
                            face_str += f" det={self.face_tracker._found_count}/{self.face_tracker._detect_count}({rate:.0%})"

                    # Segmentation info
                    seg_str = ""
                    if not isinstance(self.segmenter, DummySegmenter):
                        seg_str = f" seg=({seg_left:.2f}L/{seg_right:.2f}R)"

                    print(f"  [{elapsed:.0f}s]{mode_str}"
                          f" raw=[{raw_linear:+.3f},{raw_angular:+.3f}]"
                          f" → lin={linear_vel:+.3f} ang={angular_vel:+.3f}"
                          f" depth={min_depth:.2f}m (L={left_depth:.2f} R={right_depth:.2f})"
                          f"{obj_str}{odom_str}{dest_str}{obs_str}{cc_str}{face_str}{seg_str}")

                # --- 12. Gimbal control ---
                # If face nav is actively tracking (NAVIGATE/LOST_HOLD), it controls
                # the gimbal directly — don't let the scanner fight it
                face_tracking_active = (self.face_nav is not None
                                        and self.face_nav.state in ('NAVIGATE', 'LOST_HOLD'))
                if not face_tracking_active:
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
    parser.add_argument('--target', choices=['distance', 'face', 'red'], default='distance',
                        help='Navigation mode: distance, face-tracking, or red-flag target (default: distance)')
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
    parser.add_argument('--frame-dir', type=str, default=None,
                        help='Write camera frames to this dir for web UI streaming')

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
        frame_dir=args.frame_dir,
    )

    # Create frame dir if specified
    if args.frame_dir:
        os.makedirs(args.frame_dir, exist_ok=True)

    controller.run(duration=args.duration, target_distance=args.distance)


if __name__ == '__main__':
    main()
