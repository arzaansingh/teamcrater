#!/usr/bin/env python3
"""
Rover Control — Hardware Abstraction Layer
=============================================
Clean interface for all rover hardware: motors, gimbal, LEDs, camera, telemetry.
Separated from the web UI (mission_control.py) and autonomy (rl_deploy.py) so
changes to one don't affect the others.

Usage:
    from rover_control import RoverHardware, CameraManager
    rover = RoverHardware()
    rover.drive(0.1, 0.0)   # forward at 0.1 m/s
    rover.stop()
"""

import sys
import os
import time
import threading
import numpy as np

# Add ugv_jetson to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ugv_jetson'))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 115200
MAX_LINEAR = 0.3    # m/s hard limit
MAX_ANGULAR = 2.0   # rad/s hard limit (UI default 1.0, max 2.0)


# ---------------------------------------------------------------------------
# RoverHardware — motor, gimbal, LEDs, telemetry, serial lifecycle
# ---------------------------------------------------------------------------

class RoverHardware:
    """Unified hardware interface for the Waveshare UGV Rover PT.

    Wraps BaseController with safe clamping, serial lifecycle management
    for mode transitions, and a best-effort emergency stop.
    """

    def __init__(self, serial_port=SERIAL_PORT, baud=BAUD_RATE):
        self._port = serial_port
        self._baud = baud
        self.base = None
        self._active = False
        try:
            from base_ctrl import BaseController
            self.base = BaseController(serial_port, baud)
            self._active = True
            print(f"[ROVER] Connected to {serial_port}")
        except Exception as e:
            print(f"[ROVER] Hardware unavailable: {e} (manual control disabled)")

    # --- Motor control ---

    def drive(self, linear, angular):
        """Send velocity command. Clamps to safe limits."""
        if not self._active or not self.base:
            return
        linear = max(-MAX_LINEAR, min(MAX_LINEAR, float(linear)))
        angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, float(angular)))
        self.base.base_json_ctrl({"T": 13, "X": round(linear, 3), "Z": round(angular, 3)})

    def stop(self):
        """Stop all movement."""
        self.drive(0.0, 0.0)

    # --- Gimbal ---

    def gimbal_set(self, pan, tilt, speed=60, accel=40):
        """Set gimbal position. pan/tilt in degrees."""
        if not self._active or not self.base:
            return
        self.base.gimbal_ctrl(int(pan), int(tilt), int(speed), int(accel))

    def gimbal_center(self):
        """Center gimbal (pan=0, tilt=0)."""
        self.gimbal_set(0, 0, 0, 0)

    # --- LEDs ---

    def lights_set(self, pwm_a, pwm_b):
        """Set LED brightness. pwm_a=base LEDs, pwm_b=head LEDs. Range 0-255."""
        if not self._active or not self.base:
            return
        self.base.lights_ctrl(max(0, min(255, int(pwm_a))),
                              max(0, min(255, int(pwm_b))))

    # --- Telemetry ---

    def get_telemetry(self):
        """Read IMU/battery/encoder data from ESP32. Returns dict or {}."""
        if not self._active or not self.base:
            return {}
        try:
            data = self.base.feedback_data()
            return data if data else {}
        except Exception:
            return {}

    # --- Serial lifecycle (for mode transitions) ---

    def release_serial(self):
        """Close serial port so rl_deploy.py subprocess can use it."""
        if self.base and self._active:
            try:
                self.base.ser.close()
            except Exception:
                pass
            self._active = False
            print("[ROVER] Serial released for autonomous mode")

    def reacquire_serial(self):
        """Reopen serial port after subprocess exits."""
        try:
            from base_ctrl import BaseController
            self.base = BaseController(self._port, self._baud)
            self._active = True
            print("[ROVER] Serial reacquired")
        except Exception as e:
            print(f"[ROVER] Could not reacquire serial: {e}")
            self._active = False

    @property
    def is_connected(self):
        return self._active and self.base is not None

    # --- E-STOP ---

    def emergency_stop(self):
        """Best-effort stop — reopens serial if needed."""
        if not self._active:
            try:
                self.reacquire_serial()
            except Exception:
                pass
        self.stop()
        # Double-send for reliability
        time.sleep(0.05)
        self.stop()
        print("[ROVER] EMERGENCY STOP sent")


# ---------------------------------------------------------------------------
# DummyRoverHardware — for testing without rover
# ---------------------------------------------------------------------------

class DummyRoverHardware:
    """No-op hardware for local testing."""

    def drive(self, linear, angular):
        pass

    def stop(self):
        pass

    def gimbal_set(self, pan, tilt, speed=60, accel=40):
        pass

    def gimbal_center(self):
        pass

    def lights_set(self, pwm_a, pwm_b):
        pass

    def get_telemetry(self):
        return {}

    def release_serial(self):
        pass

    def reacquire_serial(self):
        pass

    def emergency_stop(self):
        pass

    @property
    def is_connected(self):
        return False


# ---------------------------------------------------------------------------
# CameraManager — RGB via cv2.VideoCapture + depth via depthai
# ---------------------------------------------------------------------------

# JPEG quality for streaming (matches the built-in Waveshare page config.yaml)
VIDEO_QUALITY = 20

class CameraManager:
    """Manages camera feeds with start/stop lifecycle for mode transitions.

    RGB: Uses cv2.VideoCapture (same method as the built-in Waveshare web
    page at port 5000). This is standard USB Video Class — proven reliable,
    no frame degradation over time.

    Depth: Uses depthai stereo-only pipeline (mono cameras only, no RGB
    node) so it doesn't conflict with cv2.VideoCapture's access to the
    RGB sensor.

    Segmentation: Runs on RGB frames via PyTorch (no depthai needed).
    """

    # MobileNet SSD class names (same model as built-in Waveshare page)
    DNN_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor",
    ]

    def __init__(self, enable_seg=False, seg_model_path=None):
        self._enable_seg = enable_seg
        self._seg_model_path = seg_model_path
        self._thread = None
        self._active = False
        self._cap = None          # cv2.VideoCapture for RGB
        self._depth_cam = None    # depthai pipeline for depth only
        self._detect_enabled = False  # DNN object detection toggle

        # Thread-safe frame buffers (pre-encoded JPEG bytes)
        self._lock = threading.Lock()
        self._rgb_jpeg = None
        self._depth_jpeg = None
        self._seg_jpeg = None
        self._telemetry = {
            'fps': 0, 'min_depth': 0, 'left_depth': 0, 'right_depth': 0,
            'detection': 'Initializing...', 'objects': '-',
        }

    def start(self, retries=3):
        """Initialize camera and start capture thread."""
        if self._active:
            return
        for attempt in range(retries):
            try:
                self._start_capture()
                return
            except Exception as e:
                print(f"[CAM] Start attempt {attempt + 1}/{retries} failed: {e}")
                time.sleep(2.0)
        print("[CAM] Could not start camera after retries")

    def stop(self):
        """Release camera for subprocess use. Joins capture thread."""
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._depth_cam:
            try:
                self._depth_cam.close()
            except Exception:
                pass
            self._depth_cam = None
        with self._lock:
            self._rgb_jpeg = None
            self._depth_jpeg = None
            self._seg_jpeg = None
        print("[CAM] Camera released")

    def _start_capture(self):
        """Internal: open cv2.VideoCapture for RGB + optional depthai for depth."""
        import cv2

        # --- RGB via cv2.VideoCapture (same as built-in page) ---
        self._cap = cv2.VideoCapture(-1)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera via cv2.VideoCapture")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[CAM] RGB camera opened (cv2.VideoCapture — same as built-in page)")

        # --- Depth via depthai (stereo only, no RGB node) ---
        try:
            from rl_deploy import DepthCamera
            self._depth_cam = DepthCamera(enable_rgb=False)
            print("[CAM] Depth camera opened (depthai stereo)")
        except Exception as e:
            print(f"[CAM] Depth unavailable: {e} (RGB-only mode)")
            self._depth_cam = None

        time.sleep(0.5)
        self._active = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[CAM] Camera started")

    def _capture_loop(self):
        """Background thread: grab RGB from VideoCapture, depth from depthai.

        Uses the exact same approach as the built-in Waveshare web page
        (ugv_jetson/cv_ctrl.py) for RGB: cv2.VideoCapture + imencode at
        quality 20. No depthai for RGB = no USB bandwidth degradation.

        DNN object detection (MobileNet SSD) is safe here because we use
        cv2.VideoCapture, NOT depthai for RGB. The cv2.dnn + depthai
        conflict only happens when both share the same USB pipeline.
        """
        import cv2

        # --- DNN object detection (MobileNet SSD, same as built-in page) ---
        dnn_net = None
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'ugv_jetson', 'models')
        proto = os.path.join(models_dir, 'deploy.prototxt')
        model = os.path.join(models_dir, 'mobilenet_iter_73000.caffemodel')
        if os.path.exists(proto) and os.path.exists(model):
            try:
                dnn_net = cv2.dnn.readNetFromCaffe(proto, model)
                print("[CAM] MobileNet SSD detection model loaded")
            except Exception as e:
                print(f"[CAM] Detection model failed to load: {e}")
        else:
            print(f"[CAM] Detection model not found at {models_dir}")

        # Segmentation uses PyTorch on RGB frames — runs in a SEPARATE thread
        # because inference takes ~1.7s on Jetson CPU and would block all feeds.
        segmenter = None
        seg_colors = np.array([
            [50, 50, 50], [255, 80, 80], [0, 200, 0], [0, 0, 255]
        ], dtype=np.uint8)

        if self._enable_seg and self._seg_model_path:
            try:
                seg_path = self._seg_model_path
                if not os.path.isabs(seg_path):
                    seg_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), seg_path)
                print(f"[CAM] Loading segmenter from: {seg_path} "
                      f"(exists={os.path.exists(seg_path)})", flush=True)
                from rl_deploy import TerrainSegmenter
                segmenter = TerrainSegmenter(seg_path)
                print("[CAM] Terrain segmenter loaded OK", flush=True)
            except Exception as e:
                import traceback
                print(f"[CAM] Segmenter unavailable: {e}", flush=True)
                traceback.print_exc()
        else:
            print(f"[CAM] Segmentation disabled (enable_seg={self._enable_seg}, "
                  f"model_path={self._seg_model_path})", flush=True)

        # Start seg worker thread — runs inference asynchronously
        seg_input_frame = [None]   # shared: capture loop writes, seg thread reads
        seg_input_lock = threading.Lock()
        if segmenter is not None:
            def _seg_worker():
                """Background thread: runs segmentation on latest frame, updates seg JPEG."""
                while self._active:
                    with seg_input_lock:
                        frame_to_seg = seg_input_frame[0]
                        seg_input_frame[0] = None  # consume it
                    if frame_to_seg is None:
                        time.sleep(0.1)
                        continue
                    try:
                        overall, left_f, right_f = segmenter.analyze(frame_to_seg)
                        if hasattr(segmenter, '_last_mask') and segmenter._last_mask is not None:
                            mask = segmenter._last_mask
                            seg_vis = seg_colors[mask]
                            rgb_sm = cv2.resize(frame_to_seg, (mask.shape[1], mask.shape[0]))
                            seg_vis = cv2.addWeighted(rgb_sm, 0.4, seg_vis, 0.6, 0)
                            seg_vis = cv2.resize(seg_vis, (640, 480))
                            cv2.putText(seg_vis,
                                        f"Feasibility: L={left_f:.2f} R={right_f:.2f} All={overall:.2f}",
                                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            _, seg_buf = cv2.imencode('.jpg', seg_vis,
                                                      [cv2.IMWRITE_JPEG_QUALITY, VIDEO_QUALITY])
                            with self._lock:
                                self._seg_jpeg = seg_buf.tobytes()
                    except Exception as e:
                        print(f"[CAM] Seg error: {e}", flush=True)
                    # Seg takes ~1.7s on Jetson — no sleep needed, it self-throttles

            seg_thread = threading.Thread(target=_seg_worker, daemon=True)
            seg_thread.start()
            print("[CAM] Segmentation worker thread started", flush=True)

        fps_count = 0
        fps_time = time.time()
        fps_val = 0
        frame_step = 0

        while self._active:
            try:
                # --- RGB from cv2.VideoCapture ---
                success, frame = self._cap.read()
                if not success:
                    try:
                        self._cap.release()
                        time.sleep(1)
                        self._cap = cv2.VideoCapture(0)
                    except Exception:
                        pass
                    time.sleep(0.1)
                    continue

                # --- DNN detection overlay (if enabled + model loaded) ---
                if self._detect_enabled and dnn_net is not None:
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(
                        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                    dnn_net.setInput(blob)
                    detections = dnn_net.forward()
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.2:
                            idx = int(detections[0, 0, i, 1])
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (sX, sY, eX, eY) = box.astype("int")
                            label = f"{self.DNN_CLASSES[idx]}: {confidence*100:.1f}%"
                            cv2.rectangle(frame, (sX, sY), (eX, eY), (0, 255, 0), 2)
                            y = sY - 12 if sY - 12 > 12 else sY + 12
                            cv2.putText(frame, label, (sX, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- Depth from depthai (if available) ---
                min_depth = 0
                left_depth = 0
                right_depth = 0
                depth_color = None
                if self._depth_cam:
                    try:
                        depth_frame, min_depth, left_depth, right_depth = \
                            self._depth_cam.get_depth_frame()
                        depth_vis = np.clip(depth_frame / 5.0, 0, 1)
                        depth_vis = (depth_vis * 255).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        depth_color = cv2.resize(depth_color, (640, 480))
                    except Exception:
                        pass

                # --- Feed frame to seg worker thread (non-blocking) ---
                if segmenter is not None and frame_step % 10 == 0:
                    with seg_input_lock:
                        seg_input_frame[0] = frame.copy()

                frame_step += 1

                # FPS
                fps_count += 1
                now = time.time()
                if now - fps_time > 1.0:
                    fps_val = fps_count
                    fps_count = 0
                    fps_time = now

                # Encode to JPEG — quality 20 matches built-in page
                _, rgb_buf = cv2.imencode('.jpg', frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, VIDEO_QUALITY])
                depth_buf = None
                if depth_color is not None:
                    _, depth_buf = cv2.imencode('.jpg', depth_color,
                                                [cv2.IMWRITE_JPEG_QUALITY, VIDEO_QUALITY])
                # Seg JPEG is updated by the seg worker thread (non-blocking)

                with self._lock:
                    self._rgb_jpeg = rgb_buf.tobytes()
                    if depth_buf is not None:
                        self._depth_jpeg = depth_buf.tobytes()
                    self._telemetry = {
                        'fps': fps_val,
                        'min_depth': round(min_depth, 3),
                        'left_depth': round(left_depth, 3),
                        'right_depth': round(right_depth, 3),
                        'detection': 'Monitoring',
                        'objects': '-',
                    }

            except Exception as e:
                print(f"[CAM] Capture error: {e}")
                time.sleep(0.5)

    # --- Detection toggle ---

    def set_detection(self, enabled):
        """Enable/disable DNN object detection overlay on RGB feed."""
        self._detect_enabled = bool(enabled)
        print(f"[CAM] Detection overlay {'ON' if enabled else 'OFF'}")

    @property
    def detect_enabled(self):
        return self._detect_enabled

    # --- Frame getters ---

    def get_rgb_jpeg(self):
        with self._lock:
            return self._rgb_jpeg

    def get_depth_jpeg(self):
        with self._lock:
            return self._depth_jpeg

    def get_seg_jpeg(self):
        with self._lock:
            return self._seg_jpeg

    def get_telemetry(self):
        with self._lock:
            return dict(self._telemetry)

    @property
    def is_active(self):
        return self._active
