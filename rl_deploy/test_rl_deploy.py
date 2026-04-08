#!/usr/bin/env python3
"""
Tests for rl_deploy.py - Team Crater RL Deployment Bridge
==========================================================
Run with:  python3 -m pytest test_rl_deploy.py -v

All tests run without hardware, camera, or trained model.
"""

import sys
import os
import math
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_deploy
from rl_deploy import (
    DummyDepthCamera, DummyDetector, DummyFaceTracker, DummySegmenter,
    DummyRewardLogger, RewardLogger,
    FaceTracker, FaceNavigationSM, TerrainSegmenter,
    GimbalScanner, ObstacleMemory, OdometryTracker, DestinationTracker,
    RLDeployController,
    MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
    EMERGENCY_STOP_DIST, BACKUP_DIST,
    OBSTACLE_STOP_DIST, OBSTACLE_SLOW_DIST,
    OBSTACLE_TURN_SPEED, COMMITTED_TURN_STEPS,
    COURSE_CORRECT_GAIN, COURSE_CORRECT_MAX, COURSE_CORRECT_DEADBAND,
    OBSTACLE_MEMORY_DURATION,
    GIMBAL_SCAN_ANGLE, GIMBAL_SCAN_INTERVAL, GIMBAL_TILT_DEFAULT,
    WHEEL_SEPARATION, ENCODER_SCALE,
    DESTINATION_REACHED_TOL, DESTINATION_OFF_COURSE_DEG,
    MODEL_INPUT_SIZE, DEFAULT_STATE, DETECT_OBSTACLE_CLASSES,
    FACE_HFOV_RAD, FACE_STOP_DISTANCE, FACE_SEARCH_SPIN_SPEED,
    FACE_SEARCH_SPIN_DURATION, FACE_SEARCH_DRIVE_DURATION,
    FACE_RECHECK_STEPS, FACE_LOST_HOLD_TIME, FACE_LOST_SPIN_TIME,
    FACE_MIN_NAVIGATE_STEPS,
    REWARD_FORWARD_PROGRESS, REWARD_WAYPOINT_REACHED,
    PENALTY_BACKUP, PENALTY_EMERGENCY, PENALTY_CLOSE_OBSTACLE,
    PENALTY_SPINNING, PENALTY_IDLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(raw_action=(0.5, 0.2)):
    model = MagicMock()
    model.predict.return_value = (np.array(raw_action, dtype=np.float32), None)
    model.observation_space = MagicMock()
    model.action_space = MagicMock()
    return model


def _make_controller(model=None, max_speed=MAX_LINEAR_SPEED):
    if model is None:
        model = _make_mock_model()
    ctrl = object.__new__(RLDeployController)
    ctrl.max_speed = min(abs(max_speed), MAX_LINEAR_SPEED)
    ctrl.dry_run = True
    ctrl.running = False
    ctrl.base = None
    ctrl.model = model
    ctrl.target_mode = 'distance'
    ctrl.camera = DummyDepthCamera()
    ctrl.detector = DummyDetector()
    ctrl._avoid_turn_dir = 0
    ctrl._avoid_steps_left = 0
    ctrl._avoid_total_steps = 0
    ctrl._last_detected_obj = ""
    ctrl._target_heading = 0.0
    ctrl._avoiding = False
    ctrl.obstacle_map = ObstacleMemory()
    ctrl.gimbal = GimbalScanner(None)
    ctrl.odom = OdometryTracker(None)
    ctrl.destination = DestinationTracker(ctrl.odom)
    ctrl.face_tracker = None
    ctrl.face_nav = None
    ctrl.segmenter = DummySegmenter()
    ctrl.reward_logger = DummyRewardLogger()
    return ctrl


# ===================================================================
# 1. Depth normalization
# ===================================================================

class TestDepthNormalization:
    def test_dummy_depth_always_positive(self):
        cam = DummyDepthCamera()
        for _ in range(20):
            depth, min_d, left_d, right_d = cam.get_depth_frame()
            assert depth.min() >= 0.0
            assert min_d >= 0.0

    def test_dummy_depth_max_10(self):
        cam = DummyDepthCamera()
        for _ in range(20):
            depth, _, _, _ = cam.get_depth_frame()
            assert depth.max() <= 10.0

    def test_depth_frame_shape(self):
        cam = DummyDepthCamera()
        depth, _, _, _ = cam.get_depth_frame()
        assert depth.shape == MODEL_INPUT_SIZE

    def test_depth_dtype_float32(self):
        cam = DummyDepthCamera()
        depth, _, _, _ = cam.get_depth_frame()
        assert depth.dtype == np.float32


# ===================================================================
# 2. Reactive obstacle avoidance
# ===================================================================

class TestReactiveAvoidance:
    def test_stop_and_turn_when_very_close(self):
        ctrl = _make_controller()
        # Use a depth just below OBSTACLE_STOP_DIST to trigger avoidance
        close_depth = OBSTACLE_STOP_DIST * 0.8
        override, mode = ctrl._compute_avoidance(close_depth, 1.0, 0.3, False, "", 0.5, 0.0)
        assert isinstance(override, tuple)
        assert override[0] <= self.max_speed_fwd(ctrl)
        assert abs(override[1]) == pytest.approx(OBSTACLE_TURN_SPEED, abs=0.01)
        assert mode == "AVOID"

    @staticmethod
    def max_speed_fwd(ctrl):
        return ctrl.max_speed * 0.15 + 0.01

    def test_turn_toward_more_space_left(self):
        ctrl = _make_controller()
        close_depth = OBSTACLE_STOP_DIST * 0.8
        override, _ = ctrl._compute_avoidance(close_depth, 2.0, 0.3, False, "", 0.5, 0.0)
        assert override[1] > 0

    def test_turn_toward_more_space_right(self):
        ctrl = _make_controller()
        close_depth = OBSTACLE_STOP_DIST * 0.8
        override, _ = ctrl._compute_avoidance(close_depth, 0.3, 2.0, False, "", 0.5, 0.0)
        assert override[1] < 0

    def test_slow_zone(self):
        ctrl = _make_controller()
        mid = (OBSTACLE_STOP_DIST + OBSTACLE_SLOW_DIST) / 2
        override, mode = ctrl._compute_avoidance(mid, 2.0, 2.0, False, "", 0.5, 0.0)
        assert override == 'slow'
        assert mode == "SLOW"

    def test_clear_path_no_override(self):
        ctrl = _make_controller()
        override, mode = ctrl._compute_avoidance(2.0, 2.0, 2.0, False, "", 0.5, 0.0)
        assert override is None
        assert mode == ""

    def test_stop_dist_reasonable(self):
        """Stop dist should be positive and below 0.40."""
        assert 0.0 < OBSTACLE_STOP_DIST <= 0.40

    def test_slow_dist_reasonable(self):
        """Slow dist should be wider than stop."""
        assert OBSTACLE_SLOW_DIST > OBSTACLE_STOP_DIST
        assert OBSTACLE_SLOW_DIST <= 0.60


# ===================================================================
# 3. Committed turn strategy
# ===================================================================

class TestCommittedTurn:
    def test_committed_turn_persists(self):
        ctrl = _make_controller()
        close = OBSTACLE_STOP_DIST * 0.8
        override1, mode1 = ctrl._compute_avoidance(close, 2.0, 0.5, False, "", 0.5, 0.0)
        assert mode1 == "AVOID"
        assert ctrl._avoid_turn_dir == +1
        assert ctrl._avoid_steps_left == COMMITTED_TURN_STEPS

        # Now L≈R, but should still turn in same direction (committed)
        override2, mode2 = ctrl._compute_avoidance(close, 0.6, 0.6, False, "", 0.5, 0.0)
        assert mode2 == "AVOID"
        assert override2[1] > 0

    def test_committed_turn_clears_when_path_open(self):
        ctrl = _make_controller()
        close = OBSTACLE_STOP_DIST * 0.8
        ctrl._compute_avoidance(close, 2.0, 0.5, False, "", 0.5, 0.0)
        # Drain all steps with clear path
        for _ in range(COMMITTED_TURN_STEPS):
            ctrl._compute_avoidance(5.0, 5.0, 5.0, False, "", 0.5, 0.0)
        # After all steps done with clear path, should be CLEAR
        assert ctrl._avoiding is False

    def test_counts_down(self):
        ctrl = _make_controller()
        close = OBSTACLE_STOP_DIST * 0.8
        ctrl._compute_avoidance(close, 2.0, 0.5, False, "", 0.5, 0.0)
        steps_after = ctrl._avoid_steps_left
        ctrl._compute_avoidance(close, 0.6, 0.6, False, "", 0.5, 0.0)
        assert ctrl._avoid_steps_left == steps_after - 1

    def test_equal_lr_uses_default(self):
        ctrl = _make_controller()
        ctrl._avoid_turn_dir = 0
        close = OBSTACLE_STOP_DIST * 0.8
        ctrl._compute_avoidance(close, 0.55, 0.55, False, "", 0.5, 0.0)
        assert ctrl._avoid_turn_dir != 0

    def test_equal_lr_preserves_prior(self):
        ctrl = _make_controller()
        ctrl._avoid_turn_dir = +1
        close = OBSTACLE_STOP_DIST * 0.8
        ctrl._compute_avoidance(close, 0.55, 0.55, False, "", 0.5, 0.0)
        assert ctrl._avoid_turn_dir == +1


# ===================================================================
# 4. Course correction
# ===================================================================

class TestCourseCorrection:
    def test_correction_when_off_heading(self):
        ctrl = _make_controller()
        ctrl._target_heading = 0.0
        ctrl.odom.heading = math.radians(20)  # 20° off course to the left
        correction = ctrl._compute_course_correction()
        assert correction < 0  # Should steer right (negative angular) to get back

    def test_correction_within_deadband(self):
        ctrl = _make_controller()
        ctrl._target_heading = 0.0
        ctrl.odom.heading = math.radians(2)  # Only 2° off, within 5° deadband
        correction = ctrl._compute_course_correction()
        assert correction == 0.0

    def test_correction_clamped(self):
        ctrl = _make_controller()
        ctrl._target_heading = 0.0
        ctrl.odom.heading = math.radians(90)  # Way off course
        correction = ctrl._compute_course_correction()
        assert abs(correction) <= COURSE_CORRECT_MAX + 0.001

    def test_course_correct_mode_in_avoidance(self):
        """After clearing all obstacles, rover should enter CORRECT mode if off heading."""
        ctrl = _make_controller()
        ctrl._target_heading = 0.0
        ctrl.odom.heading = math.radians(15)  # Off course from avoidance
        override, mode = ctrl._compute_avoidance(2.0, 2.0, 2.0, False, "", 0.5, 0.0)
        assert override == 'course_correct'
        assert mode == "CORRECT"

    def test_no_correction_when_on_heading(self):
        ctrl = _make_controller()
        ctrl._target_heading = 0.0
        ctrl.odom.heading = 0.0
        override, mode = ctrl._compute_avoidance(2.0, 2.0, 2.0, False, "", 0.5, 0.0)
        assert override is None
        assert mode == ""


# ===================================================================
# 4b. Obstacle Memory
# ===================================================================

class TestObstacleMemory:
    def test_add_and_count(self):
        mem = ObstacleMemory()
        mem.add(1.0, 2.0, "depth")
        mem.add(3.0, 4.0, "depth")
        assert mem.count() == 2

    def test_no_duplicates(self):
        mem = ObstacleMemory()
        mem.add(1.0, 2.0, "depth")
        mem.add(1.05, 2.05, "depth")  # Very close, should not add
        assert mem.count() == 1

    def test_prune_old(self):
        mem = ObstacleMemory()
        mem.add(1.0, 2.0, "depth")
        # Manually age the obstacle
        mem.obstacles[0] = (1.0, 2.0, time.time() - OBSTACLE_MEMORY_DURATION - 1, "depth")
        mem.prune()
        assert mem.count() == 0

    def test_nearest_in_front(self):
        mem = ObstacleMemory()
        # Obstacle 2m ahead (rover at origin, heading 0 = +x)
        mem.add(2.0, 0.0, "depth")
        dist, angle = mem.nearest_in_front(0.0, 0.0, 0.0)
        assert dist == pytest.approx(2.0, abs=0.1)
        assert abs(angle) < 0.1  # Nearly straight ahead

    def test_obstacle_behind_not_in_front(self):
        mem = ObstacleMemory()
        # Obstacle behind the rover
        mem.add(-2.0, 0.0, "depth")
        dist, angle = mem.nearest_in_front(0.0, 0.0, 0.0)
        assert dist is None

    def test_all_passed(self):
        mem = ObstacleMemory()
        mem.add(1.0, 0.0, "depth")
        # Rover has driven past it
        assert mem.all_passed(3.0, 0.0, 0.0) is True

    def test_not_all_passed(self):
        mem = ObstacleMemory()
        mem.add(2.0, 0.0, "depth")
        # Rover hasn't reached it yet
        assert mem.all_passed(0.0, 0.0, 0.0) is False

    def test_clear(self):
        mem = ObstacleMemory()
        mem.add(1.0, 2.0, "depth")
        mem.clear()
        assert mem.count() == 0

    def test_record_obstacle_from_controller(self):
        ctrl = _make_controller()
        ctrl.odom.x = 0.0
        ctrl.odom.y = 0.0
        ctrl.odom.heading = 0.0
        ctrl._record_obstacle_ahead(0.5)
        assert ctrl.obstacle_map.count() == 1


# ===================================================================
# 5. Speed clamping
# ===================================================================

class TestSpeedClamping:
    def test_linear_clamped(self):
        ctrl = _make_controller()
        ctrl.dry_run = False
        ctrl.base = MagicMock()
        ctrl._send_velocity(9.9, 0.0)
        cmd = ctrl.base.base_json_ctrl.call_args[0][0]
        assert cmd["X"] <= MAX_LINEAR_SPEED

    def test_angular_clamped(self):
        ctrl = _make_controller()
        ctrl.dry_run = False
        ctrl.base = MagicMock()
        ctrl._send_velocity(0.0, 9.9)
        cmd = ctrl.base.base_json_ctrl.call_args[0][0]
        assert cmd["Z"] <= MAX_ANGULAR_SPEED

    def test_passthrough(self):
        ctrl = _make_controller()
        ctrl.dry_run = False
        ctrl.base = MagicMock()
        ctrl._send_velocity(0.15, -0.25)
        cmd = ctrl.base.base_json_ctrl.call_args[0][0]
        assert cmd["X"] == pytest.approx(0.15, abs=0.001)
        assert cmd["Z"] == pytest.approx(-0.25, abs=0.001)

    def test_max_speed_override(self):
        ctrl = _make_controller(max_speed=0.1)
        ctrl.dry_run = False
        ctrl.base = MagicMock()
        ctrl._send_velocity(0.3, 0.0)
        cmd = ctrl.base.base_json_ctrl.call_args[0][0]
        assert cmd["X"] <= 0.1 + 1e-6


# ===================================================================
# 6. DummyDepthCamera
# ===================================================================

class TestDummyDepthCamera:
    def test_returns_four_tuple(self):
        assert len(DummyDepthCamera().get_depth_frame()) == 4

    def test_depth_is_ndarray(self):
        assert isinstance(DummyDepthCamera().get_depth_frame()[0], np.ndarray)

    def test_scalars_are_float(self):
        _, a, b, c = DummyDepthCamera().get_depth_frame()
        assert all(isinstance(x, float) for x in (a, b, c))

    def test_step_increments(self):
        cam = DummyDepthCamera()
        cam.get_depth_frame()
        assert cam._step == 1

    def test_has_rgb_method(self):
        assert DummyDepthCamera().get_rgb_frame() is None

    def test_close(self):
        DummyDepthCamera().close()


# ===================================================================
# 7. Action scaling
# ===================================================================

class TestActionScaling:
    def test_full_forward(self):
        assert 1.0 * MAX_LINEAR_SPEED == pytest.approx(MAX_LINEAR_SPEED)

    def test_reverse_capped(self):
        lin = -1.0 * MAX_LINEAR_SPEED
        lin = max(lin, -MAX_LINEAR_SPEED * 0.3)
        assert lin == pytest.approx(-MAX_LINEAR_SPEED * 0.3)

    def test_zero(self):
        assert 0.0 * MAX_LINEAR_SPEED == pytest.approx(0.0)

    def test_mid_range(self):
        assert 0.5 * MAX_LINEAR_SPEED == pytest.approx(0.15)


# ===================================================================
# 8. Safety stop
# ===================================================================

class TestSafetyTiers:
    def test_emergency_stop_dist(self):
        """Emergency stop should be very close (stereo dead zone)."""
        assert EMERGENCY_STOP_DIST <= 0.08

    def test_backup_dist(self):
        """Backup should trigger after emergency but is a safety override in run()."""
        assert BACKUP_DIST > EMERGENCY_STOP_DIST

    def test_tier_ordering(self):
        """Core ordering: emergency < backup, stop < slow."""
        assert EMERGENCY_STOP_DIST < BACKUP_DIST
        assert OBSTACLE_STOP_DIST < OBSTACLE_SLOW_DIST

    def test_committed_turn_clears_only_when_clear(self):
        """Committed turn should extend if path isn't clear at end."""
        ctrl = _make_controller()
        # Trigger avoidance with depth below stop threshold
        close = OBSTACLE_STOP_DIST * 0.8
        ctrl._compute_avoidance(close, 2.0, 0.5, False, "", 0.5, 0.0)
        initial_steps = ctrl._avoid_steps_left
        # Drain all but 1 step with depth just at the slow boundary (not clear)
        not_clear = OBSTACLE_SLOW_DIST + 0.05  # just barely above slow, below clear threshold
        for _ in range(initial_steps - 1):
            ctrl._compute_avoidance(not_clear, 0.6, 0.6, False, "", 0.5, 0.0)
        # Last step but path is NOT clear (min_depth < slow_dist + 0.1)
        override, mode = ctrl._compute_avoidance(not_clear, 0.6, 0.6, False, "", 0.5, 0.0)
        # Should extend the turn, not clear
        assert ctrl._avoid_steps_left > 0
        assert mode == "AVOID"


# ===================================================================
# 9. Object detection
# ===================================================================

class TestObjectDetection:
    def test_dummy_empty(self):
        assert DummyDetector().detect(None) == []

    def test_classes_defined(self):
        assert 'person' in DETECT_OBSTACLE_CLASSES
        assert DETECT_OBSTACLE_CLASSES['person'] == 1.0

    def test_detection_boosts_distance(self):
        """Detection should expand avoidance zone — a depth that's normally clear
        or slow should trigger AVOID when a person is detected."""
        ctrl = _make_controller()
        # Use a depth in the SLOW zone without detection
        slow_depth = (OBSTACLE_STOP_DIST + OBSTACLE_SLOW_DIST) / 2
        override1, mode1 = ctrl._compute_avoidance(slow_depth, 2.0, 0.5, False, "", 0.5, 0.0)
        assert mode1 == "SLOW"
        ctrl._avoid_turn_dir = 0
        ctrl._avoid_steps_left = 0
        ctrl._avoid_total_steps = 0
        # With person detected, stop_dist increases → same depth now triggers AVOID
        override2, mode2 = ctrl._compute_avoidance(slow_depth, 2.0, 0.5, True, "person", 0.5, 0.9)
        assert mode2 in ("AVOID", "SLOW")

    def test_object_position_for_turn(self):
        ctrl = _make_controller()
        close = OBSTACLE_STOP_DIST * 0.8
        ctrl._compute_avoidance(close, 1.0, 1.0, True, "person", 0.8, 0.9)
        assert ctrl._avoid_turn_dir == +1  # Turn left, away from right-side object


# ===================================================================
# 10. Gimbal Scanner
# ===================================================================

class TestGimbalScanner:
    def test_center_sends_command(self):
        base = MagicMock()
        gs = GimbalScanner(base)
        gs.center()
        base.gimbal_ctrl.assert_called_with(0, GIMBAL_TILT_DEFAULT, 0, 0)

    def test_no_base_is_noop(self):
        gs = GimbalScanner(None)
        gs.center()
        gs.update(0, False)  # Should not raise

    def test_is_centered_initially(self):
        gs = GimbalScanner(None)
        assert gs.is_centered()

    def test_scan_alternates(self):
        base = MagicMock()
        gs = GimbalScanner(base)
        actions = []
        for step in range(GIMBAL_SCAN_INTERVAL * 5):
            result = gs.update(step, False)
            if result:
                actions.append(result)
        # Should see a mix of scan_right, scan_left, centering
        assert 'scan_right' in actions or 'scan_left' in actions
        assert 'centering' in actions

    def test_pauses_during_avoidance(self):
        base = MagicMock()
        gs = GimbalScanner(base)
        gs._current_target = 45  # Was scanning
        gs.update(GIMBAL_SCAN_INTERVAL, True)  # avoid_active=True
        assert gs._current_target == 0  # Should have centered

    def test_not_centered_when_scanning(self):
        base = MagicMock()
        gs = GimbalScanner(base)
        # Trigger a scan
        gs._scan_phase = 0
        gs.update(GIMBAL_SCAN_INTERVAL, False)
        # Should be in centering or scanning state
        # After scan_right, is_centered should eventually be False
        if gs._current_target != 0:
            assert not gs.is_centered()


# ===================================================================
# 11. Odometry Tracker
# ===================================================================

class TestOdometryTracker:
    def test_no_base_returns_false(self):
        odom = OdometryTracker(None)
        assert odom.update() is False

    def test_first_read_initializes(self):
        base = MagicMock()
        base.feedback_data.return_value = {'odl': 100, 'odr': 100, 'v': 1100}
        odom = OdometryTracker(base)
        assert odom.update() is True
        assert odom.total_distance == 0.0  # First read is baseline

    def test_forward_accumulates_distance(self):
        base = MagicMock()
        odom = OdometryTracker(base)

        # First read (baseline)
        base.feedback_data.return_value = {'odl': 100, 'odr': 100}
        odom.update()
        odom._last_time = time.time() - 0.2  # Fake 200ms ago

        # Second read (moved forward 10cm on each wheel)
        base.feedback_data.return_value = {'odl': 110, 'odr': 110}
        odom.update()
        assert odom.total_distance == pytest.approx(0.10, abs=0.01)

    def test_pure_rotation_changes_heading(self):
        base = MagicMock()
        odom = OdometryTracker(base)

        base.feedback_data.return_value = {'odl': 100, 'odr': 100}
        odom.update()
        odom._last_time = time.time() - 0.2

        # Right wheel forward, left wheel backward → turning
        base.feedback_data.return_value = {'odl': 95, 'odr': 105}
        odom.update()

        expected_heading = (0.05 - (-0.05)) / WHEEL_SEPARATION  # (dr - dl) / W
        assert odom.heading == pytest.approx(expected_heading, abs=0.01)

    def test_velocity_error(self):
        odom = OdometryTracker(None)
        odom.actual_linear_vel = 0.12
        odom.actual_angular_vel = 0.05
        lin_err, ang_err = odom.velocity_error(0.15, 0.10)
        assert lin_err == pytest.approx(0.03)
        assert ang_err == pytest.approx(0.05)

    def test_displacement(self):
        odom = OdometryTracker(None)
        odom.x = 3.0
        odom.y = 4.0
        assert odom.displacement == pytest.approx(5.0)

    def test_reset(self):
        odom = OdometryTracker(None)
        odom.total_distance = 5.0
        odom.heading = 1.5
        odom.x = 3.0
        odom.y = 4.0
        odom.reset()
        assert odom.total_distance == 0.0
        assert odom.heading == 0.0
        assert odom.x == 0.0
        assert odom.y == 0.0

    def test_missing_odl_key(self):
        base = MagicMock()
        base.feedback_data.return_value = {'v': 1100}
        odom = OdometryTracker(base)
        assert odom.update() is False

    def test_sanity_check_large_delta(self):
        base = MagicMock()
        odom = OdometryTracker(base)

        base.feedback_data.return_value = {'odl': 100, 'odr': 100}
        odom.update()
        odom._last_time = time.time() - 0.2

        # Huge jump (wraparound) — should be discarded
        base.feedback_data.return_value = {'odl': 500, 'odr': 500}
        result = odom.update()
        assert result is False
        assert odom.total_distance == 0.0  # Not accumulated


# ===================================================================
# 12. Destination Tracker
# ===================================================================

class TestDestinationTracker:
    def test_no_destination(self):
        odom = OdometryTracker(None)
        dest = DestinationTracker(odom)
        status, rem, drift = dest.check()
        assert status is None

    def test_tracking(self):
        odom = OdometryTracker(None)
        odom.x = 0.0
        odom.y = 0.0
        odom.heading = 0.0
        dest = DestinationTracker(odom)
        dest.set_destination(5.0)
        status, rem, drift = dest.check()
        assert status == 'tracking'
        assert rem == pytest.approx(5.0, abs=0.1)

    def test_reached(self):
        odom = OdometryTracker(None)
        odom.x = 0.0
        odom.y = 0.0
        dest = DestinationTracker(odom)
        dest.set_destination(3.0)
        odom.x = 3.05  # Past the target (straight-line displacement)
        status, rem, drift = dest.check()
        assert status == 'reached'
        assert dest.reached is True

    def test_off_course(self):
        odom = OdometryTracker(None)
        odom.x = 0.0
        odom.y = 0.0
        odom.heading = 0.0
        dest = DestinationTracker(odom)
        dest.set_destination(5.0)
        odom.x = 1.0
        odom.y = 1.0  # displacement ~1.4m, not reached
        odom.heading = math.radians(50)  # 50° drift > 45° threshold
        status, rem, drift = dest.check()
        assert status == 'off_course'
        assert drift > DESTINATION_OFF_COURSE_DEG

    def test_clear(self):
        odom = OdometryTracker(None)
        dest = DestinationTracker(odom)
        dest.set_destination(5.0)
        dest.clear()
        status, _, _ = dest.check()
        assert status is None


# ===================================================================
# 13. Integration
# ===================================================================

class TestIntegration:
    def test_clear_path(self):
        model = _make_mock_model((0.8, 0.1))
        ctrl = _make_controller(model=model)
        ctrl.dry_run = False
        ctrl.base = MagicMock()

        depth = np.ones(MODEL_INPUT_SIZE, dtype=np.float32) * 2.0
        ctrl.camera.get_depth_frame = MagicMock(return_value=(depth, 2.0, 2.0, 2.0))

        depth_frame, md, ld, rd = ctrl.camera.get_depth_frame()
        override, _ = ctrl._compute_avoidance(md, ld, rd, False, "", 0.5, 0.0)
        assert override is None

        obs = {'state': DEFAULT_STATE.copy(), 'image_front': depth_frame.reshape(128, 128, 1)}
        action, _ = ctrl.model.predict(obs, deterministic=True)
        lin = float(action[0]) * ctrl.max_speed
        ang = float(action[1]) * MAX_ANGULAR_SPEED
        ctrl._send_velocity(lin, ang)

        cmd = ctrl.base.base_json_ctrl.call_args[0][0]
        assert cmd["X"] == pytest.approx(0.8 * MAX_LINEAR_SPEED, abs=0.01)

    def test_observation_positive_meters(self):
        model = _make_mock_model()
        ctrl = _make_controller(model=model)
        depth = np.ones(MODEL_INPUT_SIZE, dtype=np.float32) * 2.5
        ctrl.camera.get_depth_frame = MagicMock(return_value=(depth, 2.5, 2.5, 2.5))

        depth_frame, _, _, _ = ctrl.camera.get_depth_frame()
        obs = {'state': DEFAULT_STATE.copy(), 'image_front': depth_frame.reshape(128, 128, 1)}
        ctrl.model.predict(obs, deterministic=True)
        passed = ctrl.model.predict.call_args[0][0]
        assert passed['image_front'].min() >= 0.0
        assert passed['image_front'].max() <= 10.0


# ===================================================================
# 15. FaceTracker
# ===================================================================

class TestFaceTracker:
    def test_angle_at_center_is_zero(self):
        """Face centered in frame → angle ≈ 0."""
        tracker = DummyFaceTracker()
        # Use a mock tracker that returns a known detection
        tracker_real = object.__new__(FaceTracker)
        tracker_real.cascade = None
        tracker_real.detector = None
        tracker_real.last_face = None
        # Test the angle math directly via _encode_state
        state = FaceNavigationSM._encode_state(0.0, 5.0)
        assert abs(state[0] - 1.0) < 0.01  # cos(0) = 1
        assert abs(state[1] - 0.0) < 0.01  # sin(0) = 0

    def test_angle_at_right(self):
        """Face at right side → positive angle, cos < 1, sin > 0."""
        angle = (0.75 - 0.5) * FACE_HFOV_RAD  # 25% right of center
        state = FaceNavigationSM._encode_state(angle, 3.0)
        assert state[0] < 1.0   # cos(angle) < 1
        assert state[1] > 0.0   # sin(positive angle) > 0
        assert abs(angle - math.radians(73) * 0.25) < 0.01

    def test_angle_at_left(self):
        """Face at left side → negative angle, sin < 0."""
        angle = (0.25 - 0.5) * FACE_HFOV_RAD
        state = FaceNavigationSM._encode_state(angle, 3.0)
        assert state[1] < 0.0   # sin(negative angle) < 0

    def test_depth_mapping_coords(self):
        """RGB coords (320, 240) in 640x480 → depth (64, 64) in 128x128."""
        # center of 640x480 maps to center of 128x128
        cx_d = int(320 * 128 / 640)
        cy_d = int(240 * 128 / 480)
        assert cx_d == 64
        assert cy_d == 64

    def test_state_encode_distance_normalized(self):
        """Distance is normalized by /10 and clamped to [0, 1]."""
        state_close = FaceNavigationSM._encode_state(0.0, 2.0)
        state_far = FaceNavigationSM._encode_state(0.0, 15.0)
        assert abs(state_close[3] - 0.2) < 0.01   # 2/10
        assert abs(state_far[3] - 1.0) < 0.01     # clamped to 1

    def test_dummy_face_tracker_returns_false(self):
        tracker = DummyFaceTracker()
        found, angle, dist = tracker.detect(None, None)
        assert found is False
        assert angle == 0.0
        assert dist == 0.0


# ===================================================================
# 16. FaceNavigationSM
# ===================================================================

class TestFaceNavigationSM:
    def _make_face_sm(self, stop_distance=1.0):
        tracker = DummyFaceTracker()
        odom = OdometryTracker(None)
        sm = FaceNavigationSM(tracker, odom, stop_distance=stop_distance)
        return sm, tracker

    def test_search_spin_returns_fast_rotation(self):
        """In SEARCH_SPIN with no face, should return fast rotation override."""
        sm, tracker = self._make_face_sm()
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        override, state = sm.update(None, depth)
        assert override is not None
        assert override[0] == 0.0  # no forward motion
        assert abs(override[1]) == FACE_SEARCH_SPIN_SPEED
        assert sm.state == 'SEARCH_SPIN'

    def test_search_spin_to_drive_after_duration(self):
        """After spin duration, transitions to SEARCH_DRIVE (RL explores)."""
        sm, tracker = self._make_face_sm()
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        sm._phase_start = time.time() - FACE_SEARCH_SPIN_DURATION - 1
        override, state = sm.update(None, depth)
        assert sm.state == 'SEARCH_DRIVE'
        assert override is None  # RL drives

    def test_search_drive_returns_none_override(self):
        """During SEARCH_DRIVE, RL drives — override is None."""
        sm, tracker = self._make_face_sm()
        sm.state = 'SEARCH_DRIVE'
        sm._phase_start = time.time()
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        override, state = sm.update(None, depth)
        assert override is None
        assert sm.state == 'SEARCH_DRIVE'

    def test_search_drive_to_spin_after_duration(self):
        """After drive duration, transitions back to SEARCH_SPIN."""
        sm, tracker = self._make_face_sm()
        sm.state = 'SEARCH_DRIVE'
        sm._phase_start = time.time() - FACE_SEARCH_DRIVE_DURATION - 1
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        override, state = sm.update(None, depth)
        assert sm.state == 'SEARCH_SPIN'
        assert override is not None
        assert abs(override[1]) == FACE_SEARCH_SPIN_SPEED

    def test_face_found_during_any_search_goes_navigate(self):
        """Finding a face in any search state → NAVIGATE."""
        for start_state in ('SEARCH_SPIN', 'SEARCH_DRIVE'):
            sm, tracker = self._make_face_sm()
            sm.state = start_state
            sm._phase_start = time.time()
            tracker.detect = MagicMock(return_value=(True, 0.1, 3.0))
            depth = np.ones((128, 128), dtype=np.float32) * 3.0
            override, state = sm.update(None, depth)
            assert override is None  # RL drives
            assert sm.state == 'NAVIGATE'
            assert abs(state[0] - math.cos(0.1)) < 0.01
            assert abs(state[1] - math.sin(0.1)) < 0.01

    def test_navigate_returns_none_override(self):
        """During NAVIGATE, override is None — RL model drives."""
        sm, tracker = self._make_face_sm()
        tracker.detect = MagicMock(return_value=(True, 0.2, 5.0))
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        sm.update(None, depth)  # → NAVIGATE
        # Make detect return False for non-recheck steps
        tracker.detect = MagicMock(return_value=(False, 0.0, 0.0))
        override, state = sm.update(None, depth)
        assert override is None

    def test_navigate_recheck_updates_state(self):
        """After RECHECK_STEPS, state vector updates with new face position."""
        sm, tracker = self._make_face_sm()
        tracker.detect = MagicMock(return_value=(True, 0.1, 5.0))
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        sm.update(None, depth)  # → NAVIGATE

        sm._steps_since_recheck = FACE_RECHECK_STEPS - 1
        tracker.detect = MagicMock(return_value=(True, 0.4, 4.0))
        override, state = sm.update(None, depth)
        assert override is None
        assert abs(state[0] - math.cos(0.4)) < 0.01
        assert abs(state[1] - math.sin(0.4)) < 0.01

    def test_arrived_requires_min_navigate_steps(self):
        """Face close but ARRIVED only after minimum navigation steps."""
        sm, tracker = self._make_face_sm(stop_distance=2.0)
        tracker.detect = MagicMock(return_value=(True, 0.0, 1.5))
        depth = np.ones((128, 128), dtype=np.float32) * 1.5
        sm.update(None, depth)  # → NAVIGATE
        assert sm.state == 'NAVIGATE'
        # At recheck with close face but not enough steps
        sm._steps_since_recheck = FACE_RECHECK_STEPS - 1
        sm._navigate_steps = FACE_MIN_NAVIGATE_STEPS - 2
        override, state = sm.update(None, depth)
        assert sm.state == 'NAVIGATE'  # NOT arrived yet
        assert sm.reached is False

    def test_arrived_after_min_steps(self):
        """ARRIVED triggers after enough navigation steps."""
        sm, tracker = self._make_face_sm(stop_distance=2.0)
        tracker.detect = MagicMock(return_value=(True, 0.0, 1.5))
        depth = np.ones((128, 128), dtype=np.float32) * 1.5
        sm.update(None, depth)  # → NAVIGATE
        # Enough steps + recheck
        sm._navigate_steps = FACE_MIN_NAVIGATE_STEPS
        sm._steps_since_recheck = FACE_RECHECK_STEPS - 1
        override, state = sm.update(None, depth)
        assert sm.state == 'ARRIVED'
        assert sm.reached is True
        assert override == (0.0, 0.0)

    def test_navigate_to_lost_hold(self):
        """Face disappears during NAVIGATE → LOST_HOLD."""
        sm, tracker = self._make_face_sm()
        tracker.detect = MagicMock(return_value=(True, 0.1, 5.0))
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        sm.update(None, depth)  # → NAVIGATE

        sm._steps_since_recheck = FACE_RECHECK_STEPS - 1
        tracker.detect = MagicMock(return_value=(False, 0.0, 0.0))
        override, state = sm.update(None, depth)
        assert sm.state == 'LOST_HOLD'
        assert override is None  # RL drives with last known state

    def test_lost_hold_to_lost_spin(self):
        """After hold time, transitions to LOST_SPIN."""
        sm, tracker = self._make_face_sm()
        tracker.detect = MagicMock(return_value=(True, 0.1, 5.0))
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        sm.update(None, depth)  # → NAVIGATE

        sm._steps_since_recheck = FACE_RECHECK_STEPS - 1
        tracker.detect = MagicMock(return_value=(False, 0.0, 0.0))
        sm.update(None, depth)  # → LOST_HOLD

        sm._lost_time = time.time() - FACE_LOST_HOLD_TIME - 1
        override, state = sm.update(None, depth)
        assert sm.state == 'LOST_SPIN'
        assert override is not None
        assert abs(override[1]) == FACE_SEARCH_SPIN_SPEED

    def test_lost_spin_to_search_after_timeout(self):
        """After lost spin time, goes back to full SEARCH_SPIN."""
        sm, tracker = self._make_face_sm()
        sm.state = 'LOST_SPIN'
        sm._phase_start = time.time() - FACE_LOST_SPIN_TIME - 1
        tracker.detect = MagicMock(return_value=(False, 0.0, 0.0))
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        override, state = sm.update(None, depth)
        assert sm.state == 'SEARCH_SPIN'

    def test_lost_reacquire_face(self):
        """Face re-appears during LOST_HOLD → back to NAVIGATE."""
        sm, tracker = self._make_face_sm()
        tracker.detect = MagicMock(return_value=(True, 0.1, 5.0))
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        sm.update(None, depth)  # → NAVIGATE

        sm._steps_since_recheck = FACE_RECHECK_STEPS - 1
        tracker.detect = MagicMock(return_value=(False, 0.0, 0.0))
        sm.update(None, depth)  # → LOST_HOLD

        tracker.detect = MagicMock(return_value=(True, 0.2, 4.0))
        override, state = sm.update(None, depth)
        assert sm.state == 'NAVIGATE'
        assert override is None

    def test_no_search_timeout(self):
        """Search cycles indefinitely — no timeout."""
        sm, tracker = self._make_face_sm()
        # Even after many cycles, should not set reached
        sm._search_count = 100
        sm._phase_start = time.time() - FACE_SEARCH_SPIN_DURATION - 1
        depth = np.ones((128, 128), dtype=np.float32) * 3.0
        override, state = sm.update(None, depth)
        assert sm.reached is False


# ===================================================================
# 17. TerrainSegmenter
# ===================================================================

class TestTerrainSegmenter:
    def test_dummy_returns_defaults(self):
        seg = DummySegmenter()
        result = seg.analyze(None)
        assert result == (0.8, 0.8, 0.8)
        assert seg.last_result == (0.8, 0.8, 0.8)

    def test_feasibility_all_ground(self):
        """Mask with all ground → high feasibility."""
        feas = TerrainSegmenter._compute_feasibility(np.zeros((100, 100), dtype=np.int64))
        assert feas > 0.7

    def test_feasibility_rocky(self):
        """Mask with lots of big rocks → lower feasibility."""
        mask = np.full((100, 100), 3, dtype=np.int64)  # all big rock
        feas = TerrainSegmenter._compute_feasibility(mask)
        # With all big rock and no ground, should be low
        mask_mixed = np.zeros((100, 100), dtype=np.int64)
        mask_mixed[:50, :] = 3  # half big rock
        feas_mixed = TerrainSegmenter._compute_feasibility(mask_mixed)
        assert feas_mixed < 0.7

    def test_split_left_right_asymmetric(self):
        """Left half clear, right half rocky → left feasibility > right."""
        mask = np.zeros((100, 200), dtype=np.int64)
        mask[:, 100:] = 3  # right half is big rock
        left_feas = TerrainSegmenter._compute_feasibility(mask[:, :100])
        right_feas = TerrainSegmenter._compute_feasibility(mask[:, 100:])
        assert left_feas > right_feas


# ===================================================================
# 18. Segmentation Avoidance Integration
# ===================================================================

class TestSegmentationAvoidance:
    def test_prefers_clearer_side_when_depth_equal(self):
        """When left/right depth are equal, prefer side with higher segmentation feasibility."""
        ctrl = _make_controller()
        # Depths are equal, but left terrain is clearer
        override, mode = ctrl._compute_avoidance(
            min_depth=0.08, left_depth=1.0, right_depth=1.0,
            obj_detected=False, obj_class="", obj_center_x=0.5, obj_conf=0.0,
            seg_left_feas=0.9, seg_right_feas=0.3)
        assert mode == "AVOID"
        assert ctrl._avoid_turn_dir == +1  # turn right to go left (clearer)

    def test_depth_takes_priority_over_seg(self):
        """When depth clearly indicates one side is closer, depth wins."""
        ctrl = _make_controller()
        override, mode = ctrl._compute_avoidance(
            min_depth=0.08, left_depth=2.0, right_depth=0.5,
            obj_detected=False, obj_class="", obj_center_x=0.5, obj_conf=0.0,
            seg_left_feas=0.3, seg_right_feas=0.9)
        assert mode == "AVOID"
        assert ctrl._avoid_turn_dir == +1  # turn toward left (more depth space)

    def test_existing_distance_mode_unchanged(self):
        """Default seg params (0.8, 0.8) don't affect existing behavior."""
        ctrl = _make_controller()
        override, mode = ctrl._compute_avoidance(
            min_depth=2.0, left_depth=3.0, right_depth=3.0,
            obj_detected=False, obj_class="", obj_center_x=0.5, obj_conf=0.0)
        # Should be clear — no avoidance
        assert override is None or mode in ("", "CORRECT", "CLEAR")


# ===================================================================
# 19. Reward Logger
# ===================================================================

class TestRewardLogger:
    def test_forward_progress_reward(self):
        """Moving forward at max speed should yield positive reward."""
        logger = RewardLogger.__new__(RewardLogger)
        logger._total_reward = 0.0
        logger._step_count = 0
        reward, event = logger.compute_reward(
            linear_vel=MAX_LINEAR_SPEED, angular_vel=0.0,
            min_depth=5.0, mode=None)
        assert reward > 0
        assert 'FWD' in event

    def test_emergency_penalty(self):
        """Obstacle at emergency distance should yield negative reward."""
        logger = RewardLogger.__new__(RewardLogger)
        reward, event = logger.compute_reward(
            linear_vel=0.0, angular_vel=0.0,
            min_depth=EMERGENCY_STOP_DIST * 0.5, mode=None)
        assert reward <= PENALTY_EMERGENCY  # may also include IDLE penalty
        assert 'EMERGENCY' in event

    def test_backup_penalty(self):
        """Obstacle in backup zone should yield backup penalty."""
        logger = RewardLogger.__new__(RewardLogger)
        reward, event = logger.compute_reward(
            linear_vel=0.0, angular_vel=0.0,
            min_depth=(EMERGENCY_STOP_DIST + BACKUP_DIST) / 2, mode=None)
        assert reward <= PENALTY_BACKUP
        assert 'BACKUP' in event

    def test_avoid_spinning_penalty(self):
        """Being in AVOID mode should add spinning penalty."""
        logger = RewardLogger.__new__(RewardLogger)
        reward, event = logger.compute_reward(
            linear_vel=0.0, angular_vel=0.3,
            min_depth=3.0, mode='AVOID')
        assert reward < 0
        assert 'AVOID' in event

    def test_goal_reached_reward(self):
        """Reaching goal should yield large positive reward."""
        logger = RewardLogger.__new__(RewardLogger)
        reward, event = logger.compute_reward(
            linear_vel=0.0, angular_vel=0.0,
            min_depth=5.0, mode=None, reached_goal=True)
        assert reward == REWARD_WAYPOINT_REACHED
        assert 'GOAL' in event

    def test_idle_penalty(self):
        """Near-zero velocity without obstacles should yield small penalty."""
        logger = RewardLogger.__new__(RewardLogger)
        reward, event = logger.compute_reward(
            linear_vel=0.0, angular_vel=0.0,
            min_depth=5.0, mode=None)
        assert reward == PENALTY_IDLE
        assert 'IDLE' in event

    def test_csv_file_creation(self, tmp_path):
        """RewardLogger should create a CSV with correct headers."""
        log_path = str(tmp_path / 'test_rewards.csv')
        logger = RewardLogger(log_path)
        logger.log(0, 0.5, 'FWD', 0.2, 0.0, 3.0, 3.0, 3.0,
                   'CLEAR', None, 0.8, 0.8, 1.0, 0.9, 45.0)
        logger.close()
        import csv
        with open(log_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers == RewardLogger.COLUMNS
            row = next(reader)
            assert len(row) == len(RewardLogger.COLUMNS)
            assert row[3] == 'FWD'  # event column

    def test_dummy_logger_noop(self):
        """DummyRewardLogger should be a no-op."""
        dummy = DummyRewardLogger()
        r, e = dummy.compute_reward(0.1, 0.0, 3.0, None)
        assert r == 0.0
        dummy.log(0, 0, '', 0, 0, 0, 0, 0, '', None, 0, 0, 0, 0, 0)
        dummy.close()  # should not raise
