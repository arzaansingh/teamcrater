#!/usr/bin/env python3
"""
Tests for rover_control.py and mission_control.py — Team Crater
=================================================================
Run with:  python3 -m pytest test_mission_control.py -v

All tests run without hardware, camera, or trained model.
Uses constants from source modules — no hardcoded magic numbers.
"""

import sys
import os
import time
import math
import threading
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rover_control import (
    RoverHardware, DummyRoverHardware, CameraManager,
    MAX_LINEAR, MAX_ANGULAR, SERIAL_PORT, BAUD_RATE,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_mock_base():
    """Create a mock BaseController with all expected methods."""
    base = MagicMock()
    base.base_json_ctrl = MagicMock()
    base.gimbal_ctrl = MagicMock()
    base.lights_ctrl = MagicMock()
    base.feedback_data = MagicMock(return_value={'T': 1003, 'battery': 12.1})
    base.ser = MagicMock()
    return base


def _make_rover_with_mock():
    """Create a RoverHardware with a mocked BaseController."""
    with patch('rover_control.RoverHardware.__init__', lambda self, *a, **kw: None):
        hw = RoverHardware.__new__(RoverHardware)
        hw.base = _make_mock_base()
        hw._port = SERIAL_PORT
        hw._baud = BAUD_RATE
        hw._active = True
        return hw


# ===========================================================================
# TestDummyRoverHardware — no-op stub
# ===========================================================================

class TestDummyRoverHardware:
    """DummyRoverHardware should be a safe no-op for all methods."""

    def test_drive_does_nothing(self):
        d = DummyRoverHardware()
        d.drive(0.2, 0.3)  # should not raise

    def test_stop_does_nothing(self):
        d = DummyRoverHardware()
        d.stop()

    def test_gimbal_does_nothing(self):
        d = DummyRoverHardware()
        d.gimbal_set(45, 10)
        d.gimbal_center()

    def test_lights_does_nothing(self):
        d = DummyRoverHardware()
        d.lights_set(128, 255)

    def test_telemetry_returns_empty(self):
        d = DummyRoverHardware()
        assert d.get_telemetry() == {}

    def test_is_not_connected(self):
        d = DummyRoverHardware()
        assert d.is_connected is False

    def test_serial_lifecycle_noop(self):
        d = DummyRoverHardware()
        d.release_serial()
        d.reacquire_serial()
        d.emergency_stop()


# ===========================================================================
# TestRoverHardware — motor, gimbal, LEDs, telemetry
# ===========================================================================

class TestRoverHardwareDrive:
    """Test drive command clamping and serial forwarding."""

    def test_drive_sends_correct_json(self):
        hw = _make_rover_with_mock()
        hw.drive(0.1, 0.2)
        hw.base.base_json_ctrl.assert_called_once_with(
            {"T": 13, "X": 0.1, "Z": 0.2}
        )

    def test_drive_clamps_linear_to_max(self):
        hw = _make_rover_with_mock()
        hw.drive(1.0, 0.0)  # way above MAX_LINEAR
        call_args = hw.base.base_json_ctrl.call_args[0][0]
        assert call_args["X"] == round(MAX_LINEAR, 3)

    def test_drive_clamps_linear_negative(self):
        hw = _make_rover_with_mock()
        hw.drive(-1.0, 0.0)
        call_args = hw.base.base_json_ctrl.call_args[0][0]
        assert call_args["X"] == round(-MAX_LINEAR, 3)

    def test_drive_clamps_angular_to_max(self):
        hw = _make_rover_with_mock()
        hw.drive(0.0, 2.0)
        call_args = hw.base.base_json_ctrl.call_args[0][0]
        assert call_args["Z"] == round(MAX_ANGULAR, 3)

    def test_drive_clamps_angular_negative(self):
        hw = _make_rover_with_mock()
        hw.drive(0.0, -2.0)
        call_args = hw.base.base_json_ctrl.call_args[0][0]
        assert call_args["Z"] == round(-MAX_ANGULAR, 3)

    def test_stop_sends_zero(self):
        hw = _make_rover_with_mock()
        hw.stop()
        call_args = hw.base.base_json_ctrl.call_args[0][0]
        assert call_args["X"] == 0.0
        assert call_args["Z"] == 0.0

    def test_drive_noop_when_inactive(self):
        hw = _make_rover_with_mock()
        hw._active = False
        hw.drive(0.1, 0.1)
        hw.base.base_json_ctrl.assert_not_called()

    def test_drive_noop_when_no_base(self):
        hw = _make_rover_with_mock()
        hw.base = None
        hw.drive(0.1, 0.1)  # should not raise


class TestRoverHardwareGimbal:
    """Test gimbal commands."""

    def test_gimbal_set_sends_correct_json(self):
        hw = _make_rover_with_mock()
        hw.gimbal_set(30, -10, speed=50, accel=30)
        hw.base.gimbal_ctrl.assert_called_once_with(30, -10, 50, 30)

    def test_gimbal_center_sends_zeros(self):
        hw = _make_rover_with_mock()
        hw.gimbal_center()
        hw.base.gimbal_ctrl.assert_called_once_with(0, 0, 0, 0)

    def test_gimbal_noop_when_inactive(self):
        hw = _make_rover_with_mock()
        hw._active = False
        hw.gimbal_set(45, 20)
        hw.base.gimbal_ctrl.assert_not_called()


class TestRoverHardwareLights:
    """Test LED control with clamping."""

    def test_lights_set_sends_values(self):
        hw = _make_rover_with_mock()
        hw.lights_set(100, 200)
        hw.base.lights_ctrl.assert_called_once_with(100, 200)

    def test_lights_clamps_to_255(self):
        hw = _make_rover_with_mock()
        hw.lights_set(300, -10)
        hw.base.lights_ctrl.assert_called_once_with(255, 0)

    def test_lights_noop_when_inactive(self):
        hw = _make_rover_with_mock()
        hw._active = False
        hw.lights_set(100, 100)
        hw.base.lights_ctrl.assert_not_called()


class TestRoverHardwareTelemetry:
    """Test telemetry reading."""

    def test_telemetry_returns_data(self):
        hw = _make_rover_with_mock()
        data = hw.get_telemetry()
        assert data['T'] == 1003
        assert data['battery'] == 12.1

    def test_telemetry_returns_empty_on_error(self):
        hw = _make_rover_with_mock()
        hw.base.feedback_data.side_effect = Exception("serial error")
        assert hw.get_telemetry() == {}

    def test_telemetry_returns_empty_when_none(self):
        hw = _make_rover_with_mock()
        hw.base.feedback_data.return_value = None
        assert hw.get_telemetry() == {}

    def test_telemetry_returns_empty_when_inactive(self):
        hw = _make_rover_with_mock()
        hw._active = False
        assert hw.get_telemetry() == {}


class TestRoverHardwareSerialLifecycle:
    """Test serial port release/reacquire for mode transitions."""

    def test_release_closes_serial(self):
        hw = _make_rover_with_mock()
        hw.release_serial()
        hw.base.ser.close.assert_called_once()
        assert hw._active is False

    def test_release_sets_inactive(self):
        hw = _make_rover_with_mock()
        hw.release_serial()
        assert hw._active is False
        # Drive should be noop after release
        hw.drive(0.1, 0.0)
        hw.base.base_json_ctrl.assert_not_called()

    def test_reacquire_sets_active(self):
        hw = _make_rover_with_mock()
        hw._active = False
        # BaseController is imported inside reacquire_serial via 'from base_ctrl import'
        with patch.dict('sys.modules', {'base_ctrl': MagicMock(BaseController=MagicMock(return_value=_make_mock_base()))}):
            hw.reacquire_serial()
        assert hw._active is True

    def test_is_connected_property(self):
        hw = _make_rover_with_mock()
        assert hw.is_connected is True
        hw._active = False
        assert hw.is_connected is False
        hw._active = True
        hw.base = None
        assert hw.is_connected is False

    def test_emergency_stop_sends_double_stop(self):
        hw = _make_rover_with_mock()
        hw.emergency_stop()
        # Should call base_json_ctrl at least twice (double-send)
        assert hw.base.base_json_ctrl.call_count >= 2
        for call in hw.base.base_json_ctrl.call_args_list:
            assert call[0][0]["X"] == 0.0
            assert call[0][0]["Z"] == 0.0


# ===========================================================================
# TestCameraManager — lifecycle, thread safety
# ===========================================================================

class TestCameraManager:
    """Test CameraManager initialization and state management."""

    def test_init_defaults(self):
        cm = CameraManager()
        assert cm.is_active is False
        assert cm.get_rgb_jpeg() is None
        assert cm.get_depth_jpeg() is None
        assert cm.get_seg_jpeg() is None

    def test_init_with_segmentation(self):
        cm = CameraManager(enable_seg=True, seg_model_path='/fake/model.pth')
        assert cm._enable_seg is True
        assert cm._seg_model_path == '/fake/model.pth'

    def test_telemetry_returns_dict(self):
        cm = CameraManager()
        telem = cm.get_telemetry()
        assert isinstance(telem, dict)
        assert 'fps' in telem
        assert 'min_depth' in telem
        assert 'detection' in telem

    def test_stop_when_not_started(self):
        """Stop should be safe to call even when never started."""
        cm = CameraManager()
        cm.stop()  # should not raise
        assert cm.is_active is False

    def test_double_stop_safe(self):
        cm = CameraManager()
        cm.stop()
        cm.stop()  # should not raise

    def test_frame_buffers_thread_safe(self):
        """Verify that frame getters use the lock (no crash under concurrent access)."""
        cm = CameraManager()
        # Simulate setting frames from "capture thread"
        with cm._lock:
            cm._rgb_jpeg = b'\xff\xd8fake_jpeg'
            cm._depth_jpeg = b'\xff\xd8fake_depth'
            cm._seg_jpeg = b'\xff\xd8fake_seg'
        assert cm.get_rgb_jpeg() == b'\xff\xd8fake_jpeg'
        assert cm.get_depth_jpeg() == b'\xff\xd8fake_depth'
        assert cm.get_seg_jpeg() == b'\xff\xd8fake_seg'

    def test_stop_clears_buffers(self):
        cm = CameraManager()
        with cm._lock:
            cm._rgb_jpeg = b'data'
        cm.stop()
        assert cm.get_rgb_jpeg() is None


# ===========================================================================
# TestSubprocessManager — command building, lifecycle
# ===========================================================================

class TestSubprocessManagerCommandBuild:
    """Test that config dicts produce correct CLI commands."""

    def _get_manager(self):
        from mission_control import SubprocessManager
        sio = MagicMock()
        return SubprocessManager(sio)

    def test_minimal_config(self):
        sm = self._get_manager()
        cmd = sm._build_command({})
        assert cmd[0] == sys.executable
        assert cmd[1] == 'rl_deploy.py'
        # Always includes --frame-dir for web UI streaming
        assert '--frame-dir' in cmd

    def test_demo_flag(self):
        sm = self._get_manager()
        cmd = sm._build_command({'demo': True})
        assert '--demo' in cmd

    def test_dry_run_flag(self):
        sm = self._get_manager()
        cmd = sm._build_command({'dry_run': True})
        assert '--dry-run' in cmd

    def test_max_speed(self):
        sm = self._get_manager()
        cmd = sm._build_command({'max_speed': 0.15})
        idx = cmd.index('--max-speed')
        assert cmd[idx + 1] == '0.15'

    def test_duration(self):
        sm = self._get_manager()
        cmd = sm._build_command({'duration': 120})
        idx = cmd.index('--duration')
        assert cmd[idx + 1] == '120'

    def test_distance(self):
        sm = self._get_manager()
        cmd = sm._build_command({'distance': 5.0})
        idx = cmd.index('--distance')
        assert cmd[idx + 1] == '5.0'

    def test_port(self):
        sm = self._get_manager()
        cmd = sm._build_command({'port': '/dev/ttyUSB0'})
        idx = cmd.index('--port')
        assert cmd[idx + 1] == '/dev/ttyUSB0'

    def test_no_detect(self):
        sm = self._get_manager()
        cmd = sm._build_command({'enable_detect': False})
        assert '--no-detect' in cmd

    def test_detect_default_true(self):
        """Detection enabled by default — should NOT have --no-detect."""
        sm = self._get_manager()
        cmd = sm._build_command({})
        assert '--no-detect' not in cmd

    def test_no_gimbal(self):
        sm = self._get_manager()
        cmd = sm._build_command({'enable_gimbal': False})
        assert '--no-gimbal' in cmd

    def test_face_mode(self):
        sm = self._get_manager()
        cmd = sm._build_command({'target_mode': 'face'})
        assert '--target' in cmd
        idx = cmd.index('--target')
        assert cmd[idx + 1] == 'face'

    def test_distance_mode_no_target_flag(self):
        sm = self._get_manager()
        cmd = sm._build_command({'target_mode': 'distance'})
        assert '--target' not in cmd

    def test_face_distance(self):
        sm = self._get_manager()
        cmd = sm._build_command({'face_distance': 1.5})
        idx = cmd.index('--face-distance')
        assert cmd[idx + 1] == '1.5'

    def test_segmentation(self):
        sm = self._get_manager()
        cmd = sm._build_command({'segmentation': True})
        assert '--segmentation' in cmd

    def test_seg_model(self):
        sm = self._get_manager()
        cmd = sm._build_command({'seg_model': '/path/to/model.pth'})
        idx = cmd.index('--seg-model')
        assert cmd[idx + 1] == '/path/to/model.pth'

    def test_log_rewards(self):
        sm = self._get_manager()
        cmd = sm._build_command({'log_rewards': True})
        assert '--log-rewards' in cmd

    def test_model_path(self):
        sm = self._get_manager()
        cmd = sm._build_command({'model': '/custom/model.zip'})
        idx = cmd.index('--model')
        assert cmd[idx + 1] == '/custom/model.zip'

    def test_full_config(self):
        """Test a realistic full config produces all expected flags."""
        sm = self._get_manager()
        cfg = {
            'model': 'my_model.zip',
            'demo': True,
            'dry_run': True,
            'max_speed': 0.2,
            'duration': 60,
            'distance': 3.0,
            'port': '/dev/ttyTHS1',
            'enable_detect': True,
            'enable_gimbal': False,
            'target_mode': 'face',
            'face_distance': 0.15,
            'segmentation': True,
            'seg_model': 'seg.pth',
            'log_rewards': True,
        }
        cmd = sm._build_command(cfg)
        assert '--demo' in cmd
        assert '--dry-run' in cmd
        assert '--no-gimbal' in cmd
        assert '--target' in cmd
        assert '--segmentation' in cmd
        assert '--log-rewards' in cmd
        assert '--no-detect' not in cmd  # detect is True

    def test_null_optional_fields_omitted(self):
        """None/empty optional fields should not produce flags."""
        sm = self._get_manager()
        cmd = sm._build_command({'duration': None, 'distance': None, 'model': ''})
        assert '--duration' not in cmd
        assert '--distance' not in cmd
        assert '--model' not in cmd


class TestSubprocessManagerLifecycle:
    """Test subprocess start/stop/e_stop behavior."""

    def _get_manager(self):
        from mission_control import SubprocessManager
        sio = MagicMock()
        return SubprocessManager(sio)

    def test_not_running_initially(self):
        sm = self._get_manager()
        assert sm.is_running is False

    def test_stop_when_not_running(self):
        """Stop should be safe when nothing is running."""
        sm = self._get_manager()
        sm.stop()  # should not raise

    def test_e_stop_when_not_running(self):
        sm = self._get_manager()
        sm.e_stop()  # should not raise

    def test_start_blocks_double_start(self):
        """Cannot start if already running."""
        sm = self._get_manager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        sm.proc = mock_proc
        result = sm.start({'demo': True})
        assert result is False

    def test_start_with_real_subprocess(self):
        """Actually launch a tiny subprocess and verify it runs."""
        sm = self._get_manager()
        # Launch a quick python command that exits immediately
        with patch.object(sm, '_build_command', return_value=[sys.executable, '-c', 'print("hello")']):
            ok = sm.start({})
        assert ok is True
        time.sleep(1)
        # Process should have exited
        assert sm.is_running is False

    def test_stop_kills_running_process(self):
        """Start a sleep process, then stop it."""
        sm = self._get_manager()
        with patch.object(sm, '_build_command', return_value=[sys.executable, '-c', 'import time; time.sleep(30)']):
            ok = sm.start({})
        assert ok is True
        assert sm.is_running is True
        sm.stop()
        time.sleep(0.5)
        assert sm.is_running is False

    def test_e_stop_kills_immediately(self):
        sm = self._get_manager()
        with patch.object(sm, '_build_command', return_value=[sys.executable, '-c', 'import time; time.sleep(30)']):
            sm.start({})
        assert sm.is_running is True
        sm.e_stop()
        assert sm.proc is None


class TestSubprocessManagerTelemetryParsing:
    """Test log line parsing for telemetry extraction."""

    def _get_manager(self):
        from mission_control import SubprocessManager
        sio = MagicMock()
        return SubprocessManager(sio)

    def test_parse_depth_line(self):
        sm = self._get_manager()
        line = "[42s] [CLEAR] raw=[+0.15,-0.30] -> lin=+0.150 ang=-0.450 depth=0.35m (L=0.45 R=0.28) dist=2.34m disp=1.89m hdg=45"
        sm._parse_telemetry(line)
        sm.sio.emit.assert_called()
        call_args = sm.sio.emit.call_args[0]
        assert call_args[0] == 'telemetry'
        telem = call_args[1]
        assert abs(telem['min_depth'] - 0.35) < 0.01
        assert abs(telem['left_depth'] - 0.45) < 0.01
        assert abs(telem['right_depth'] - 0.28) < 0.01
        assert abs(telem['speed'] - 0.15) < 0.01
        assert abs(telem['heading'] - 45) < 0.1
        assert abs(telem['distance'] - 2.34) < 0.01

    def test_parse_mode_emergency(self):
        sm = self._get_manager()
        sm._parse_telemetry("[EMERGENCY] Obstacle at 0.03m!")
        sm.sio.emit.assert_called()
        telem = sm.sio.emit.call_args[0][1]
        assert telem['mode'] == 'EMERGENCY'

    def test_parse_mode_avoid(self):
        sm = self._get_manager()
        sm._parse_telemetry("[AVOID] committed turn step 5/15")
        sm.sio.emit.assert_called()
        telem = sm.sio.emit.call_args[0][1]
        assert telem['mode'] == 'AVOID'

    def test_parse_mode_slow(self):
        sm = self._get_manager()
        sm._parse_telemetry("[SLOW] depth=0.18m speed_scale=0.4")
        sm.sio.emit.assert_called()

    def test_parse_face_state_search_spin(self):
        sm = self._get_manager()
        sm._parse_telemetry("[FACE] SEARCH_SPIN: rotating to find face")
        sm.sio.emit.assert_called()
        telem = sm.sio.emit.call_args[0][1]
        assert telem['face_state'] == 'SEARCH_SPIN'

    def test_parse_face_state_navigate(self):
        sm = self._get_manager()
        sm._parse_telemetry("[FACE] NAVIGATE: driving toward face at 15deg")
        sm.sio.emit.assert_called()
        telem = sm.sio.emit.call_args[0][1]
        assert telem['face_state'] == 'NAVIGATE'

    def test_parse_face_state_arrived(self):
        sm = self._get_manager()
        sm._parse_telemetry("[FACE] ARRIVED — target reached!")
        sm.sio.emit.assert_called()
        telem = sm.sio.emit.call_args[0][1]
        assert telem['face_state'] == 'ARRIVED'

    def test_parse_no_telemetry_in_generic_line(self):
        sm = self._get_manager()
        sm._parse_telemetry("[INFO] Loading model from: /path/model.zip")
        sm.sio.emit.assert_not_called()

    def test_parse_negative_speed(self):
        sm = self._get_manager()
        sm._parse_telemetry("lin=-0.100 ang=+0.300 depth=0.50m (L=0.60 R=0.40)")
        telem = sm.sio.emit.call_args[0][1]
        assert telem['speed'] == 0.1  # absolute value

    def test_parse_combined_depth_and_face(self):
        """A line with both depth telemetry and face state."""
        sm = self._get_manager()
        sm._parse_telemetry("[FACE] NAVIGATE depth=0.80m (L=1.00 R=0.60) lin=+0.100 hdg=30 dist=1.50m")
        telem = sm.sio.emit.call_args[0][1]
        assert 'min_depth' in telem
        assert telem['face_state'] == 'NAVIGATE'
        assert abs(telem['distance'] - 1.50) < 0.01


# ===========================================================================
# TestFlaskRoutes — HTTP endpoints
# ===========================================================================

class TestFlaskRoutes:
    """Test Flask HTTP routes return correct responses."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked globals."""
        import mission_control as mc
        mc.rover = DummyRoverHardware()
        mc.camera_mgr = CameraManager()
        mc.proc_mgr = mc.SubprocessManager(MagicMock())
        mc._app_config = {'segmentation': False, 'port': 5050}
        mc.app.config['TESTING'] = True
        return mc.app.test_client()

    def test_index_returns_200(self, client):
        resp = client.get('/')
        assert resp.status_code == 200

    def test_index_contains_title(self, client):
        resp = client.get('/')
        assert b'CRATER MISSION CONTROL' in resp.data

    def test_index_contains_tulane(self, client):
        resp = client.get('/')
        assert b'TULANE UNIVERSITY' in resp.data

    def test_index_contains_socketio_script(self, client):
        resp = client.get('/')
        assert b'socket.io' in resp.data

    def test_index_contains_wasd_keys(self, client):
        resp = client.get('/')
        assert b'key-w' in resp.data
        assert b'key-a' in resp.data
        assert b'key-s' in resp.data
        assert b'key-d' in resp.data

    def test_index_contains_config_form(self, client):
        resp = client.get('/')
        assert b'cfg-model' in resp.data
        assert b'cfg-speed' in resp.data
        assert b'cfg-target' in resp.data
        assert b'cfg-demo' in resp.data

    def test_index_contains_estop_button(self, client):
        resp = client.get('/')
        assert b'E-STOP' in resp.data
        assert b'btn-estop' in resp.data

    def test_index_contains_camera_feeds(self, client):
        resp = client.get('/')
        assert b'/stream/rgb' in resp.data          # MJPEG stream for RGB
        assert b'/stream/depth' in resp.data      # MJPEG stream for depth
        assert b'/snapshot/seg' in resp.data      # Seg uses polling (thread pool limit)

    def test_index_contains_telemetry_gauges(self, client):
        resp = client.get('/')
        assert b'g-speed' in resp.data
        assert b'g-heading' in resp.data
        assert b'g-depth' in resp.data
        assert b'g-fps' in resp.data
        assert b'g-mode' in resp.data
        assert b'g-face' in resp.data

    def test_index_contains_log_panel(self, client):
        resp = client.get('/')
        assert b'log-body' in resp.data
        assert b'ROVER LOG' in resp.data

    def test_api_status_returns_json(self, client):
        resp = client.get('/api/status')
        assert resp.status_code == 200
        import json
        data = json.loads(resp.data)
        assert 'camera_active' in data
        assert 'autonomous' in data
        assert 'hardware' in data

    def test_api_status_camera_inactive(self, client):
        import json
        resp = client.get('/api/status')
        data = json.loads(resp.data)
        assert data['camera_active'] is False

    def test_api_status_not_autonomous(self, client):
        import json
        resp = client.get('/api/status')
        data = json.loads(resp.data)
        assert data['autonomous'] is False

    def test_api_status_contains_telemetry_fields(self, client):
        import json
        resp = client.get('/api/status')
        data = json.loads(resp.data)
        assert 'fps' in data
        assert 'min_depth' in data
        assert 'detection' in data


# ===========================================================================
# TestRoverControlConstants — verify constants are reasonable
# ===========================================================================

class TestRoverControlConstants:
    """Verify safety constants from rover_control are sensible."""

    def test_max_linear_positive(self):
        assert MAX_LINEAR > 0

    def test_max_linear_reasonable(self):
        """Max linear should not exceed 1 m/s for safety."""
        assert MAX_LINEAR <= 1.0

    def test_max_angular_positive(self):
        assert MAX_ANGULAR > 0

    def test_max_angular_reasonable(self):
        """Max angular should not exceed 2 rad/s for safety."""
        assert MAX_ANGULAR <= 2.0

    def test_serial_port_is_string(self):
        assert isinstance(SERIAL_PORT, str)

    def test_baud_rate_standard(self):
        assert BAUD_RATE in [9600, 19200, 38400, 57600, 115200, 230400, 460800]


# ===========================================================================
# TestHTMLTemplate — verify critical UI elements in template
# ===========================================================================

class TestHTMLTemplate:
    """Verify the HTML template contains all required UI elements."""

    @pytest.fixture
    def template(self):
        from mission_control import HTML_TEMPLATE
        return HTML_TEMPLATE

    def test_has_css_variables(self, template):
        assert '--bg:' in template
        assert '--accent:' in template
        assert '--green:' in template
        assert '--red:' in template

    def test_tulane_green_color(self, template):
        assert '#006747' in template

    def test_tulane_sky_blue_color(self, template):
        assert '#87CEEB' in template

    def test_dark_background_color(self, template):
        assert '#1B3A2D' in template

    def test_has_socketio_connection(self, template):
        assert "io()" in template

    def test_has_manual_drive_events(self, template):
        assert "manual_drive" in template
        assert "manual_stop" in template

    def test_has_mission_events(self, template):
        assert "start_mission" in template
        assert "stop_mission" in template
        assert "e_stop" in template

    def test_has_gimbal_events(self, template):
        assert "gimbal_move" in template

    def test_has_lights_events(self, template):
        assert "lights_set" in template

    def test_has_keyboard_handling(self, template):
        assert "keydown" in template
        assert "keyup" in template

    def test_has_escape_estop(self, template):
        assert "Escape" in template

    def test_has_autoscroll_toggle(self, template):
        assert "log-autoscroll" in template

    def test_has_face_options_toggle(self, template):
        assert "toggleFaceOpts" in template
        assert "face-opts" in template

    def test_has_config_getter(self, template):
        assert "getConfig" in template

    def test_has_log_coloring(self, template):
        assert "log-error" in template
        assert "log-warn" in template
        assert "log-ok" in template
        assert "log-info" in template

    def test_has_telemetry_handler(self, template):
        assert "socket.on('telemetry'" in template

    def test_has_mission_state_handler(self, template):
        assert "socket.on('mission_state'" in template

    def test_has_autonomous_overlay(self, template):
        assert "auto-overlay" in template
        assert "AUTONOMOUS MODE ACTIVE" in template

    def test_has_grid_layout(self, template):
        assert "grid-template-areas" in template
        assert "header" in template
        assert "left" in template
        assert "center" in template
        assert "right" in template
        assert "log" in template


# ===========================================================================
# TestModuleSeparation — verify modules don't have unwanted coupling
# ===========================================================================

class TestModuleSeparation:
    """Verify the 3-module architecture stays clean."""

    def test_rover_control_does_not_import_flask(self):
        """rover_control.py should have zero web dependencies."""
        import rover_control
        import inspect
        source = inspect.getsource(rover_control)
        assert 'flask' not in source.lower().split('from')[0] if 'from' in source else True
        assert 'import flask' not in source
        assert 'from flask' not in source

    def test_rover_control_does_not_import_socketio(self):
        import rover_control
        import inspect
        source = inspect.getsource(rover_control)
        assert 'socketio' not in source

    def test_mission_control_uses_rover_control(self):
        """mission_control.py should import from rover_control."""
        import mission_control
        import inspect
        source = inspect.getsource(mission_control)
        assert 'from rover_control import' in source

    def test_mission_control_does_not_import_base_ctrl(self):
        """mission_control.py should NOT directly import base_ctrl."""
        import mission_control
        import inspect
        source = inspect.getsource(mission_control)
        assert 'from base_ctrl' not in source
        assert 'import base_ctrl' not in source

    def test_rl_deploy_not_modified(self):
        """rl_deploy.py should not import from mission_control or rover_control."""
        import rl_deploy
        import inspect
        source = inspect.getsource(rl_deploy)
        assert 'mission_control' not in source
        assert 'rover_control' not in source
