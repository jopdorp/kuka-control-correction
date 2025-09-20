"""
Unit tests for timestamp-aligned controller state selection in VisionCorrectionSystem.
Avoids threads; calls private methods directly.
"""
import os
import sys
import time
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'raspberry-pi', 'src'))

from vision_correction_system import VisionCorrectionSystem, SystemConfig


class _StubCameraPose:
    def __init__(self, translation_m, rotation_matrix=np.eye(3), confidence=1.0):
        self.translation = np.asarray(translation_m, dtype=float)  # meters
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=float)
        self.confidence = float(confidence)


def make_controller_state(ts: float, x_mm: float) -> dict:
    return {
        'type': 'CONTROLLER_STATE',
        'timestamp': ts,
        'tcp': {'X': float(x_mm), 'Y': 0.0, 'Z': 0.0, 'A': 0.0, 'B': 0.0, 'C': 0.0},
        'base': {}
    }


def test_uses_nearest_controller_state_by_timestamp():
    cfg = SystemConfig()
    cfg.max_time_skew = 0.2  # seconds
    cfg.max_correction = 1000.0  # disable clamping for test
    system = VisionCorrectionSystem(cfg)

    # Stub camera pose: at origin (0 m) so correction equals -expected (mm)
    system.aruco_detector.estimate_camera_pose = lambda detected: _StubCameraPose([0.0, 0.0, 0.0])

    now = time.time()
    # Two states around frame_ts: nearest has X=123, farther has X=999
    near = make_controller_state(now + 0.05, 123.0)
    far = make_controller_state(now - 0.15, 999.0)

    # Populate buffer (order shouldn't matter)
    with system.controller_state_lock:
        system.controller_state_buffer.clear()
        system.controller_state_buffer.append(far)
        system.controller_state_buffer.append(near)

    # Run correction with frame_ts close to 'near'
    corr = system._calculate_base_correction(detected_markers=[1, 2, 3], frame_ts=now + 0.04)
    assert corr is not None
    # Expect translation_correction[0] == -near.X (actual 0mm - expected 123mm)
    assert np.isclose(corr.translation_correction[0], -123.0)


def test_skips_when_state_is_stale():
    cfg = SystemConfig()
    cfg.max_time_skew = 0.05  # tighten skew
    cfg.max_correction = 1000.0  # disable clamping for test
    system = VisionCorrectionSystem(cfg)

    # Stub camera pose: anything valid
    system.aruco_detector.estimate_camera_pose = lambda detected: _StubCameraPose([0.01, 0.0, 0.0])

    now = time.time()
    stale = make_controller_state(now - 1.0, 50.0)
    with system.controller_state_lock:
        system.controller_state_buffer.clear()
        system.controller_state_buffer.append(stale)
        system.latest_controller_state = stale

    # Frame timestamp far from state -> should skip
    corr = system._calculate_base_correction(detected_markers=[42], frame_ts=now)
    assert corr is None

    # If frame_ts None -> fallback to latest allowed (no skew check)
    corr2 = system._calculate_base_correction(detected_markers=[42], frame_ts=None)
    assert corr2 is not None
    # Expected correction = actual_mm (10mm) - expected_mm (50mm) = -40mm
    assert np.isclose(corr2.translation_correction[0], -40.0)
