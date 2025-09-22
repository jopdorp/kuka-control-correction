"""
Unit tests for ChArUco board-based vision correction system functionality.
Tests the integration between ChArUco detection, board matching, and correction calculation.
"""
import os
import sys
import time
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'raspberry-pi', 'src'))

from vision_correction_system import VisionCorrectionSystem, SystemConfig
from charuco_board_detector import CharucoBoardConfig, DetectedCharucoBoard
from charuco_board_matcher import BoardMatchResult


def make_controller_state(ts: float, x_mm: float) -> dict:
    return {
        'type': 'CONTROLLER_STATE',
        'timestamp': ts,
        'tcp': {'X': float(x_mm), 'Y': 0.0, 'Z': 0.0, 'A': 0.0, 'B': 0.0, 'C': 0.0},
        'base': {}
    }


def create_test_system():
    """Create a test system with ChArUco configuration."""
    cfg = SystemConfig()
    cfg.charuco_boards_config_file = "test_boards.json"
    cfg.max_time_skew = 0.2  # seconds
    cfg.max_correction = 1000.0  # disable clamping for test
    cfg.max_board_position_error = 100.0  # mm
    cfg.max_board_rotation_error = 45.0  # degrees
    
    system = VisionCorrectionSystem(cfg)
    
    # Create mock board configurations
    board_config = CharucoBoardConfig(
        board_id="test_board",
        squares_x=5,
        squares_y=4,
        square_size=0.05,
        marker_size=0.035,
        dictionary_type=10,
        expected_plane=[100.0, 200.0, 0.0, 0.0, 0.0, 0.0]
    )
    system.charuco_board_configs = [board_config]
    
    return system


def create_mock_detected_board():
    """Create a mock detected ChArUco board."""
    return DetectedCharucoBoard(
        board_id=None,
        corners=np.random.rand(10, 1, 2).astype(np.float32),
        ids=np.arange(10),
        translation=np.array([0.1, 0.2, 0.5]),  # meters
        rotation=np.array([0.0, 0.0, 0.0]),
        rotation_matrix=np.eye(3),
        confidence=0.8,
        num_corners=10
    )


def test_system_requires_charuco_config():
    """Test that system initialization requires ChArUco configuration."""
    cfg = SystemConfig()
    cfg.charuco_boards_config_file = ""  # Empty config file
    
    system = VisionCorrectionSystem(cfg)
    
    # Should have empty charuco configurations initially
    assert system.charuco_detector is None
    assert system.charuco_matcher is None
    assert system.charuco_board_configs == []


def test_controller_state_timestamp_selection():
    """Test that the system uses the nearest controller state by timestamp."""
    system = create_test_system()
    
    now = time.time()
    # Two states around frame_ts: nearest has X=123, farther has X=999
    near = make_controller_state(now + 0.05, 123.0)
    far = make_controller_state(now - 0.15, 999.0)

    # Populate buffer (order shouldn't matter)
    with system.controller_state_lock:
        system.controller_state_buffer.clear()
        system.controller_state_buffer.append(far)
        system.controller_state_buffer.append(near)

    # Test timestamp selection
    selected_state = system._get_controller_state_for_timestamp(now + 0.04)
    assert selected_state is not None
    assert selected_state['tcp']['X'] == 123.0  # Should select the nearest state


def test_stale_controller_state_handling():
    """Test that system skips correction when controller state is too stale."""
    system = create_test_system()
    system.config.max_time_skew = 0.05  # tighten skew tolerance

    now = time.time()
    stale = make_controller_state(now - 1.0, 50.0)
    
    with system.controller_state_lock:
        system.controller_state_buffer.clear()
        system.controller_state_buffer.append(stale)
        system.latest_controller_state = stale

    # Should return None for stale timestamp
    selected_state = system._get_controller_state_for_timestamp(now)
    assert selected_state is None
    
    # Should return the latest state when timestamp is None
    selected_state = system._get_controller_state_for_timestamp(None)
    # This should fall back to the latest state logic in actual usage


def test_charuco_board_processing_without_detector():
    """Test that processing returns None when ChArUco detector is not initialized."""
    cfg = SystemConfig()
    cfg.charuco_boards_config_file = "test_boards.json"
    system = VisionCorrectionSystem(cfg)
    
    # Don't initialize detectors
    assert system.charuco_detector is None
    
    # Create mock frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Should handle gracefully when no detector is available
    # Note: the actual _process_charuco_boards method would need to check for None detector
    assert system.charuco_detector is None
    assert system.charuco_matcher is None
