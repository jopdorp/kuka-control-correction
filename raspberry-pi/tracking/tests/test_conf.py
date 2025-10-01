"""
Test configuration and shared fixtures for the KUKA vision correction system tests.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock
import sys
import os

# Add correction directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'correction'))

@pytest.fixture
def mock_camera():
    """Mock camera that returns test frames."""
    camera = Mock()
    camera.isOpened.return_value = True
    camera.read.return_value = (True, create_test_frame())
    camera.release = Mock()
    return camera

@pytest.fixture 
def sample_aruco_dict():
    """Sample ArUco dictionary for testing."""
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

@pytest.fixture
def sample_camera_matrix():
    """Sample camera calibration matrix."""
    return np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0], 
        [0.0, 0.0, 1.0]
    ])

@pytest.fixture
def sample_distortion_coeffs():
    """Sample camera distortion coefficients."""
    return np.array([0.1, -0.2, 0.001, 0.002, 0.1])

@pytest.fixture
def sample_marker_positions():
    """Sample marker positions for testing."""
    return {
        23: {
            'position': [100.0, 200.0, 0.0],
            'orientation': [0.0, 0.0, 0.0]
        },
        42: {
            'position': [300.0, 400.0, 50.0],
            'orientation': [0.0, 0.0, 90.0]
        }
    }

@pytest.fixture
def sample_system_config():
    """Sample system configuration for testing."""
    from vision_correction_system import SystemConfig
    
    config = SystemConfig()
    config.camera_id = 0
    config.camera_width = 640
    config.camera_height = 480
    config.camera_fps = 30
    config.controller_ip = "192.168.1.50"
    config.controller_port = 7001
    config.marker_size = 0.05  # 50mm markers
    config.position_threshold = 0.1
    config.rotation_threshold = 0.01
    config.correction_rate = 20
    
    return config

@pytest.fixture
def sample_tool_to_camera_transform():
    """Sample tool-to-camera transformation matrix."""
    # Camera mounted 100mm in X direction from tool center
    return np.array([
        [1.0, 0.0, 0.0, 100.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def create_test_frame(width=640, height=480):
    """Create a test frame with ArUco markers."""
    # Create a blank image
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some test markers using simple rectangles as placeholders
    # In real tests, we'd generate actual ArUco markers
    cv2.rectangle(frame, (100, 100), (150, 150), (0, 0, 0), -1)  # Marker 23
    cv2.rectangle(frame, (300, 200), (350, 250), (0, 0, 0), -1)  # Marker 42
    
    return frame

@pytest.fixture
def mock_tcp_socket():
    """Mock TCP socket for network testing."""
    socket_mock = Mock()
    socket_mock.connect = Mock()
    socket_mock.sendall = Mock()
    socket_mock.recv.return_value = b"OK"
    socket_mock.close = Mock()
    socket_mock.shutdown = Mock()
    return socket_mock

@pytest.fixture
def sample_correction_data():
    """Sample correction data for testing."""
    from vision_correction_system import CorrectionData
    import time
    
    return CorrectionData(
        timestamp=time.time(),
        sequence_id=123,
        translation=(1.0, -0.5, 0.2),  # mm
        rotation=(0.001, 0.002, -0.001),  # radians
        confidence=0.95,
        marker_count=2
    )
