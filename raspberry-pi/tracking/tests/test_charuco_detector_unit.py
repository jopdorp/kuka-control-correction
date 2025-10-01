"""
Unit tests for CharucoBoardDetector with minimal mocking. Skips if SciPy is not available.
"""
import os
import sys
import numpy as np
import pytest

# Ensure correction is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'correction'))

scipy_available = True
try:
    from scipy.spatial.transform import Rotation as R  # type: ignore
except Exception:
    scipy_available = False

cv2_available = True
try:
    import cv2
except Exception:
    cv2_available = False

from charuco_board_detector import CharucoBoardDetector, CharucoBoardConfig, DetectedCharucoBoard  # type: ignore

pytestmark = pytest.mark.skipif(not scipy_available or not cv2_available, reason="SciPy or CV2 not installed")


def test_detector_initialization():
    """Test that detector initializes correctly with board configurations."""
    board_config = CharucoBoardConfig(
        squares_x=5,
        squares_y=4,
        square_size=0.05,
        marker_size=0.035,
        dictionary_type=10,  # cv2.aruco.DICT_6X6_250
        expected_plane=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    distortion_coeffs = np.zeros(5, dtype=np.float32)
    
    detector = CharucoBoardDetector(
        board_configs=[board_config],
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs
    )
    
    assert len(detector.charuco_boards) == 1
    assert len(detector.charuco_detectors) == 1


def test_detector_detect_boards_without_calibration():
    """Test that detector returns empty list without camera calibration."""
    board_config = CharucoBoardConfig(
        squares_x=5,
        squares_y=4,
        square_size=0.05,
        marker_size=0.035,
        dictionary_type=10,
        expected_plane=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    
    detector = CharucoBoardDetector(
        board_configs=[board_config],
        camera_matrix=None,
        distortion_coeffs=None
    )
    
    # Create test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Should return empty list without calibration
    detected_boards = detector.detect_boards(test_image)
    assert detected_boards == []


def test_detected_charuco_board_structure():
    """Test the structure of DetectedCharucoBoard."""
    # Create mock data for detected board
    corners = np.random.rand(10, 1, 2).astype(np.float32)
    ids = np.arange(10)
    translation = np.array([0.1, 0.2, 0.5])
    rotation = np.array([0.1, 0.2, 0.3])
    rotation_matrix = np.eye(3)
    
    board = DetectedCharucoBoard(
        corners=corners,
        ids=ids,
        translation=translation,
        rotation=rotation,
        rotation_matrix=rotation_matrix,
        confidence=0.8,
        num_corners=10,
    )
    
    assert board.corners.shape == (10, 1, 2)
    assert len(board.ids) == 10
    assert board.translation.shape == (3,)
    assert board.rotation.shape == (3,)
    assert board.rotation_matrix.shape == (3, 3)
    assert board.confidence == 0.8
    assert board.num_corners == 10
