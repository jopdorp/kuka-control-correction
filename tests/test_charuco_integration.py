#!/usr/bin/env python3
"""
Simple integration test for CharucoDetector to verify the migration from ArucoDetector.
"""
import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'raspberry-pi', 'src'))

from charuco_board_detector import CharucoBoardDetector, CharucoBoardConfig


def test_charuco_detector():
    """Test that CharucoDetector can be created and configured correctly."""
    print("Testing CharucoDetector creation...")
    
    # Create a simple board configuration
    board_config = CharucoBoardConfig(
        board_id="test_board",
        squares_x=5,
        squares_y=4,
        square_size=0.05,  # 50mm squares
        marker_size=0.035,  # 35mm markers
        dictionary_type=cv2.aruco.DICT_6X6_250,
        expected_plane=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Origin plane
    )
    
    # Initialize detector
    detector = CharucoBoardDetector(
        board_configs=[board_config],
        min_corners_for_pose=6
    )
    
    print(f"✓ CharucoDetector created successfully with {len(detector.board_configs)} board(s)")
    
    # Test with a dummy camera calibration
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    distortion_coeffs = np.zeros(5, dtype=np.float32)
    
    detector.camera_matrix = camera_matrix
    detector.distortion_coeffs = distortion_coeffs
    
    print("✓ Camera calibration set successfully")
    
    # Test detection on a dummy image (should return empty list)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    detected_boards = detector.detect_boards(dummy_image)
    
    print(f"✓ Detection test completed, found {len(detected_boards)} boards (expected 0 on black image)")
    
    # Test drawing function
    output_image = detector.draw_detected_boards(dummy_image, detected_boards)
    
    print("✓ Drawing function completed successfully")
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_charuco_detector()
