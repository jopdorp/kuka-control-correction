#!/usr/bin/env python3
"""
Test script for multiple ChArUco board detection system.

This script demonstrates the new functionality to detect and match
multiple ChArUco boards simultaneously.
"""

import sys
import os
import numpy as np
import cv2
import json
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'raspberry-pi', 'src'))

from vision_correction_system import VisionCorrectionSystem, SystemConfig
from charuco_board_detector import CharucoBoardDetector, CharucoBoardConfig
from charuco_board_matcher import CharucoBoardMatcher

def create_test_config():
    """Create a test configuration for ChArUco boards."""
    config = SystemConfig()
    config.charuco_boards_config_file = "charuco_boards_config.json"
    config.camera_index = 0
    config.max_board_position_error = 100.0  # mm
    config.max_board_rotation_error = 45.0   # degrees
    return config

def generate_test_charuco_boards():
    """Generate test ChArUco board images for testing."""
    # Load board configurations
    with open("charuco_boards_config.json", 'r') as f:
        boards_data = json.load(f)
    
    for i, board_data in enumerate(boards_data['boards']):
        # Create ArUco dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(board_data['dictionary_type'])
        
        # Create ChArUco board
        board = cv2.aruco.CharucoBoard(
            (board_data['squares_x'], board_data['squares_y']),
            board_data['square_size'],
            board_data['marker_size'],
            dictionary
        )
        
        # Generate board image
        image_size = (800, 600)  # Fixed size for testing
        board_image = board.generateImage(image_size)
        
        # Save board image
        filename = f"test_charuco_board_{board_data['board_id']}.png"
        cv2.imwrite(filename, board_image)
        print(f"Generated {filename}")

def test_board_detection():
    """Test ChArUco board detection without camera."""
    print("Testing ChArUco board detection system...")
    
    # Create test configuration
    config = create_test_config()
    
    # Initialize vision system
    system = VisionCorrectionSystem(config)
    
    # Load configurations (mock camera calibration)
    try:
        # Create mock camera calibration if it doesn't exist
        if not os.path.exists("camera_calibration.npz"):
            print("Creating mock camera calibration...")
            camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
            distortion_coeffs = np.zeros(5, dtype=np.float32)
            np.savez("camera_calibration.npz", 
                    camera_matrix=camera_matrix, 
                    distortion_coeffs=distortion_coeffs)
        
        # Create mock robot config file
        if not os.path.exists("robot_config.json"):
            print("Creating mock robot config...")
            robot_data = {
                "camera_tool_offset": {
                    "translation": [0, 0, 100],
                    "rotation": [0, 0, 0]
                }
            }
            with open("robot_config.json", 'w') as f:
                json.dump(robot_data, f)
        
        # Load configurations
        if system.load_configuration_files(
            "camera_calibration.npz",
            "robot_config.json"
        ):
            print("✓ Configuration loaded successfully")
            print(f"✓ Loaded {len(system.charuco_board_configs)} ChArUco board configurations")
            
            # Print board configurations
            for config in system.charuco_board_configs:
                print(f"  - Board '{config.board_id}': {config.squares_x}×{config.squares_y} squares, "
                      f"expected at {config.expected_plane}")
        else:
            print("✗ Failed to load configuration")
            return False
            
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    print("✓ ChArUco board detection system initialized successfully")
    return True

def test_board_matching():
    """Test board matching algorithm."""
    print("\nTesting board matching algorithm...")
    
    # Load board configurations
    with open("charuco_boards_config.json", 'r') as f:
        boards_data = json.load(f)
    
    # Create board configs
    board_configs = []
    for board_data in boards_data['boards']:
        config = CharucoBoardConfig(
            board_id=board_data['board_id'],
            squares_x=board_data['squares_x'],
            squares_y=board_data['squares_y'],
            square_size=board_data['square_size'],
            marker_size=board_data['marker_size'],
            dictionary_type=board_data['dictionary_type'],
            expected_plane=board_data['expected_plane']
        )
        board_configs.append(config)
    
    # Create matcher
    matcher = CharucoBoardMatcher(board_configs)
    print(f"✓ Board matcher initialized for {len(board_configs)} boards")
    
    return True

def main():
    """Main test function."""
    print("ChArUco Multiple Board Detection Test")
    print("=" * 40)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate test images
    print("Generating test ChArUco board images...")
    try:
        generate_test_charuco_boards()
        print("✓ Test board images generated")
    except Exception as e:
        print(f"✗ Failed to generate test images: {e}")
        return
    
    # Test detection system
    if not test_board_detection():
        return
    
    # Test matching algorithm
    if not test_board_matching():
        return
    
    print("\n" + "=" * 40)
    print("✓ All tests completed successfully!")
    print("\nNext steps:")
    print("1. Print the generated board images")
    print("2. Use with actual camera and robot system")
    print("3. Configure expected board positions in charuco_boards_config.json")

if __name__ == "__main__":
    main()