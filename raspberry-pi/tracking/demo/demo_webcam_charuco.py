#!/usr/bin/env python3
"""
Simple webcam test script for ChArUco board detection.

Stable, no fallbacks: board parameters are inferred from charuco_board.png
and dictionary type is fixed to DICT_6X6_250. Camera calibration is required.
"""
import sys
import os
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'raspberry-pi', 'src'))

from charuco_board_detector import CharucoBoardDetector, CharucoBoardConfig
from charuco_board_inference import infer_charuco_board_config_from_image


def main():
    print("Starting ChArUco board webcam test...")
    print("Press 'q' to quit, 's' to save current frame")

    # Derive a single board configuration from the repo image
    board_image = os.path.join(os.path.dirname(__file__), 'charuco_board.png')
    board_config = infer_charuco_board_config_from_image(board_image)
    print(f"Inferred ChArUco grid from image {os.path.basename(board_image)}: "
          f"{board_config.squares_x}x{board_config.squares_y}")
    
    # Initialize detector
    detector = CharucoBoardDetector(board_configs=[board_config], min_corners_for_pose=6)
    
    cap = cv2.VideoCapture(0)
    
    # Set MJPG format for better frame rates
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Set resolution
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Set frame rate
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Load camera calibration (required; no fallbacks)
    calibration_file = "camera_calibration.npz"
    if os.path.exists(calibration_file):
        print(f"Loading camera calibration from {calibration_file}...")
        try:
            calib_data = np.load(calibration_file)
            camera_matrix = calib_data['camera_matrix']
            distortion_coeffs = calib_data['distortion_coeffs']
            calibration_error = float(calib_data['calibration_error'])
            print(f"Camera calibration loaded successfully!")
            print(f"Reprojection error: {calibration_error:.3f} pixels")
            print(f"Focal lengths: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
            print(f"Principal point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
            print(f"Distortion coeffs: {distortion_coeffs.flatten()}")
            
            # Set calibration in detector
            detector.camera_matrix = camera_matrix
            detector.distortion_coeffs = distortion_coeffs
        except Exception as e:
            print(f"Failed to load camera calibration: {e}")
            print("Camera calibration is required. Exiting.")
            cap.release()
            return
    else:
        print(f"No camera calibration found at {calibration_file}")
        print("Camera calibration is required. Exiting.")
        cap.release()
        return

    frame_count = 0
    fps_start = time.time()
    
    # Initialize persistent board pose for stable display
    last_board_pose = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Detect ChArUco boards
        detected_boards = detector.detect_boards(frame)

        # Draw boards on frame
        display_frame = detector.draw_detected_boards(frame, detected_boards)

        # Add info overlay
        info_text = []
        info_text.append(f"Boards detected: {len(detected_boards)}")
        info_text.append("Keys: q=quit, s=save")

        # Get board pose information
        if detected_boards:
            board = detected_boards[0]  # Use first detected board
            last_board_pose = board

            info_text.append(f"Corners: {board.num_corners}")
            info_text.append(
                f"Board pos: [{board.translation[0]:.3f}, {board.translation[1]:.3f}, {board.translation[2]:.3f}]"
            )

            # Convert rotation vector to Euler angles (in degrees) for display
            euler_angles = R.from_rotvec(board.rotation).as_euler('xyz', degrees=True)
            info_text.append(
                f"Board rot: [{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}]"
            )

            info_text.append(f"Confidence: {board.confidence:.2f}")
        elif last_board_pose:
            # Show last known pose when board not detected
            info_text.append(f"Last corners: {last_board_pose.num_corners}")
            info_text.append(
                f"Last pos: [{last_board_pose.translation[0]:.3f}, {last_board_pose.translation[1]:.3f}, {last_board_pose.translation[2]:.3f}]"
            )
            euler_angles = R.from_rotvec(last_board_pose.rotation).as_euler('xyz', degrees=True)
            info_text.append(
                f"Last rot: [{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}]"
            )
            info_text.append(f"Last confidence: {last_board_pose.confidence:.2f}")
        else:
            info_text.append("Board pos: [No board detected]")
            info_text.append("Board rot: [No rotation available]")

        # Calculate FPS
        frame_count += 1
        fps = 1 / (time.time() - fps_start)
        fps_start = time.time()
        info_text.append(f"FPS: {fps:.1f}")

        # Draw info text
        y_offset = 30
        for text in info_text:
            cv2.putText(display_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

        # Show frame
        cv2.imshow('ChArUco Detection Test', display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            filename = f"charuco_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == "__main__":
    main()
