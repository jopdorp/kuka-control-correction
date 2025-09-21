#!/usr/bin/env python3
"""
Simple webcam test script for ArUco detection using our tuned parameters.
Shows live detection, marker IDs, and camera poses.
"""
import sys
import os
import cv2
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'raspberry-pi', 'src'))

from aruco_detector import ArucoDetector


def main():
    print("Starting ArUco webcam test...")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Initialize detector with tuned parameters
    detector = ArucoDetector(
        dictionary_type=cv2.aruco.DICT_6X6_1000,
        marker_size=0.02,  # 20mm markers
    )
    
    # Add basic camera calibration for pose estimation
    # These are rough estimates for a typical webcam - replace with actual calibration
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240], 
        [0, 0, 1]
    ], dtype=np.float32)
    distortion_coeffs = np.zeros(5, dtype=np.float32)  # Assume no distortion
    
    detector.camera_matrix = camera_matrix
    detector.distortion_coeffs = distortion_coeffs
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    fps_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Detect markers
        detected_markers = detector.detect_markers(frame)
        
        # Draw markers on frame
        display_frame = detector.draw_detected_markers(frame, detected_markers, draw_axes=False)
        
        # Add info overlay
        info_text = []
        info_text.append(f"Markers detected: {len(detected_markers)}")
        
        if detected_markers:
            marker_ids = [m.marker_id for m in detected_markers]
            info_text.append(f"IDs: {marker_ids}")
            
            # Try to estimate camera pose (if calibration was available)
            cam_pose = detector.estimate_camera_pose(detected_markers)
            if cam_pose:
                info_text.append(f"Cam pos: [{cam_pose.translation[0]:.3f}, {cam_pose.translation[1]:.3f}, {cam_pose.translation[2]:.3f}]")
                info_text.append(f"Confidence: {cam_pose.confidence:.2f}")
                info_text.append(f"Used {cam_pose.num_markers_used} markers")
            else:
                info_text.append("No camera calibration - pose estimation disabled")
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
            info_text.append(f"FPS: {fps:.1f}")
        
        # Draw info text
        y_offset = 30
        for text in info_text:
            cv2.putText(display_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Show frame
        cv2.imshow('ArUco Detection Test', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"aruco_test_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Saved frame as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == "__main__":
    main()
