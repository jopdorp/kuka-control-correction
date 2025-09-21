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


def calibrate_from_photo(detector, photo_path):
    """Extract marker positions from a reference photo."""
    print(f"Loading calibration photo: {photo_path}")
    img = cv2.imread(photo_path)
    if img is None:
        print("Could not load photo")
        return None
    
    # Detect markers in photo
    markers = detector.detect_markers(img)
    if len(markers) < 2:
        print("Need at least 2 markers in calibration photo")
        return None
    
    print(f"Found {len(markers)} markers in photo")
    
    # Set up world coordinates: assume photo is taken from above, Z=0 plane
    # Place first marker at origin, arrange others relative to it
    marker_positions = {}
    
    if markers:
        # Sort by marker ID for consistency
        markers.sort(key=lambda m: m.marker_id)
        
        # Get marker centers and sizes in image
        centers = []
        marker_sizes_px = []
        for m in markers:
            center = m.corners.mean(axis=0)
            centers.append(center)
            
            # Calculate marker size in pixels (width of marker)
            width = np.linalg.norm(m.corners[1] - m.corners[0])
            height = np.linalg.norm(m.corners[3] - m.corners[0])
            marker_size_px = (width + height) / 2  # Average of width and height
            marker_sizes_px.append(marker_size_px)
            
            print(f"Marker {m.marker_id} at image pos: [{center[0]:.1f}, {center[1]:.1f}], size: {marker_size_px:.1f}px")
        
        # Calculate scale using actual marker size (20mm = 0.02m)
        avg_marker_size_px = np.mean(marker_sizes_px)
        marker_size_real = 0.02  # 2cm in meters
        pixels_per_meter = avg_marker_size_px / marker_size_real
        print(f"Calibrated scale: {pixels_per_meter:.1f} pixels per meter")
        print(f"Average marker size in image: {avg_marker_size_px:.1f} pixels")
        
        # Set first marker as origin
        origin_center = centers[0]
        print(f"Using marker {markers[0].marker_id} as origin at {origin_center}")
        
        for i, marker in enumerate(markers):
            center = centers[i]
            # Convert to world coordinates using marker size calibration
            x = (center[0] - origin_center[0]) / pixels_per_meter
            y = (center[1] - origin_center[1]) / pixels_per_meter
            z = 0.0  # Assume flat surface
            
            marker_positions[marker.marker_id] = [x, y, z]
            
            # Calculate distance from origin for validation
            distance = np.sqrt(x*x + y*y)
            print(f"Marker {marker.marker_id} world pos: [{x:.3f}, {y:.3f}, {z:.3f}] (dist: {distance*100:.1f}cm)")
        
        # Validate distances - markers should be reasonable distances apart
        distances = []
        marker_ids = list(marker_positions.keys())
        for i in range(len(marker_ids)):
            for j in range(i+1, len(marker_ids)):
                pos1 = np.array(marker_positions[marker_ids[i]])
                pos2 = np.array(marker_positions[marker_ids[j]])
                dist = np.linalg.norm(pos1 - pos2)
                distances.append(dist)
                print(f"Distance between markers {marker_ids[i]} and {marker_ids[j]}: {dist*100:.1f}cm")
        
        min_dist = min(distances) * 100 if distances else 0
        max_dist = max(distances) * 100 if distances else 0
        print(f"Distance range: {min_dist:.1f}cm to {max_dist:.1f}cm")
        
        # Sanity check: distances should be reasonable (1cm to 50cm)
        if min_dist < 1 or max_dist > 50:
            print(f"WARNING: Unusual distances detected. This might cause PnP issues.")
            print(f"Expected marker distances: 2-10cm typically")
    
    return marker_positions


def main():
    print("Starting ArUco webcam test...")
    print("Press 'q' to quit, 's' to save current frame, 'c' to calibrate from saved frame")
    
    # Initialize detector with tuned parameters
    detector = ArucoDetector(
        dictionary_type=cv2.aruco.DICT_6X6_250,
        marker_size=0.02,  # 20mm markers
        max_reprojection_error=8.0,  # pixels
    )
    
    cap = cv2.VideoCapture(0)
    # Try to open webcam first to get actual capabilities
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    

    # Set resolution
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Change this to desired width
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Change this to desired height
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Actual resolution: {width}x{height}")
    
    fx, fy = 800, 800  # Base focal lengths
    
    cx = width / 2.0   # Principal point at center
    cy = height / 2.0
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy], 
        [0, 0, 1]
    ], dtype=np.float32)
    distortion_coeffs = np.zeros(5, dtype=np.float32)  # Assume no distortion
    
    detector.camera_matrix = camera_matrix
    detector.distortion_coeffs = distortion_coeffs
    
    # Try to load marker positions from a saved calibration
    calibration_file = "marker_calibration.jpg"
    if os.path.exists(calibration_file):
        print(f"Found {calibration_file}, loading marker positions...")
        marker_positions = calibrate_from_photo(detector, calibration_file)
        if marker_positions:
            detector.load_marker_positions(marker_positions)
            print("Marker positions loaded successfully!")
        else:
            print("Failed to load marker positions from photo")
            print("The existing calibration file might be incompatible.")
            print("Press 'c' during live view to recalibrate with current camera.")
    else:
        print(f"No {calibration_file} found. Save a frame with 'c' to calibrate.")

    frame_count = 0
    fps_start = time.time()
    
    # Initialize persistent camera pose for stable display
    last_cam_pose = None
    
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
        info_text.append("Keys: q=quit, s=save, c=calibrate")
        
        # Try to estimate camera pose if markers are detected
        marker_ids = [m.marker_id for m in detected_markers or []]
        info_text.append(f"IDs: {marker_ids}")
        
        # Try to estimate camera pose
        cam_pose = detector.estimate_camera_pose(detected_markers)
        
        if cam_pose:
            # Update persistent pose with new estimate
            last_cam_pose = cam_pose
        # Debug why pose estimation failed
        if hasattr(detector, 'marker_positions') and detector.marker_positions:
            known_markers = [m for m in detected_markers if m.marker_id in detector.marker_positions]
            info_text.append(f"Known markers: {len(known_markers)}/{len(detected_markers)}")
        else:
            info_text.append("No marker positions loaded - press 'c' to calibrate")
        
        # Always display camera pose info if we have it (even when no markers detected)
        if last_cam_pose:
            info_text.append(f"Cam pos: [{last_cam_pose.translation[0]:.3f}, {last_cam_pose.translation[1]:.3f}, {last_cam_pose.translation[2]:.3f}]")
            info_text.append(f"Confidence: {last_cam_pose.confidence:.2f}")
            info_text.append(f"Used {last_cam_pose.num_markers_used} markers")
        else:
            info_text.append("Cam pos: [No pose available]")
        
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
        cv2.imshow('ArUco Detection Test', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"aruco_test_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"Saved frame as {filename}")
        elif key == ord('c'):
            # Save current frame for calibration
            calibration_file = "marker_calibration.jpg"
            cv2.imwrite(calibration_file, frame)
            print(f"Saved calibration image as {calibration_file}")
            
            # Try to calibrate from this frame
            print("Attempting to calibrate marker positions...")
            marker_positions = calibrate_from_photo(detector, calibration_file)
            if marker_positions:
                detector.load_marker_positions(marker_positions)
                print("Calibration successful! Pose estimation now enabled.")
            else:
                print("Calibration failed. Make sure at least 2 markers are visible.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == "__main__":
    main()
