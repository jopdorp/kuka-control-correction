#!/usr/bin/env python3
"""
Simple camera calibration using ChArUco board.
Just finds the camera matrix and distortion coefficients.
"""

import cv2
import numpy as np
from PIL import Image

def main():
    print("Simple Camera Calibration")
    print("========================")
    
    # Create ChArUco board: 4x3 squares, 5cm squares, 3cm markers
    marker_size = 0.03  # meters
    square_size = 0.05  # meters
    # Board configuration: 4 squares wide × 3 squares tall
    squares_width = 4
    squares_height = 3

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    charuco = cv2.aruco.CharucoBoard(
        (squares_width, squares_height), square_size, marker_size, dictionary
    )
    
    # Create ChArUco detector
    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(charuco, charuco_params, detector_params)

    # Calculate image size for 600 DPI
    dpi = 600
    
    # Convert square size from meters to centimeters
    square_size_cm = square_size * 100  # 0.05m = 5cm
    
    # Calculate total board dimensions in centimeters
    board_width_cm = squares_width * square_size_cm   # 4 squares × 5cm = 20cm
    board_height_cm = squares_height * square_size_cm  # 3 squares × 5cm = 15cm
    
    # Convert to pixels: cm ÷ 2.54 (cm/inch) × DPI
    width_pixels = int(board_width_cm / 2.54 * dpi)
    height_pixels = int(board_height_cm / 2.54 * dpi)
    
    print(f"Generating board: {width_pixels}×{height_pixels} pixels for {board_width_cm:.1f}×{board_height_cm:.1f}cm at {dpi} DPI")

    # Generate and save the board
    board_image = charuco.generateImage((width_pixels, height_pixels))
    
    # Save with DPI metadata using PIL
    pil_image = Image.fromarray(board_image)
    pil_image.save("charuco_board.png", dpi=(dpi, dpi))
    
    print("Saved charuco_board.png - print this at 600 DPI for correct size!")
    print(f"When printing, ensure your printer is set to 600 DPI and 'Actual Size' (100% scale)")
    
    input("Print the board and press Enter to start calibration...")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    # Storage for calibration
    all_corners = []
    all_ids = []
    
    print("Move the board around. Press 'c' to capture, 'q' to quit, 'k' to calibrate")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ChArUco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        
                # Show detection status
        if charuco_ids is not None and len(charuco_ids) >= 4:
            # Draw ChArUco corners and ArUco markers
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, (0, 255, 0))
            
            status = f"Ready: {len(charuco_ids)} corners, {len(all_corners)} images"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif marker_ids is not None:
            # Show detected markers but not enough corners
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, (255, 255, 0))
            corner_count = len(charuco_ids) if charuco_ids is not None else 0
            status = f"Detected: {corner_count} corners (need 4+), {len(marker_ids)} markers"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No board detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "c: capture, k: calibrate, q: quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if charuco_ids is not None and len(charuco_ids) >= 4:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print(f"✓ Captured image {len(all_corners)} with {len(charuco_ids)} corners")
            else:
                corner_count = len(charuco_ids) if charuco_ids is not None else 0
                print(f"✗ Need at least 4 corners, got {corner_count}")
        elif key == ord('k'):
            if len(all_corners) > 10:
                print("Calibrating...")
                
                # Calibrate camera using ChArUco corners
                # Convert ChArUco corners to object points and image points
                object_points = []
                image_points = []
                
                for corners, ids in zip(all_corners, all_ids):
                    # Get 3D object points for detected corners
                    obj_pts, img_pts = charuco.matchImagePoints(corners, ids)
                    if len(obj_pts) > 0:
                        object_points.append(obj_pts)
                        image_points.append(img_pts)
                
                if len(object_points) > 0:
                    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                        object_points, image_points, gray.shape[::-1], None, None
                    )
                
                print(f"Calibration done! Reprojection error: {ret:.3f}")
                print(f"Camera matrix:\n{camera_matrix}")
                print(f"Distortion coefficients: {dist_coeffs.flatten()}")
                
                # Save results
                np.savez('camera_calibration.npz', 
                         camera_matrix=camera_matrix, 
                         distortion_coeffs=dist_coeffs,
                         calibration_error=ret)
                print("Saved to camera_calibration.npz")
                break
            else:
                print(f"Need more images. Have {len(all_corners)}, need at least 10")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
