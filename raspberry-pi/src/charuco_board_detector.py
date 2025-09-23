"""
ChArUco Board Detection Module

This module provides ChArUco board detection and matching functionality
for the KUKA vision correction system to support multiple boards simultaneously.
"""

import cv2
import numpy as np
from typing import List, Optional
import logging
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class CharucoBoardConfig:
    """Configuration for a ChArUco board."""
    squares_x: int  # Number of squares in X direction
    squares_y: int  # Number of squares in Y direction
    square_size: float  # Size of each square in meters
    marker_size: float  # Size of ArUco markers in meters
    dictionary_type: int  # ArUco dictionary type
    # Expected plane definition [x, y, z, a, b, c] where board should be located
    expected_plane: List[float]  # [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg]


@dataclass
class DetectedCharucoBoard:
    """Observed ChArUco board in the current frame.

    Note: This intentionally does not inherit board configuration to decouple
    detection outputs from configuration inputs. Identification is by array position.
    """
    corners: np.ndarray  # Detected ChArUco corners in image coordinates
    ids: np.ndarray  # ChArUco corner IDs
    translation: np.ndarray  # [x, y, z] board pose in camera coordinates (meters)
    rotation: np.ndarray  # Rotation vector (rodrigues)
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    confidence: float  # Detection confidence [0-1]
    num_corners: int  # Number of detected corners


class CharucoBoardDetector:
    """
    ChArUco board detection system for multiple simultaneous boards.
    
    Handles detection of multiple ChArUco boards in a single image and
    provides pose estimation for each detected board.
    """
    
    def __init__(self, 
                 board_configs: List[CharucoBoardConfig],
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coeffs: Optional[np.ndarray] = None,
                 min_corners_for_pose: int = 6):
        """
        Initialize the ChArUco board detector.
        
        Args:
            board_configs: List of board configurations to detect
            camera_matrix: Camera intrinsic matrix (3x3)
            distortion_coeffs: Camera distortion coefficients
            min_corners_for_pose: Minimum corners needed for pose estimation
        """
        self.logger = logging.getLogger(__name__)
        self.board_configs = board_configs
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.min_corners_for_pose = min_corners_for_pose

        # Create ChArUco boards and detectors lists (index-based)
        self.charuco_boards: List[cv2.aruco.CharucoBoard] = []
        self.charuco_detectors: List[cv2.aruco.CharucoDetector] = []

        for config in board_configs:
            # Create ArUco dictionary
            dictionary = cv2.aruco.getPredefinedDictionary(config.dictionary_type)

            # Create ChArUco board
            board = cv2.aruco.CharucoBoard(
                (config.squares_x, config.squares_y),
                config.square_size,
                config.marker_size,
                dictionary,
            )

            # Create detector with optimized parameters
            detector_params = cv2.aruco.DetectorParameters()
            charuco_params = cv2.aruco.CharucoParameters()

            # Optimize for multiple boards
            try:
                detector_params.adaptiveThreshWinSizeMin = 3
                detector_params.adaptiveThreshWinSizeMax = 23
                detector_params.minMarkerPerimeterRate = 0.03
                detector_params.maxMarkerPerimeterRate = 4.0
                detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            except AttributeError:
                # Some OpenCV versions may not have these attributes
                pass

            detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

            self.charuco_boards.append(board)
            self.charuco_detectors.append(detector)

        self.logger.info(
            f"Initialized ChArUco detector for {len(self.board_configs)} boards"
        )
    
    def detect_boards(self, image: np.ndarray) -> List[DetectedCharucoBoard]:
        """
        Detect all ChArUco boards in the image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of detected boards
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            self.logger.error("Camera calibration not loaded")
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        detected_boards = []
        
        # Try to detect each board type
        for idx, config in enumerate(self.board_configs):
            detector = self.charuco_detectors[idx]
            
            # Detect ChArUco board
            det_res = detector.detectBoard(gray)
            charuco_corners, charuco_ids = det_res[0], det_res[1]
            
            # Check if we have enough corners for pose estimation
            if (charuco_corners is not None and charuco_ids is not None and 
                len(charuco_corners) >= self.min_corners_for_pose):
                
                # Estimate pose
                success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids,
                    self.charuco_boards[idx],
                    self.camera_matrix, self.distortion_coeffs,
                    None, None
                )
                
                if success:
                    # Convert rotation vector to matrix
                    rotation_matrix = R.from_rotvec(rvec.flatten()).as_matrix()
                    
                    # Calculate confidence based on number of detected corners
                    max_corners = config.squares_x * config.squares_y
                    confidence = min(1.0, len(charuco_corners) / max_corners * 1.5)
                    
                    detected_board = DetectedCharucoBoard(
                        corners=charuco_corners,
                        ids=charuco_ids,
                        translation=tvec.flatten(),
                        rotation=rvec.flatten(),
                        rotation_matrix=rotation_matrix,
                        confidence=confidence,
                        num_corners=len(charuco_corners),
                    )
                    
                    detected_boards.append(detected_board)
                    
                    self.logger.debug(
                        f"Detected potential board_{idx} with {len(charuco_corners)} corners, "
                        f"confidence: {confidence:.2f}"
                    )
        
        self.logger.debug(f"Total detected boards: {len(detected_boards)}")
        return detected_boards
    
    def draw_detected_boards(self, 
                           image: np.ndarray, 
                           detected_boards: List[DetectedCharucoBoard]) -> np.ndarray:
        """
        Draw detected ChArUco boards on image for visualization.
        
        Args:
            image: Input image
            detected_boards: List of detected boards
            
        Returns:
            Image with drawn boards
        """
        output_image = image.copy()
        
        for i, board in enumerate(detected_boards):
            # Draw ChArUco corners
            cv2.aruco.drawDetectedCornersCharuco(
                output_image, board.corners, board.ids, (0, 255, 0)
            )
            
            # Draw label by index
            center = board.corners.mean(axis=0).astype(int).flatten()
            cv2.putText(output_image, f"Board_{i}", 
                        tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw coordinate axes if camera calibration is available
            if self.camera_matrix is not None and self.distortion_coeffs is not None:
                axis_length = 0.05  # 5cm
                axis_points = np.array([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, -axis_length]
                ], dtype=np.float32)
                
                projected_points, _ = cv2.projectPoints(
                    axis_points, board.rotation, board.translation,
                    self.camera_matrix, self.distortion_coeffs
                )
                projected_points = projected_points.reshape(-1, 2).astype(int)
                
                # Draw axes (X: red, Y: green, Z: blue)
                cv2.arrowedLine(output_image, tuple(projected_points[0]), 
                               tuple(projected_points[1]), (0, 0, 255), 3)  # X-axis: red
                cv2.arrowedLine(output_image, tuple(projected_points[0]), 
                               tuple(projected_points[2]), (0, 255, 0), 3)  # Y-axis: green
                cv2.arrowedLine(output_image, tuple(projected_points[0]), 
                               tuple(projected_points[3]), (255, 0, 0), 3)  # Z-axis: blue
        
        return output_image