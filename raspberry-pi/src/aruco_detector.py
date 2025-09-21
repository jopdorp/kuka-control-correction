"""
ArUco Marker Detection and Pose Estimation Module

This module provides comprehensive ArUco marker detection and pose estimation
functionality for the KUKA vision correction system.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class MarkerPose:
    """Represents a detected ArUco marker with pose information."""
    marker_id: int
    translation: np.ndarray  # [x, y, z] in camera coordinates
    rotation: np.ndarray     # Rotation vector
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    corners: np.ndarray      # 2D corner points in image
    confidence: float        # Detection confidence [0-1]


@dataclass
class CameraPose:
    """Represents estimated camera pose in world coordinates."""
    translation: np.ndarray  # [x, y, z] in world coordinates
    rotation: np.ndarray     # Rotation vector
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    confidence: float        # Estimation confidence [0-1]
    num_markers_used: int    # Number of markers used for estimation


class ArucoDetector:
    """
    ArUco marker detection and pose estimation system.
    
    Handles camera calibration, marker detection, pose estimation,
    and coordinate system transformations.
    """
    
    def __init__(self, 
                 dictionary_type: int = cv2.aruco.DICT_6X6_250,
                 marker_size: float = 0.05,  # meters
                 camera_matrix: Optional[np.ndarray] = None,
                 distortion_coeffs: Optional[np.ndarray] = None):
        """
        Initialize the ArUco detector.
        
        Args:
            dictionary_type: ArUco dictionary type
            marker_size: Physical size of markers in meters
            camera_matrix: Camera intrinsic matrix (3x3)
            distortion_coeffs: Camera distortion coefficients
        """
        self.logger = logging.getLogger(__name__)
        
        # ArUco detection setup
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        # Tuned parameters for better precision and stability
        try:
            self.parameters.adaptiveThreshWinSizeMin = 5
            self.parameters.adaptiveThreshWinSizeMax = 31
            self.parameters.adaptiveThreshWinSizeStep = 6
            self.parameters.minMarkerPerimeterRate = 0.02
            self.parameters.maxMarkerPerimeterRate = 4.0
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.parameters.cornerRefinementWinSize = 5
            self.parameters.minCornerDistanceRate = 0.02
            self.parameters.minOtsuStdDev = 2.0
        except Exception:
            # Some OpenCV builds may not expose all attributes; ignore gracefully
            pass
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Marker and camera properties
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        
        # Marker positions in world coordinates (loaded from config)
        self.marker_positions: Dict[int, np.ndarray] = {}
        
        # Pose estimation settings
        self.min_markers_for_pose = 2
        self.max_reprojection_error = 2.0  # pixels
        
        self.logger.info(f"ArUco detector initialized with dictionary {dictionary_type}")
    
    def load_camera_calibration(self, calibration_file: str) -> bool:
        """
        Load camera calibration parameters from file.
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            calibration_data = np.load(calibration_file)
            self.camera_matrix = calibration_data['camera_matrix']
            self.distortion_coeffs = calibration_data['distortion_coeffs']
            self.logger.info(f"Camera calibration loaded from {calibration_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load camera calibration: {e}")
            return False
    
    def load_marker_positions(self, marker_positions: Dict[int, List[float]]):
        """
        Load known marker positions in world coordinates.
        
        Args:
            marker_positions: Dictionary mapping marker IDs to [x, y, z, rx, ry, rz]
        """
        for marker_id, position in marker_positions.items():
            if len(position) == 6:
                # Translation and rotation
                translation = np.array(position[:3])
                rotation = np.array(position[3:])
                self.marker_positions[marker_id] = {
                    'translation': translation,
                    'rotation': rotation,
                    'rotation_matrix': R.from_rotvec(rotation).as_matrix()
                }
            elif len(position) == 3:
                # Translation only (assume no rotation)
                translation = np.array(position)
                self.marker_positions[marker_id] = {
                    'translation': translation,
                    'rotation': np.zeros(3),
                    'rotation_matrix': np.eye(3)
                }
        
        self.logger.info(f"Loaded {len(self.marker_positions)} marker positions")
    
    def detect_markers(self, image: np.ndarray) -> List[MarkerPose]:
        """
        Detect ArUco markers in image and estimate their poses.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of detected marker poses
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            self.logger.error("Camera calibration not loaded")
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is None:
            return []
        
        # Estimate poses
        marker_poses = []
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.distortion_coeffs
        )
        
        for i, marker_id in enumerate(ids.flatten()):
            # Calculate confidence based on corner detection quality
            corner_area = cv2.contourArea(corners[i][0])
            confidence = min(1.0, corner_area / 1000.0)  # Normalize area to confidence
            
            # Create rotation matrix
            rotation_matrix = R.from_rotvec(rvecs[i][0]).as_matrix()
            
            marker_pose = MarkerPose(
                marker_id=int(marker_id),
                translation=tvecs[i][0],
                rotation=rvecs[i][0],
                rotation_matrix=rotation_matrix,
                corners=corners[i][0],
                confidence=confidence
            )
            marker_poses.append(marker_pose)
        
        self.logger.debug(f"Detected {len(marker_poses)} markers")
        return marker_poses
    
    def estimate_camera_pose(self, detected_markers: List[MarkerPose]) -> Optional[CameraPose]:
        """
        Estimate camera pose in world coordinates using detected markers.
        
        Args:
            detected_markers: List of detected marker poses
            
        Returns:
            Estimated camera pose or None if insufficient data
        """
        if len(detected_markers) < self.min_markers_for_pose:
            self.logger.warning(f"Insufficient markers for pose estimation: {len(detected_markers)}")
            return None
        
        # Filter markers with known world positions
        valid_markers = [m for m in detected_markers 
                        if m.marker_id in self.marker_positions]
        
        if len(valid_markers) < self.min_markers_for_pose:
            self.logger.warning(f"Insufficient known markers: {len(valid_markers)}")
            return None
        
        # Prepare point correspondences for PnP
        object_points = []
        image_points = []
        
        for marker in valid_markers:
            marker_world_pos = self.marker_positions[marker.marker_id]
            
            # Get marker corners in world coordinates
            marker_corners_world = self._get_marker_corners_world(
                marker_world_pos['translation'],
                marker_world_pos['rotation_matrix']
            )
            
            object_points.extend(marker_corners_world)
            image_points.extend(marker.corners)
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points,
            self.camera_matrix, self.distortion_coeffs,
            reprojectionError=self.max_reprojection_error
        )
        
        if not success or inliers is None:
            self.logger.warning("PnP solve failed")
            return None
        
        # Calculate confidence based on inlier ratio
        confidence = len(inliers) / len(object_points)
        
        if confidence < 0.5:  # Less than 50% inliers
            self.logger.warning(f"Low confidence pose estimation: {confidence}")
            return None
        
        # Convert to camera pose (inverse transformation)
        rotation_matrix = R.from_rotvec(rvec.flatten()).as_matrix()
        camera_rotation_matrix = rotation_matrix.T
        camera_translation = -camera_rotation_matrix @ tvec.flatten()
        camera_rvec = R.from_matrix(camera_rotation_matrix).as_rotvec()
        
        return CameraPose(
            translation=camera_translation,
            rotation=camera_rvec,
            rotation_matrix=camera_rotation_matrix,
            confidence=confidence,
            num_markers_used=len(valid_markers)
        )
    
    def _get_marker_corners_world(self, 
                                  marker_translation: np.ndarray,
                                  marker_rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Get marker corner positions in world coordinates.
        
        Args:
            marker_translation: Marker position in world coordinates
            marker_rotation_matrix: Marker orientation in world coordinates
            
        Returns:
            4x3 array of corner positions
        """
        # Standard ArUco marker corners in marker coordinate system
        half_size = self.marker_size / 2
        marker_corners_local = np.array([
            [-half_size, -half_size, 0],
            [ half_size, -half_size, 0],
            [ half_size,  half_size, 0],
            [-half_size,  half_size, 0]
        ])
        
        # Transform to world coordinates
        corners_world = (marker_rotation_matrix @ marker_corners_local.T).T + marker_translation
        return corners_world
    
    def draw_detected_markers(self, 
                            image: np.ndarray, 
                            detected_markers: List[MarkerPose],
                            draw_axes: bool = True) -> np.ndarray:
        """
        Draw detected markers on image for visualization.
        
        Args:
            image: Input image
            detected_markers: List of detected markers
            draw_axes: Whether to draw coordinate axes
            
        Returns:
            Image with drawn markers
        """
        output_image = image.copy()
        
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return output_image
        
        for marker in detected_markers:
            # Draw marker outline
            corners_int = marker.corners.astype(int)
            cv2.polylines(output_image, [corners_int], True, (0, 255, 0), 2)
            
            # Draw marker ID
            center = corners_int.mean(axis=0).astype(int)
            cv2.putText(output_image, str(marker.marker_id), 
                       tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw coordinate axes
            if draw_axes:
                axis_length = self.marker_size
                axis_points = np.array([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, -axis_length]
                ], dtype=np.float32)
                
                projected_points, _ = cv2.projectPoints(
                    axis_points, marker.rotation, marker.translation,
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
    
    def get_reprojection_error(self, 
                             detected_markers: List[MarkerPose],
                             camera_pose: CameraPose) -> float:
        """
        Calculate reprojection error for validation.
        
        Args:
            detected_markers: List of detected markers
            camera_pose: Estimated camera pose
            
        Returns:
            Mean reprojection error in pixels
        """
        if self.camera_matrix is None:
            return float('inf')
        
        total_error = 0.0
        num_points = 0
        
        for marker in detected_markers:
            if marker.marker_id not in self.marker_positions:
                continue
            
            # Project world corners to image
            marker_world_pos = self.marker_positions[marker.marker_id]
            corners_world = self._get_marker_corners_world(
                marker_world_pos['translation'],
                marker_world_pos['rotation_matrix']
            )
            
            # Transform to camera coordinates
            corners_camera = (camera_pose.rotation_matrix.T @ 
                            (corners_world - camera_pose.translation).T).T
            
            # Project to image
            projected_points, _ = cv2.projectPoints(
                corners_camera, np.zeros(3), np.zeros(3),
                self.camera_matrix, self.distortion_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Calculate error
            error = np.linalg.norm(projected_points - marker.corners, axis=1)
            total_error += error.sum()
            num_points += len(error)
        
        return total_error / num_points if num_points > 0 else float('inf')
