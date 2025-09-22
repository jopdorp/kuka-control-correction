"""
ChArUco Board Matching Module

This module provides board matching functionality using least error algorithm
to determine which detected board corresponds to which expected board configuration.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from charuco_board_detector import CharucoBoardConfig, DetectedCharucoBoard
from pose_utils import pose_to_T, kuka_abc_to_rotation_matrix


@dataclass
class BoardMatchResult:
    """Result of board matching operation."""
    detected_board: DetectedCharucoBoard
    matched_config: CharucoBoardConfig
    error: float  # Total error for this match
    position_error: float  # Position error in mm
    rotation_error: float  # Rotation error in degrees


class CharucoBoardMatcher:
    """
    Matches detected ChArUco boards with expected board configurations
    using least error algorithm based on current camera pose.
    """
    
    def __init__(self, 
                 board_configs: List[CharucoBoardConfig],
                 max_position_error: float = 100.0,  # mm
                 max_rotation_error: float = 45.0,   # degrees
                 position_weight: float = 1.0,
                 rotation_weight: float = 0.1):
        """
        Initialize the board matcher.
        
        Args:
            board_configs: List of expected board configurations
            max_position_error: Maximum acceptable position error in mm
            max_rotation_error: Maximum acceptable rotation error in degrees
            position_weight: Weight for position error in matching
            rotation_weight: Weight for rotation error in matching
        """
        self.logger = logging.getLogger(__name__)
        self.board_configs = board_configs
        self.max_position_error = max_position_error
        self.max_rotation_error = max_rotation_error
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        
        # Build lookup for quick access
        self.config_by_id = {config.board_id: config for config in board_configs}
        
        self.logger.info(f"Initialized board matcher for {len(board_configs)} boards")
    
    def match_boards(self, 
                    detected_boards: List[DetectedCharucoBoard],
                    camera_pose_base: np.ndarray) -> List[BoardMatchResult]:
        """
        Match detected boards with expected configurations using least error algorithm.
        
        Args:
            detected_boards: List of detected boards from image
            camera_pose_base: Current camera pose in base frame [x, y, z, a, b, c]
                             where [x, y, z] in mm and [a, b, c] in degrees
            
        Returns:
            List of successful matches, sorted by error
        """
        if not detected_boards or not self.board_configs:
            return []
        
        # Convert camera pose to transformation matrix
        T_base_cam = pose_to_T(
            camera_pose_base[:3],  # translation in mm
            camera_pose_base[3:]   # rotation in degrees
        )
        
        # Calculate expected board poses in camera frame for all configurations
        expected_poses_cam = self._calculate_expected_poses_camera(T_base_cam)
        
        # Create cost matrix for Hungarian algorithm
        cost_matrix = self._build_cost_matrix(detected_boards, expected_poses_cam)
        
        # Solve assignment problem using Hungarian algorithm
        detected_indices, config_indices = linear_sum_assignment(cost_matrix)
        
        # Build results for valid matches
        matches = []
        for det_idx, conf_idx in zip(detected_indices, config_indices):
            error = cost_matrix[det_idx, conf_idx]
            
            # Skip matches with error too high (indicates no good match)
            if error > self._max_total_error():
                continue
            
            # Calculate detailed errors
            detected_board = detected_boards[det_idx]
            config = self.board_configs[conf_idx]
            expected_pose = expected_poses_cam[conf_idx]
            
            pos_error, rot_error = self._calculate_detailed_errors(
                detected_board, expected_pose
            )
            
            # Check if errors are within acceptable limits
            if (pos_error <= self.max_position_error and 
                rot_error <= self.max_rotation_error):
                
                # Assign board ID
                detected_board.board_id = config.board_id
                
                match_result = BoardMatchResult(
                    detected_board=detected_board,
                    matched_config=config,
                    error=error,
                    position_error=pos_error,
                    rotation_error=rot_error
                )
                matches.append(match_result)
                
                self.logger.debug(f"Matched board {config.board_id}: "
                                f"pos_err={pos_error:.1f}mm, rot_err={rot_error:.1f}deg")
        
        # Sort by total error (best matches first)
        matches.sort(key=lambda x: x.error)
        
        self.logger.info(f"Successfully matched {len(matches)}/{len(detected_boards)} boards")
        return matches
    
    def _calculate_expected_poses_camera(self, T_base_cam: np.ndarray) -> List[Dict]:
        """
        Calculate expected board poses in camera frame.
        
        Args:
            T_base_cam: Camera pose in base frame (4x4 transformation matrix)
            
        Returns:
            List of expected poses (translation and rotation) in camera frame
        """
        T_cam_base = np.linalg.inv(T_base_cam)
        expected_poses = []
        
        for config in self.board_configs:
            # Expected board pose in base frame
            expected_plane = config.expected_plane
            T_base_board = pose_to_T(
                np.array(expected_plane[:3]),  # translation in mm
                np.array(expected_plane[3:])   # rotation in degrees
            )
            
            # Transform to camera frame
            T_cam_board = T_cam_base @ T_base_board
            
            # Extract translation (convert to meters for consistency with detection)
            translation_cam = T_cam_board[:3, 3] / 1000.0  # mm to meters
            
            # Extract rotation matrix and convert to rodrigues vector
            rotation_matrix_cam = T_cam_board[:3, :3]
            from scipy.spatial.transform import Rotation as R
            rotation_vec_cam = R.from_matrix(rotation_matrix_cam).as_rotvec()
            
            expected_poses.append({
                'translation': translation_cam,
                'rotation': rotation_vec_cam,
                'rotation_matrix': rotation_matrix_cam
            })
        
        return expected_poses
    
    def _build_cost_matrix(self, 
                          detected_boards: List[DetectedCharucoBoard], 
                          expected_poses: List[Dict]) -> np.ndarray:
        """
        Build cost matrix for Hungarian algorithm.
        
        Args:
            detected_boards: List of detected boards
            expected_poses: List of expected poses in camera frame
            
        Returns:
            Cost matrix where cost[i][j] is error between detected_board[i] and expected_pose[j]
        """
        n_detected = len(detected_boards)
        n_expected = len(expected_poses)
        
        cost_matrix = np.full((n_detected, n_expected), float('inf'))
        
        for i, detected in enumerate(detected_boards):
            for j, expected in enumerate(expected_poses):
                # Calculate position error (in mm)
                pos_diff = (detected.translation - expected['translation']) * 1000.0  # to mm
                pos_error = np.linalg.norm(pos_diff)
                
                # Calculate rotation error (in degrees)
                rot_error = self._calculate_rotation_error(
                    detected.rotation_matrix, expected['rotation_matrix']
                )
                
                # Check if within acceptable limits
                if (pos_error <= self.max_position_error and 
                    rot_error <= self.max_rotation_error):
                    
                    # Weighted total error
                    total_error = (self.position_weight * pos_error + 
                                 self.rotation_weight * rot_error)
                    cost_matrix[i, j] = total_error
        
        return cost_matrix
    
    def _calculate_detailed_errors(self, 
                                 detected_board: DetectedCharucoBoard,
                                 expected_pose: Dict) -> Tuple[float, float]:
        """
        Calculate detailed position and rotation errors.
        
        Args:
            detected_board: Detected board
            expected_pose: Expected pose in camera frame
            
        Returns:
            Tuple of (position_error_mm, rotation_error_degrees)
        """
        # Position error in mm
        pos_diff = (detected_board.translation - expected_pose['translation']) * 1000.0
        pos_error = np.linalg.norm(pos_diff)
        
        # Rotation error in degrees
        rot_error = self._calculate_rotation_error(
            detected_board.rotation_matrix, expected_pose['rotation_matrix']
        )
        
        return pos_error, rot_error
    
    def _calculate_rotation_error(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """
        Calculate rotation error between two rotation matrices in degrees.
        
        Args:
            R1: First rotation matrix (3x3)
            R2: Second rotation matrix (3x3)
            
        Returns:
            Rotation error in degrees
        """
        try:
            # Relative rotation
            R_rel = R1.T @ R2
            
            # Calculate angle of rotation
            trace = np.trace(R_rel)
            # Clamp to valid range for acos
            trace = np.clip(trace, -1.0, 3.0)
            angle_rad = np.arccos((trace - 1) / 2)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
        except Exception:
            # In case of numerical issues, return large error
            return 180.0
    
    def _max_total_error(self) -> float:
        """Calculate maximum acceptable total error."""
        return (self.position_weight * self.max_position_error + 
                self.rotation_weight * self.max_rotation_error)
    
    def get_unmatched_boards(self, 
                           detected_boards: List[DetectedCharucoBoard]) -> List[DetectedCharucoBoard]:
        """
        Get list of detected boards that were not matched.
        
        Args:
            detected_boards: List of all detected boards
            
        Returns:
            List of unmatched boards (board_id is None)
        """
        return [board for board in detected_boards if board.board_id is None]
    
    def get_missing_boards(self, 
                         matched_results: List[BoardMatchResult]) -> List[CharucoBoardConfig]:
        """
        Get list of expected boards that were not detected.
        
        Args:
            matched_results: List of successful matches
            
        Returns:
            List of board configurations that were not matched
        """
        matched_ids = {result.matched_config.board_id for result in matched_results}
        return [config for config in self.board_configs 
                if config.board_id not in matched_ids]