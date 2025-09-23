"""
ChArUco Board Matching Module

Least-error assignment between detected boards and expected planes.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from charuco_board_detector import CharucoBoardConfig, DetectedCharucoBoard
from pose_utils import pose_to_T


@dataclass
class BoardMatchResult:
    detected_board: DetectedCharucoBoard
    matched_config: CharucoBoardConfig
    detected_index: int
    config_index: int
    error: float
    position_error: float
    rotation_error: float


class CharucoBoardMatcher:
    def __init__(
        self,
        board_configs: List[CharucoBoardConfig],
        max_position_error: float = 100.0,
        max_rotation_error: float = 45.0,
        position_weight: float = 1.0,
        rotation_weight: float = 0.1,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.board_configs = board_configs
        self.max_position_error = max_position_error
        self.max_rotation_error = max_rotation_error
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight

        self.logger.info(f"Initialized board matcher for {len(board_configs)} boards")

    def match_boards(
        self,
        detected_boards: List[DetectedCharucoBoard],
        camera_pose_base: np.ndarray,
    ) -> List[BoardMatchResult]:
        if not detected_boards or not self.board_configs:
            return []

        # Convert camera pose in BASE to transform
        T_base_cam = pose_to_T(camera_pose_base[:3], camera_pose_base[3:])
        expected_poses_cam = self._calculate_expected_poses_camera(T_base_cam)

        cost_matrix = self._build_cost_matrix(detected_boards, expected_poses_cam)
        detected_indices, config_indices = linear_sum_assignment(cost_matrix)

        matches: List[BoardMatchResult] = []
        for det_idx, conf_idx in zip(detected_indices, config_indices):
            error = cost_matrix[det_idx, conf_idx]
            if error > self._max_total_error():
                continue

            detected_board = detected_boards[det_idx]
            config = self.board_configs[conf_idx]
            expected_pose = expected_poses_cam[conf_idx]

            pos_error, rot_error = self._calculate_detailed_errors(
                detected_board, expected_pose
            )

            if pos_error <= self.max_position_error and rot_error <= self.max_rotation_error:
                matches.append(
                    BoardMatchResult(
                        detected_board=detected_board,
                        matched_config=config,
                        detected_index=det_idx,
                        config_index=conf_idx,
                        error=error,
                        position_error=pos_error,
                        rotation_error=rot_error,
                    )
                )
                self.logger.debug(
                    f"Matched config[{conf_idx}] with detected[{det_idx}]: "
                    f"pos_err={pos_error:.1f}mm, rot_err={rot_error:.1f}deg"
                )

        matches.sort(key=lambda x: x.error)
        self.logger.info(
            f"Successfully matched {len(matches)}/{len(detected_boards)} boards"
        )
        return matches

    def _calculate_expected_poses_camera(self, T_base_cam: np.ndarray) -> List[Dict]:
        T_cam_base = np.linalg.inv(T_base_cam)
        expected_poses: List[Dict] = []
        for config in self.board_configs:
            expected_plane = config.expected_plane
            T_base_board = pose_to_T(
                np.array(expected_plane[:3]), np.array(expected_plane[3:])
            )
            T_cam_board = T_cam_base @ T_base_board
            translation_cam = T_cam_board[:3, 3] / 1000.0
            rotation_matrix_cam = T_cam_board[:3, :3]
            from scipy.spatial.transform import Rotation as R

            rotation_vec_cam = R.from_matrix(rotation_matrix_cam).as_rotvec()
            expected_poses.append(
                {
                    "translation": translation_cam,
                    "rotation": rotation_vec_cam,
                    "rotation_matrix": rotation_matrix_cam,
                }
            )
        return expected_poses

    def _build_cost_matrix(
        self, detected_boards: List[DetectedCharucoBoard], expected_poses: List[Dict]
    ) -> np.ndarray:
        n_detected = len(detected_boards)
        n_expected = len(expected_poses)
        cost_matrix = np.full((n_detected, n_expected), float("inf"))
        for i, detected in enumerate(detected_boards):
            for j, expected in enumerate(expected_poses):
                pos_error = float(
                    np.linalg.norm((detected.translation - expected["translation"]) * 1000.0)
                )
                rot_error = self._calculate_rotation_error(
                    detected.rotation_matrix, expected["rotation_matrix"]
                )
                if pos_error <= self.max_position_error and rot_error <= self.max_rotation_error:
                    total_error = self.position_weight * pos_error + self.rotation_weight * rot_error
                    cost_matrix[i, j] = total_error
        return cost_matrix

    def _calculate_detailed_errors(
        self, detected_board: DetectedCharucoBoard, expected_pose: Dict
    ) -> Tuple[float, float]:
        pos_error = float(
            np.linalg.norm((detected_board.translation - expected_pose["translation"]) * 1000.0)
        )
        rot_error = self._calculate_rotation_error(
            detected_board.rotation_matrix, expected_pose["rotation_matrix"]
        )
        return pos_error, rot_error

    def _calculate_rotation_error(self, R1: np.ndarray, R2: np.ndarray) -> float:
        try:
            R_rel = R1.T @ R2
            trace = float(np.trace(R_rel))
            trace = float(np.clip(trace, -1.0, 3.0))
            angle_rad = float(np.arccos((trace - 1) / 2))
            return float(np.degrees(angle_rad))
        except Exception:
            return 180.0

    def _max_total_error(self) -> float:
        return self.position_weight * self.max_position_error + self.rotation_weight * self.max_rotation_error

    def get_unmatched_boards(
        self, detected_boards: List[DetectedCharucoBoard], matches: List[BoardMatchResult]
    ) -> List[DetectedCharucoBoard]:
        matched_det_indices = {m.detected_index for m in matches}
        return [b for i, b in enumerate(detected_boards) if i not in matched_det_indices]

    def get_missing_boards(self, matched_results: List[BoardMatchResult]) -> List[int]:
        matched_conf_indices = {m.config_index for m in matched_results}
        return [i for i in range(len(self.board_configs)) if i not in matched_conf_indices]