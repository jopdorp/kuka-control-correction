"""
Pose utilities and dataclasses for vision correction.

Provides minimal SE3 helpers and types used by the system without any robot
kinematics. All positions are in mm; rotations are KUKA ABC in degrees unless
stated otherwise.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RobotPose:
    translation: np.ndarray  # [X, Y, Z] in mm
    rotation: np.ndarray     # [A, B, C] in degrees (KUKA convention)
    rotation_matrix: np.ndarray  # 3x3
    frame: str = "BASE"


@dataclass
class MoveCommand:
    command_type: str
    target_pose: RobotPose
    velocity: float
    acceleration: float
    tool_data: Optional[dict] = None
    base_data: Optional[dict] = None


def kuka_abc_to_rotation_matrix(abc_deg: np.ndarray) -> np.ndarray:
    a, b, c = np.deg2rad(abc_deg)
    Rz = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1]
    ])
    Ry = np.array([
        [ np.cos(b), 0, np.sin(b)],
        [ 0,         1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    Rx = np.array([
        [1, 0,          0],
        [0, np.cos(c), -np.sin(c)],
        [0, np.sin(c),  np.cos(c)]
    ])
    return Rz @ Ry @ Rx


def rotation_matrix_to_kuka_abc(Rm: np.ndarray) -> np.ndarray:
    sy = np.sqrt(Rm[0, 0] ** 2 + Rm[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        a = np.arctan2(Rm[1, 0], Rm[0, 0])
        b = np.arctan2(-Rm[2, 0], sy)
        c = np.arctan2(Rm[2, 1], Rm[2, 2])
    else:
        a = np.arctan2(-Rm[0, 1], Rm[1, 1])
        b = np.arctan2(-Rm[2, 0], sy)
        c = 0.0
    return np.rad2deg([a, b, c])


def pose_to_T(translation_mm: np.ndarray, rotation_abc_deg: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = kuka_abc_to_rotation_matrix(rotation_abc_deg)
    T[:3, 3] = translation_mm
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t
    return Ti
