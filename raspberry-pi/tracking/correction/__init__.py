"""
KUKA Vision Correction System - Correction Module

This package contains the core vision correction functionality including:
- Vision correction system coordination
- Camera source management  
- CharUco board detection and matching
- Pose utilities and transformations
- Network communication helpers
"""

from .vision_correction_system import VisionCorrectionSystem, SystemConfig
from .correction_helper import VisionCorrectionHelper
from .pose_utils import (
    MoveCommand,
    RobotPose,
    kuka_abc_to_rotation_matrix,
    rotation_matrix_to_kuka_abc,
    pose_to_T,
    invert_T
)

__all__ = [
    'VisionCorrectionSystem',
    'SystemConfig', 
    'VisionCorrectionHelper',
    'MoveCommand',
    'RobotPose',
    'kuka_abc_to_rotation_matrix',
    'rotation_matrix_to_kuka_abc',
    'pose_to_T',
    'invert_T'
]
