"""
Tests for pose utilities and coordinate transformations.
"""
import os
import sys
import pytest
import numpy as np

# Add correction directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'correction'))

# Import the module under test
from pose_utils import (
    RobotPose, MoveCommand,
    pose_to_T, invert_T,
    rotation_matrix_to_kuka_abc, kuka_abc_to_rotation_matrix
)


class TestRobotPose:
    """Test cases for RobotPose dataclass."""

    def test_robot_pose_creation(self):
        """Test RobotPose creation with valid inputs."""
        translation = np.array([100.0, 200.0, 300.0])
        rotation = np.array([10.0, 20.0, 30.0])
        rotation_matrix = kuka_abc_to_rotation_matrix(rotation)
        
        pose = RobotPose(
            translation=translation,
            rotation=rotation,
            rotation_matrix=rotation_matrix
        )
        
        assert np.allclose(pose.translation, translation)
        assert np.allclose(pose.rotation, rotation)
        assert np.allclose(pose.rotation_matrix, rotation_matrix)
        assert pose.frame == "BASE"

    def test_robot_pose_custom_frame(self):
        """Test RobotPose with custom frame."""
        translation = np.array([0.0, 0.0, 0.0])
        rotation = np.array([0.0, 0.0, 0.0])
        rotation_matrix = np.eye(3)
        
        pose = RobotPose(
            translation=translation,
            rotation=rotation,
            rotation_matrix=rotation_matrix,
            frame="TOOL"
        )
        
        assert pose.frame == "TOOL"


class TestMoveCommand:
    """Test cases for MoveCommand dataclass."""

    def test_move_command_creation(self):
        """Test MoveCommand creation."""
        pose = RobotPose(
            translation=np.array([100.0, 200.0, 300.0]),
            rotation=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        
        command = MoveCommand(
            command_type="PTP",
            target_pose=pose,
            velocity=50.0,
            acceleration=100.0
        )
        
        assert command.command_type == "PTP"
        assert command.target_pose == pose
        assert command.velocity == 50.0
        assert command.acceleration == 100.0
        assert command.tool_data is None
        assert command.base_data is None


class TestRotationConversions:
    """Test rotation matrix to/from KUKA ABC conversions."""

    def test_identity_rotation(self):
        """Test identity rotation conversion."""
        abc = np.array([0.0, 0.0, 0.0])
        rot_mat = kuka_abc_to_rotation_matrix(abc)
        
        assert np.allclose(rot_mat, np.eye(3))
        
        # Convert back
        abc_recovered = rotation_matrix_to_kuka_abc(rot_mat)
        assert np.allclose(abc_recovered, abc, atol=1e-10)

    def test_simple_rotations(self):
        """Test simple axis rotations."""
        # 90 degree rotation around Z (A axis)
        abc = np.array([90.0, 0.0, 0.0])
        rot_mat = kuka_abc_to_rotation_matrix(abc)
        abc_recovered = rotation_matrix_to_kuka_abc(rot_mat)
        
        assert np.allclose(abc_recovered, abc, atol=1e-10)

    def test_combined_rotations(self):
        """Test combined rotations."""
        abc = np.array([30.0, 45.0, 60.0])
        rot_mat = kuka_abc_to_rotation_matrix(abc)
        abc_recovered = rotation_matrix_to_kuka_abc(rot_mat)
        
        assert np.allclose(abc_recovered, abc, atol=1e-8)

    def test_orthogonality(self):
        """Test that rotation matrices are orthogonal."""
        abc = np.array([45.0, 30.0, 60.0])
        rot_mat = kuka_abc_to_rotation_matrix(abc)
        
        # Check orthogonality: R @ R.T should be identity
        should_be_identity = rot_mat @ rot_mat.T
        assert np.allclose(should_be_identity, np.eye(3))
        
        # Check determinant should be 1
        assert np.allclose(np.linalg.det(rot_mat), 1.0)


class TestPoseToT:
    """Test pose to transformation matrix conversion."""

    def test_identity_transform(self):
        """Test identity transformation."""
        translation = np.array([0.0, 0.0, 0.0])
        rotation = np.array([0.0, 0.0, 0.0])
        
        T = pose_to_T(translation, rotation)
        
        expected = np.eye(4)
        assert np.allclose(T, expected)

    def test_translation_only(self):
        """Test pure translation."""
        translation = np.array([100.0, 200.0, 300.0])
        rotation = np.array([0.0, 0.0, 0.0])
        
        T = pose_to_T(translation, rotation)
        
        expected = np.eye(4)
        expected[:3, 3] = translation
        assert np.allclose(T, expected)

    def test_rotation_only(self):
        """Test pure rotation."""
        translation = np.array([0.0, 0.0, 0.0])
        rotation = np.array([90.0, 0.0, 0.0])  # 90 degrees around Z
        
        T = pose_to_T(translation, rotation)
        
        # Should have rotation matrix in top-left 3x3
        expected_rot = kuka_abc_to_rotation_matrix(rotation)
        assert np.allclose(T[:3, :3], expected_rot)
        assert np.allclose(T[:3, 3], translation)

    def test_combined_transform(self):
        """Test combined rotation and translation."""
        translation = np.array([100.0, 200.0, 300.0])
        rotation = np.array([30.0, 45.0, 60.0])
        
        T = pose_to_T(translation, rotation)
        
        # Check structure
        assert T.shape == (4, 4)
        assert np.allclose(T[3, :], [0, 0, 0, 1])
        
        # Check rotation part
        expected_rot = kuka_abc_to_rotation_matrix(rotation)
        assert np.allclose(T[:3, :3], expected_rot)
        
        # Check translation part
        assert np.allclose(T[:3, 3], translation)


class TestInvertT:
    """Test transformation matrix inversion."""

    def test_invert_identity(self):
        """Test inverting identity matrix."""
        T = np.eye(4)
        T_inv = invert_T(T)
        
        assert np.allclose(T_inv, np.eye(4))

    def test_invert_translation(self):
        """Test inverting pure translation."""
        translation = np.array([100.0, 200.0, 300.0])
        rotation = np.array([0.0, 0.0, 0.0])
        
        T = pose_to_T(translation, rotation)
        T_inv = invert_T(T)
        
        # Should compose to identity
        should_be_identity = T @ T_inv
        assert np.allclose(should_be_identity, np.eye(4))

    def test_invert_rotation(self):
        """Test inverting pure rotation."""
        translation = np.array([0.0, 0.0, 0.0])
        rotation = np.array([45.0, 30.0, 60.0])
        
        T = pose_to_T(translation, rotation)
        T_inv = invert_T(T)
        
        # Should compose to identity
        should_be_identity = T @ T_inv
        assert np.allclose(should_be_identity, np.eye(4))

    def test_invert_combined(self):
        """Test inverting combined transformation."""
        translation = np.array([100.0, 200.0, 300.0])
        rotation = np.array([30.0, 45.0, 60.0])
        
        T = pose_to_T(translation, rotation)
        T_inv = invert_T(T)
        
        # Should compose to identity both ways
        should_be_identity1 = T @ T_inv
        should_be_identity2 = T_inv @ T
        
        assert np.allclose(should_be_identity1, np.eye(4))
        assert np.allclose(should_be_identity2, np.eye(4))

    def test_double_invert(self):
        """Test that inverting twice gives original."""
        translation = np.array([50.0, 100.0, 150.0])
        rotation = np.array([15.0, 25.0, 35.0])
        
        T = pose_to_T(translation, rotation)
        T_inv = invert_T(T)
        T_double_inv = invert_T(T_inv)
        
        assert np.allclose(T_double_inv, T)


class TestNumericalStability:
    """Test numerical stability of transformations."""

    def test_small_angles(self):
        """Test very small rotation angles."""
        abc = np.array([0.001, 0.002, 0.003])  # Very small angles
        rot_mat = kuka_abc_to_rotation_matrix(abc)
        abc_recovered = rotation_matrix_to_kuka_abc(rot_mat)
        
        assert np.allclose(abc_recovered, abc, atol=1e-10)

    def test_large_angles(self):
        """Test large rotation angles."""
        abc = np.array([179.0, 89.0, 179.0])  # Near singularities
        rot_mat = kuka_abc_to_rotation_matrix(abc)
        abc_recovered = rotation_matrix_to_kuka_abc(rot_mat)
        
        # Allow slightly larger tolerance for near-singularity cases
        assert np.allclose(abc_recovered, abc, atol=1e-6)

    def test_negative_angles(self):
        """Test negative rotation angles."""
        abc = np.array([-45.0, -30.0, -60.0])
        rot_mat = kuka_abc_to_rotation_matrix(abc)
        abc_recovered = rotation_matrix_to_kuka_abc(rot_mat)
        
        assert np.allclose(abc_recovered, abc, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
