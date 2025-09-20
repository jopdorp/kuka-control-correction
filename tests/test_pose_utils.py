"""
Tests for pose utilities and coordinate transformations.
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Import the module under test
from pose_utils import (
    Pose, pose_to_T, T_to_pose, invert_T, compose_T,
    rotation_matrix_to_kuka_abc, kuka_abc_to_rotation_matrix,
    apply_transform, transform_pose
)


class TestPoseClass:
    """Test cases for Pose dataclass."""

    def test_pose_creation(self):
        """Test Pose object creation."""
        pose = Pose(
            x=100.0, y=200.0, z=300.0,
            rx=0.1, ry=0.2, rz=0.3
        )
        
        assert pose.x == 100.0
        assert pose.y == 200.0
        assert pose.z == 300.0
        assert pose.rx == 0.1
        assert pose.ry == 0.2
        assert pose.rz == 0.3

    def test_pose_default_values(self):
        """Test Pose with default values."""
        pose = Pose()
        
        assert pose.x == 0.0
        assert pose.y == 0.0
        assert pose.z == 0.0
        assert pose.rx == 0.0
        assert pose.ry == 0.0
        assert pose.rz == 0.0


class TestTransformationMatrix:
    """Test cases for transformation matrix operations."""

    def test_pose_to_T_identity(self):
        """Test conversion of identity pose to transformation matrix."""
        pose = Pose()
        T = pose_to_T(pose)
        
        expected = np.eye(4)
        assert np.allclose(T, expected)

    def test_pose_to_T_translation_only(self):
        """Test conversion with translation only."""
        pose = Pose(x=100.0, y=200.0, z=300.0)
        T = pose_to_T(pose)
        
        expected = np.array([
            [1, 0, 0, 100.0],
            [0, 1, 0, 200.0],
            [0, 0, 1, 300.0],
            [0, 0, 0, 1]
        ])
        assert np.allclose(T, expected)

    def test_pose_to_T_rotation_only(self):
        """Test conversion with rotation only."""
        # 90 degree rotation around Z axis
        pose = Pose(rz=np.pi/2)
        T = pose_to_T(pose)
        
        # Check rotation part
        expected_rotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        assert np.allclose(T[:3, :3], expected_rotation, atol=1e-15)
        assert np.allclose(T[:3, 3], [0, 0, 0])

    def test_T_to_pose_identity(self):
        """Test conversion of identity matrix to pose."""
        T = np.eye(4)
        pose = T_to_pose(T)
        
        assert abs(pose.x) < 1e-15
        assert abs(pose.y) < 1e-15
        assert abs(pose.z) < 1e-15
        assert abs(pose.rx) < 1e-15
        assert abs(pose.ry) < 1e-15
        assert abs(pose.rz) < 1e-15

    def test_pose_to_T_roundtrip(self):
        """Test roundtrip conversion: pose -> T -> pose."""
        original_pose = Pose(x=100.0, y=200.0, z=300.0, rx=0.1, ry=0.2, rz=0.3)
        T = pose_to_T(original_pose)
        recovered_pose = T_to_pose(T)
        
        assert abs(recovered_pose.x - original_pose.x) < 1e-10
        assert abs(recovered_pose.y - original_pose.y) < 1e-10
        assert abs(recovered_pose.z - original_pose.z) < 1e-10
        assert abs(recovered_pose.rx - original_pose.rx) < 1e-10
        assert abs(recovered_pose.ry - original_pose.ry) < 1e-10
        assert abs(recovered_pose.rz - original_pose.rz) < 1e-10

    def test_invert_T_identity(self):
        """Test inversion of identity matrix."""
        T = np.eye(4)
        T_inv = invert_T(T)
        
        assert np.allclose(T_inv, np.eye(4))

    def test_invert_T_translation(self):
        """Test inversion of translation-only matrix."""
        T = np.array([
            [1, 0, 0, 100],
            [0, 1, 0, 200],
            [0, 0, 1, 300],
            [0, 0, 0, 1]
        ], dtype=float)
        
        T_inv = invert_T(T)
        
        expected = np.array([
            [1, 0, 0, -100],
            [0, 1, 0, -200],
            [0, 0, 1, -300],
            [0, 0, 0, 1]
        ], dtype=float)
        
        assert np.allclose(T_inv, expected)

    def test_invert_T_roundtrip(self):
        """Test that T * invert(T) = I."""
        pose = Pose(x=100.0, y=200.0, z=300.0, rx=0.1, ry=0.2, rz=0.3)
        T = pose_to_T(pose)
        T_inv = invert_T(T)
        
        result = T @ T_inv
        assert np.allclose(result, np.eye(4))

    def test_compose_T_identity(self):
        """Test composition with identity matrices."""
        T1 = np.eye(4)
        T2 = np.eye(4)
        
        result = compose_T(T1, T2)
        assert np.allclose(result, np.eye(4))

    def test_compose_T_translations(self):
        """Test composition of translation matrices."""
        T1 = pose_to_T(Pose(x=100.0, y=0.0, z=0.0))
        T2 = pose_to_T(Pose(x=0.0, y=200.0, z=0.0))
        
        result = compose_T(T1, T2)
        expected_pose = Pose(x=100.0, y=200.0, z=0.0)
        expected = pose_to_T(expected_pose)
        
        assert np.allclose(result, expected)


class TestKukaRotations:
    """Test cases for KUKA ABC rotation conversions."""

    def test_rotation_matrix_to_kuka_abc_identity(self):
        """Test conversion of identity rotation matrix."""
        R = np.eye(3)
        a, b, c = rotation_matrix_to_kuka_abc(R)
        
        assert abs(a) < 1e-15
        assert abs(b) < 1e-15
        assert abs(c) < 1e-15

    def test_kuka_abc_to_rotation_matrix_identity(self):
        """Test conversion of zero ABC angles."""
        R = kuka_abc_to_rotation_matrix(0.0, 0.0, 0.0)
        
        assert np.allclose(R, np.eye(3))

    def test_kuka_abc_roundtrip(self):
        """Test roundtrip conversion: R -> ABC -> R."""
        # Create rotation matrix for 45 degrees around Z
        angle = np.pi / 4
        R_original = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        a, b, c = rotation_matrix_to_kuka_abc(R_original)
        R_recovered = kuka_abc_to_rotation_matrix(a, b, c)
        
        assert np.allclose(R_recovered, R_original)

    def test_kuka_abc_specific_angles(self):
        """Test specific known angle conversions."""
        # 90 degrees around Z axis
        R_z90 = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        a, b, c = rotation_matrix_to_kuka_abc(R_z90)
        # For 90° rotation around Z, we expect C = 90°, A = B = 0°
        assert abs(c - np.pi/2) < 1e-10
        assert abs(a) < 1e-10
        assert abs(b) < 1e-10


class TestTransformOperations:
    """Test cases for transform application functions."""

    def test_apply_transform_point_identity(self):
        """Test applying identity transform to a point."""
        T = np.eye(4)
        point = np.array([100.0, 200.0, 300.0])
        
        result = apply_transform(T, point)
        assert np.allclose(result, point)

    def test_apply_transform_point_translation(self):
        """Test applying translation to a point."""
        T = pose_to_T(Pose(x=10.0, y=20.0, z=30.0))
        point = np.array([100.0, 200.0, 300.0])
        
        result = apply_transform(T, point)
        expected = np.array([110.0, 220.0, 330.0])
        assert np.allclose(result, expected)

    def test_apply_transform_point_rotation(self):
        """Test applying rotation to a point."""
        # 90 degree rotation around Z axis
        T = pose_to_T(Pose(rz=np.pi/2))
        point = np.array([100.0, 0.0, 0.0])
        
        result = apply_transform(T, point)
        expected = np.array([0.0, 100.0, 0.0])
        assert np.allclose(result, expected, atol=1e-15)

    def test_transform_pose_identity(self):
        """Test transforming pose with identity transform."""
        T = np.eye(4)
        pose = Pose(x=100.0, y=200.0, z=300.0, rx=0.1, ry=0.2, rz=0.3)
        
        result = transform_pose(T, pose)
        
        assert abs(result.x - pose.x) < 1e-15
        assert abs(result.y - pose.y) < 1e-15
        assert abs(result.z - pose.z) < 1e-15
        assert abs(result.rx - pose.rx) < 1e-15
        assert abs(result.ry - pose.ry) < 1e-15
        assert abs(result.rz - pose.rz) < 1e-15

    def test_transform_pose_translation(self):
        """Test transforming pose with translation."""
        T = pose_to_T(Pose(x=50.0, y=100.0, z=150.0))
        pose = Pose(x=100.0, y=200.0, z=300.0)
        
        result = transform_pose(T, pose)
        
        assert abs(result.x - 150.0) < 1e-10
        assert abs(result.y - 300.0) < 1e-10
        assert abs(result.z - 450.0) < 1e-10


class TestErrorHandling:
    """Test error handling in pose utilities."""

    def test_invalid_matrix_size(self):
        """Test handling of invalid matrix sizes."""
        # Test with 3x3 matrix instead of 4x4
        invalid_T = np.eye(3)
        
        with pytest.raises(ValueError):
            T_to_pose(invalid_T)

    def test_invalid_point_dimensions(self):
        """Test handling of invalid point dimensions."""
        T = np.eye(4)
        invalid_point = np.array([1, 2])  # 2D instead of 3D
        
        with pytest.raises(ValueError):
            apply_transform(T, invalid_point)

    def test_non_homogeneous_matrix(self):
        """Test handling of non-homogeneous transformation matrix."""
        # Create matrix with invalid bottom row
        invalid_T = np.eye(4)
        invalid_T[3, :] = [1, 2, 3, 4]  # Should be [0, 0, 0, 1]
        
        # Should still work but may give unexpected results
        # This is more of a warning case than an error
        result = T_to_pose(invalid_T)
        assert isinstance(result, Pose)
