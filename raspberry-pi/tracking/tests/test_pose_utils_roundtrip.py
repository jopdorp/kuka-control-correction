"""
Extra roundtrip tests for pose_utils.
"""
import numpy as np
import pytest

from correction.pose_utils import pose_to_T, invert_T, kuka_abc_to_rotation_matrix, rotation_matrix_to_kuka_abc


@pytest.mark.parametrize("translation", [
    np.array([0.0, 0.0, 0.0]),
    np.array([1e-9, -1e-9, 1e-9]),
    np.array([123.456, -789.012, 34.567]),
])
@pytest.mark.parametrize("rotation", [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.001, -0.002, 0.003]),
    np.array([179.0, 89.0, -179.0]),
])
def test_T_inversion_roundtrip(translation, rotation):
    T = pose_to_T(translation, rotation)
    T_inv = invert_T(T)
    I = T @ T_inv
    assert np.allclose(I, np.eye(4), atol=1e-9)


def test_kuka_abc_roundtrip_random(seed: int = 123):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        abc = rng.uniform(low=-179, high=179, size=3)
        R = kuka_abc_to_rotation_matrix(abc)
        abc_back = rotation_matrix_to_kuka_abc(R)
        # Compare the implied rotation matrices to avoid angle wrapping issues
        R_back = kuka_abc_to_rotation_matrix(abc_back)
        assert np.allclose(R_back, R, atol=1e-8)
