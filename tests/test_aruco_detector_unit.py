"""
Unit tests for ArucoDetector with minimal mocking. Skips if SciPy is not available.
"""
import os
import sys
import numpy as np
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'raspberry-pi', 'src'))

scipy_available = True
try:
    from scipy.spatial.transform import Rotation as R  # noqa: F401
except Exception:
    scipy_available = False

from aruco_detector import ArucoDetector, MarkerPose

pytestmark = pytest.mark.skipif(not scipy_available, reason="SciPy not installed")


def test_detector_detect_markers_with_calib():
    det = ArucoDetector()
    # Fake calibration
    det.camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    det.distortion_coeffs = np.zeros(5)

    # Provide a stub detector instance with the expected API
    class StubDetector:
        def detectMarkers(self, gray):
            corners = [np.array([[[100, 100], [150, 100], [150, 150], [100, 150]]], dtype=np.float32)]
            ids = np.array([[7]])
            rejected = []
            return corners, ids, rejected

    det.detector = StubDetector()

    # Patch cv2.aruco.estimatePoseSingleMarkers
    import cv2

    def fake_estimate(corners, size, K, D):
        rvecs = np.array([[[0.1, 0.2, 0.3]]], dtype=float)
        tvecs = np.array([[[0.01, 0.02, 0.5]]], dtype=float)
        return rvecs, tvecs, None

    cv2.aruco.estimatePoseSingleMarkers = fake_estimate

    img = np.ones((480, 640), dtype=np.uint8)
    markers = det.detect_markers(img)

    assert len(markers) == 1
    m: MarkerPose = markers[0]
    assert m.marker_id == 7
    assert m.corners.shape == (4, 2)
    assert m.rotation_matrix.shape == (3, 3)
    assert m.translation.shape == (3,)
