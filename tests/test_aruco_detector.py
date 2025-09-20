"""
Tests for ArUco marker detection functionality.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
from aruco_detector import ArucoDetector, DetectedMarker


class TestArucoDetector:
    """Test cases for ArucoDetector class."""

    def test_initialization(self, sample_camera_matrix, sample_distortion_coeffs):
        """Test ArucoDetector initialization."""
        detector = ArucoDetector(
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            marker_size=0.05
        )
        
        assert detector.marker_size == 0.05
        assert np.array_equal(detector.camera_matrix, sample_camera_matrix)
        assert np.array_equal(detector.distortion_coeffs, sample_distortion_coeffs)
        assert detector.aruco_dict is not None

    def test_detect_markers_empty_frame(self, sample_camera_matrix, sample_distortion_coeffs):
        """Test detection on frame with no markers."""
        detector = ArucoDetector(
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            marker_size=0.05
        )
        
        # Create empty white frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        markers = detector.detect_markers(frame)
        assert isinstance(markers, list)
        assert len(markers) == 0

    @patch('cv2.aruco.detectMarkers')
    @patch('cv2.aruco.estimatePoseSingleMarkers')
    def test_detect_markers_with_poses(self, mock_estimate_pose, mock_detect, 
                                     sample_camera_matrix, sample_distortion_coeffs):
        """Test marker detection with pose estimation."""
        detector = ArucoDetector(
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            marker_size=0.05
        )
        
        # Mock detection results
        mock_corners = [np.array([[[100, 100], [150, 100], [150, 150], [100, 150]]], dtype=np.float32)]
        mock_ids = np.array([[23]])
        mock_detect.return_value = (mock_corners, mock_ids, None)
        
        # Mock pose estimation
        mock_rvecs = np.array([[[0.1, 0.2, 0.3]]])
        mock_tvecs = np.array([[[0.1, 0.2, 0.5]]])  # 500mm away
        mock_estimate_pose.return_value = (mock_rvecs, mock_tvecs)
        
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        markers = detector.detect_markers(frame)
        
        assert len(markers) == 1
        marker = markers[0]
        assert marker.id == 23
        assert marker.corners.shape == (4, 2)
        assert marker.rvec.shape == (3,)
        assert marker.tvec.shape == (3,)
        assert marker.distance > 0

    def test_detect_markers_no_pose_estimation(self, sample_camera_matrix, sample_distortion_coeffs):
        """Test detection without pose estimation."""
        detector = ArucoDetector(
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            marker_size=0.05
        )
        
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        with patch('cv2.aruco.detectMarkers') as mock_detect:
            mock_corners = [np.array([[[100, 100], [150, 100], [150, 150], [100, 150]]], dtype=np.float32)]
            mock_ids = np.array([[23]])
            mock_detect.return_value = (mock_corners, mock_ids, None)
            
            markers = detector.detect_markers(frame, estimate_pose=False)
            
            assert len(markers) == 1
            marker = markers[0]
            assert marker.id == 23
            assert marker.rvec is None
            assert marker.tvec is None
            assert marker.distance is None

    def test_filter_markers_by_confidence(self, sample_camera_matrix, sample_distortion_coeffs):
        """Test filtering markers by detection confidence."""
        detector = ArucoDetector(
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            marker_size=0.05,
            min_marker_area=100  # Minimum area for detection
        )
        
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        with patch('cv2.aruco.detectMarkers') as mock_detect:
            # Small marker (below threshold)
            small_corners = [np.array([[[100, 100], [105, 100], [105, 105], [100, 105]]], dtype=np.float32)]
            # Large marker (above threshold)  
            large_corners = [np.array([[[200, 200], [250, 200], [250, 250], [200, 250]]], dtype=np.float32)]
            
            mock_corners = small_corners + large_corners
            mock_ids = np.array([[23], [42]])
            mock_detect.return_value = (mock_corners, mock_ids, None)
            
            markers = detector.detect_markers(frame, estimate_pose=False)
            
            # Should only return the large marker
            assert len(markers) == 1
            assert markers[0].id == 42

    def test_detected_marker_properties(self):
        """Test DetectedMarker class properties."""
        corners = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], dtype=np.float32)
        rvec = np.array([0.1, 0.2, 0.3])
        tvec = np.array([0.1, 0.2, 0.5])
        
        marker = DetectedMarker(
            id=23,
            corners=corners,
            rvec=rvec,
            tvec=tvec
        )
        
        assert marker.id == 23
        assert np.array_equal(marker.corners, corners)
        assert np.array_equal(marker.rvec, rvec)
        assert np.array_equal(marker.tvec, tvec)
        
        # Test distance calculation
        expected_distance = np.linalg.norm(tvec) * 1000  # Convert to mm
        assert abs(marker.distance - expected_distance) < 1e-6
        
        # Test center calculation
        expected_center = np.mean(corners, axis=0)
        assert np.allclose(marker.center, expected_center)

    def test_detector_with_different_dictionaries(self, sample_camera_matrix, sample_distortion_coeffs):
        """Test detector with different ArUco dictionaries."""
        # Test with different dictionary
        detector = ArucoDetector(
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            marker_size=0.05,
            aruco_dict=cv2.aruco.DICT_4X4_100
        )
        
        assert detector.aruco_dict.getBytesList() is not None

    def test_error_handling_invalid_frame(self, sample_camera_matrix, sample_distortion_coeffs):
        """Test error handling with invalid frame input."""
        detector = ArucoDetector(
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            marker_size=0.05
        )
        
        # Test with None frame
        markers = detector.detect_markers(None)
        assert markers == []
        
        # Test with empty frame
        empty_frame = np.array([])
        markers = detector.detect_markers(empty_frame)
        assert markers == []

    @patch('cv2.aruco.detectMarkers')
    def test_detection_with_refinement(self, mock_detect, sample_camera_matrix, sample_distortion_coeffs):
        """Test detection with sub-pixel refinement."""
        detector = ArucoDetector(
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            marker_size=0.05
        )
        
        # Mock detection results
        mock_corners = [np.array([[[100, 100], [150, 100], [150, 150], [100, 150]]], dtype=np.float32)]
        mock_ids = np.array([[23]])
        mock_rejected = []
        mock_detect.return_value = (mock_corners, mock_ids, mock_rejected)
        
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        with patch('cv2.aruco.refineDetectedMarkers') as mock_refine:
            mock_refine.return_value = (mock_corners, mock_ids, mock_rejected)
            
            markers = detector.detect_markers(frame, refine_corners=True)
            
            # Verify refinement was called
            mock_refine.assert_called_once()
            assert len(markers) == 1
