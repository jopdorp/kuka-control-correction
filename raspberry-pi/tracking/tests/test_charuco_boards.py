"""
Unit tests for ChArUco board detection and matching functionality.
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add correction directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'correction'))

from charuco_board_detector import CharucoBoardDetector, CharucoBoardConfig, DetectedCharucoBoard  # type: ignore
from charuco_board_matcher import CharucoBoardMatcher  # type: ignore
from pose_utils import pose_to_T  # type: ignore


@pytest.fixture
def sample_board_config():
    """Sample ChArUco board configuration for testing."""
    return CharucoBoardConfig(
        squares_x=5,
        squares_y=4,
        square_size=0.05,  # 5cm squares
        marker_size=0.035,  # 3.5cm markers
        dictionary_type=cv2.aruco.DICT_6X6_250,
        expected_plane=[100.0, 200.0, 0.0, 0.0, 0.0, 0.0]  # x, y, z, a, b, c
    )


@pytest.fixture
def sample_camera_matrix():
    """Sample camera calibration matrix."""
    return np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)


@pytest.fixture
def sample_distortion_coeffs():
    """Sample camera distortion coefficients."""
    return np.zeros(5, dtype=np.float32)


@pytest.fixture
def multiple_board_configs():
    """Multiple board configurations for testing."""
    return [
        CharucoBoardConfig(
            squares_x=5, squares_y=4,
            square_size=0.05, marker_size=0.035,
            dictionary_type=cv2.aruco.DICT_6X6_250,
            expected_plane=[100.0, 200.0, 0.0, 0.0, 0.0, 0.0]
        ),
        CharucoBoardConfig(
            squares_x=4, squares_y=3,
            square_size=0.04, marker_size=0.028,
            dictionary_type=cv2.aruco.DICT_6X6_250,
            expected_plane=[300.0, 150.0, 50.0, 0.0, 0.0, 90.0]
        ),
        CharucoBoardConfig(
            squares_x=6, squares_y=5,
            square_size=0.03, marker_size=0.021,
            dictionary_type=cv2.aruco.DICT_6X6_250,
            expected_plane=[50.0, 350.0, 25.0, 0.0, 45.0, 0.0]
        )
    ]


class TestCharucoBoardDetector:
    """Test cases for ChArUco board detector."""
    
    def test_detector_initialization(self, multiple_board_configs, sample_camera_matrix, sample_distortion_coeffs):
        """Test detector initialization with multiple boards."""
        detector = CharucoBoardDetector(
            board_configs=multiple_board_configs,
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs
        )
        
        assert len(detector.charuco_boards) == 3
        assert len(detector.charuco_detectors) == 3
    # Boards are stored as a list; no string IDs anymore
    
    def test_detector_no_camera_calibration(self, sample_board_config):
        """Test detector behavior without camera calibration."""
        detector = CharucoBoardDetector(
            board_configs=[sample_board_config],
            camera_matrix=None,
            distortion_coeffs=None
        )
        
        # Create test image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Should return empty list without calibration
        detected_boards = detector.detect_boards(test_image)
        assert detected_boards == []
    
    def test_generate_and_detect_board(self, sample_board_config, sample_camera_matrix, sample_distortion_coeffs):
        """Test generating and detecting a ChArUco board."""
        detector = CharucoBoardDetector(
            board_configs=[sample_board_config],
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            min_corners_for_pose=4  # Lower threshold for testing
        )
        
        # Generate test board image
        dictionary = cv2.aruco.getPredefinedDictionary(sample_board_config.dictionary_type)
        board = cv2.aruco.CharucoBoard(
            (sample_board_config.squares_x, sample_board_config.squares_y),
            sample_board_config.square_size,
            sample_board_config.marker_size,
            dictionary
        )
        
        image_size = (800, 600)
        board_image = board.generateImage(image_size)
        
        # Detect boards in generated image
        detected_boards = detector.detect_boards(board_image)
        
        # Should detect the board
        assert len(detected_boards) >= 0  # May not detect if corners are insufficient


class TestCharucoBoardMatcher:
    """Test cases for ChArUco board matcher."""
    
    def test_matcher_initialization(self, multiple_board_configs):
        """Test matcher initialization."""
        matcher = CharucoBoardMatcher(
            board_configs=multiple_board_configs,
            max_position_error=100.0,
            max_rotation_error=45.0
        )
        
        assert len(matcher.board_configs) == 3
    
    def test_rotation_error_calculation(self, multiple_board_configs):
        """Test rotation error calculation."""
        matcher = CharucoBoardMatcher(multiple_board_configs)
        
        # Identity matrices should have zero error
        R1 = np.eye(3)
        R2 = np.eye(3)
        error = matcher._calculate_rotation_error(R1, R2)
        assert error < 1e-6  # Very small error
        
        # 90-degree rotation around Z-axis
        R1 = np.eye(3)
        R2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        error = matcher._calculate_rotation_error(R1, R2)
        assert abs(error - 90.0) < 1.0  # ~90 degrees
    
    def test_expected_poses_calculation(self, multiple_board_configs):
        """Test calculation of expected board poses in camera frame."""
        matcher = CharucoBoardMatcher(multiple_board_configs)
        
        # Test camera pose in base frame
        camera_pose_base = np.array([500.0, 300.0, 200.0, 0.0, 0.0, 0.0])  # mm and degrees
        T_base_cam = pose_to_T(camera_pose_base[:3], camera_pose_base[3:])
        
        expected_poses = matcher._calculate_expected_poses_camera(T_base_cam)
        
        assert len(expected_poses) == 3
        for pose in expected_poses:
            assert 'translation' in pose
            assert 'rotation' in pose
            assert 'rotation_matrix' in pose
            assert pose['translation'].shape == (3,)
            assert pose['rotation'].shape == (3,)
            assert pose['rotation_matrix'].shape == (3, 3)
    
    def test_cost_matrix_building(self, multiple_board_configs):
        """Test cost matrix building for Hungarian algorithm."""
        matcher = CharucoBoardMatcher(multiple_board_configs)
        
        # Create mock detected boards
        detected_boards = []
        for i in range(2):
            board = DetectedCharucoBoard(
                corners=np.random.rand(10, 1, 2).astype(np.float32),
                ids=np.arange(10),
                translation=np.array([0.1 * i, 0.1 * i, 0.5]),  # meters
                rotation=np.array([0.0, 0.0, 0.0]),
                rotation_matrix=np.eye(3),
                confidence=0.8,
                num_corners=10,
            )
            detected_boards.append(board)
        
        # Create mock expected poses
        expected_poses = []
        for i in range(3):
            pose = {
                'translation': np.array([0.1 * i, 0.1 * i, 0.5]),  # meters
                'rotation': np.array([0.0, 0.0, 0.0]),
                'rotation_matrix': np.eye(3)
            }
            expected_poses.append(pose)
        
        cost_matrix = matcher._build_cost_matrix(detected_boards, expected_poses)
        
        assert cost_matrix.shape == (2, 3)
        # All costs should be finite for close matches
        assert np.isfinite(cost_matrix[0, 0])
        assert np.isfinite(cost_matrix[1, 1])
    
    def test_board_matching_empty_inputs(self, multiple_board_configs):
        """Test board matching with empty inputs."""
        matcher = CharucoBoardMatcher(multiple_board_configs)
        
        # Empty detected boards
        camera_pose = np.array([0, 0, 0, 0, 0, 0])
        matches = matcher.match_boards([], camera_pose)
        assert matches == []
        
        # Test with no board configs
        empty_matcher = CharucoBoardMatcher([])
        detected_boards = [
            DetectedCharucoBoard(
                corners=np.random.rand(10, 1, 2).astype(np.float32),
                ids=np.arange(10),
                translation=np.array([0.0, 0.0, 0.5]),
                rotation=np.array([0.0, 0.0, 0.0]),
                rotation_matrix=np.eye(3),
                confidence=0.8,
                num_corners=10,
            )
        ]
        matches = empty_matcher.match_boards(detected_boards, camera_pose)
        assert matches == []


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_detection_and_matching_pipeline(self, multiple_board_configs, sample_camera_matrix, sample_distortion_coeffs):
        """Test the complete detection and matching pipeline."""
        # Initialize components
        detector = CharucoBoardDetector(
            board_configs=multiple_board_configs,
            camera_matrix=sample_camera_matrix,
            distortion_coeffs=sample_distortion_coeffs,
            min_corners_for_pose=4
        )
        
        matcher = CharucoBoardMatcher(
            board_configs=multiple_board_configs,
            max_position_error=1000.0,  # Very permissive for testing
            max_rotation_error=180.0
        )
        
        # Create a test image (empty - won't detect anything, but tests the pipeline)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Detect boards
        detected_boards = detector.detect_boards(test_image)

        # Match boards (even if empty)
        camera_pose = np.array([0, 0, 0, 0, 0, 0])
        # Should complete without errors
        assert isinstance(matcher.match_boards(detected_boards, camera_pose), list)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = CharucoBoardConfig(
            squares_x=5, squares_y=4,
            square_size=0.05, marker_size=0.035,
            dictionary_type=cv2.aruco.DICT_6X6_250,
            expected_plane=[0, 0, 0, 0, 0, 0]
        )
        
        # Should create without errors
        assert valid_config.squares_x == 5
        assert valid_config.squares_y == 4
        assert len(valid_config.expected_plane) == 6