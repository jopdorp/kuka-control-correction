"""
Vision Correction System Main Module

This module integrates ArUco detection, robot kinematics, and communication
to provide real-time position correction for KUKA robots.
"""

import cv2  # type: ignore
import numpy as np
import json
import logging
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
import socket
from queue import Queue, Empty
import collections

from pose_utils import MoveCommand, RobotPose, pose_to_T, rotation_matrix_to_kuka_abc
from charuco_board_detector import CharucoBoardDetector, CharucoBoardConfig, DetectedCharucoBoard
from charuco_board_matcher import CharucoBoardMatcher
from charuco_board_inference import infer_charuco_board_config_from_image
from camera_sources import CameraSource, CameraConfig, create_camera_source


@dataclass
class SystemConfig:
    """Configuration for the vision correction system."""
    # ChArUco board settings (required)
    charuco_boards_config_file: str = 'charuco_boards_config.json'  # Path to ChArUco boards configuration file (required)
    
    # Camera settings
    camera_source_type: str = 'physical'  # 'physical' or 'tcp_stream'
    camera_index: int = 0                 # For physical cameras
    camera_resolution: tuple = (1920, 1080)
    camera_fps: int = 30
    
    # Network camera settings (for tcp_stream source)
    camera_host: str = "127.0.0.1"
    camera_port: int = 8080
    
    # Robot settings
    robot_model: str = "KR250_R2700-2"

    # Communication settings
    controller_ip: str = "127.0.0.1"     # Helper TCP server IP (helper runs locally on the Pi)
    controller_port: int = 7001           # Helper TCP port for corrections
    
    # Correction settings
    position_threshold: float = 0.1  # mm
    rotation_threshold: float = 0.01  # degrees
    confidence_threshold: float = 0.8
    max_correction: float = 5.0  # mm
    
    # Board matching settings
    max_board_position_error: float = 50.0  # mm
    max_board_rotation_error: float = 30.0  # degrees
    board_position_weight: float = 1.0
    board_rotation_weight: float = 0.1
    
    # Timing settings
    processing_rate: float = 30.0  # Hz
    communication_timeout: float = 1.0  # seconds
    # Buffering and sync
    buffer_size: int = 200  # number of samples to keep in deques
    max_time_skew: float = 0.5  # seconds, allowable difference between frame and controller state
    
    @classmethod
    def from_json(cls, config_file: str) -> 'SystemConfig':
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            SystemConfig instance
        """
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Create config with file-specified values
        config = cls(
            charuco_boards_config_file=config_data.get('files', {}).get('charuco_boards_config', 'charuco_boards_config.json')
        )
        
        # Update camera settings
        camera_config = config_data.get('camera', {})
        config.camera_source_type = camera_config.get('source_type', config.camera_source_type)
        config.camera_index = camera_config.get('index', config.camera_index)
        config.camera_resolution = tuple(camera_config.get('resolution', config.camera_resolution))
        config.camera_fps = camera_config.get('fps', config.camera_fps)
        config.camera_host = camera_config.get('host', config.camera_host)
        config.camera_port = camera_config.get('port', config.camera_port)
        
        # Update robot settings
        robot_config = config_data.get('robot', {})
        config.robot_model = robot_config.get('model', config.robot_model)
        
        # Update communication settings
        comm_config = config_data.get('communication', {})
        config.controller_ip = comm_config.get('controller_ip', config.controller_ip)
        config.controller_port = comm_config.get('controller_port', config.controller_port)
        config.communication_timeout = comm_config.get('timeout', config.communication_timeout)
        
        # Update vision correction settings
        vision_config = config_data.get('vision_correction', {})
        config.position_threshold = vision_config.get('position_threshold', config.position_threshold)
        config.rotation_threshold = vision_config.get('rotation_threshold', config.rotation_threshold)
        config.confidence_threshold = vision_config.get('confidence_threshold', config.confidence_threshold)
        config.max_correction = vision_config.get('max_correction', config.max_correction)
        
        # Update board matching settings
        matching_config = config_data.get('board_matching', {})
        config.max_board_position_error = matching_config.get('max_position_error', config.max_board_position_error)
        config.max_board_rotation_error = matching_config.get('max_rotation_error', config.max_board_rotation_error)
        config.board_position_weight = matching_config.get('position_weight', config.board_position_weight)
        config.board_rotation_weight = matching_config.get('rotation_weight', config.board_rotation_weight)
        
        # Update processing settings
        processing_config = config_data.get('processing', {})
        config.processing_rate = processing_config.get('rate', config.processing_rate)
        config.buffer_size = processing_config.get('buffer_size', config.buffer_size)
        config.max_time_skew = processing_config.get('max_time_skew', config.max_time_skew)
        
        return config


@dataclass
class CorrectionData:
    """Position correction data to send to KUKA."""
    translation_correction: np.ndarray  # [dx, dy, dz] in mm
    rotation_correction: np.ndarray     # [da, db, dc] in degrees
    confidence: float
    timestamp: float
    sequence_id: int


class VisionCorrectionSystem:
    """
    Main vision correction system that coordinates all components.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the vision correction system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize ChArUco board components
        self.charuco_detector = None
        self.charuco_matcher = None
        self.charuco_board_configs = []

        # TOOL->CAM transform (to be configured)
        self.T_tool_cam = np.eye(4)

        # Camera setup
        self.camera_source = None
        self.camera_running = False

        # Communication
        self.tcp_socket = None  # TCP client to controller helper
        self.communication_running = False
        self.message_queue = Queue()
        self.latest_controller_state = None
        self.controller_state_lock = threading.Lock()
        self.controller_state_buffer = collections.deque(maxlen=self.config.buffer_size)

        # Processing
        self.processing_running = False
        self.camera_pose_buffer = collections.deque(maxlen=self.config.buffer_size)

        # State tracking
        self.sequence_counter = 0
        self.last_correction_time = 0
        self.system_statistics = {
            'frames_processed': 0,
            'markers_detected': 0,
            'corrections_sent': 0,
            'communication_errors': 0,
        }

        self.logger.info("Vision correction system initialized")
    
    def load_configuration_files(self, 
                               camera_calibration_file: str,
                               robot_config_file: str) -> bool:
        """
        Load configuration files for the system.
        
        Args:
            camera_calibration_file: Path to camera calibration file
            robot_config_file: Path to robot configuration file
            
        Returns:
            True if all files loaded successfully
        """
        try:
            # Load camera calibration first (needed for ChArUco detector)
            calibration_data = np.load(camera_calibration_file)
            camera_matrix = calibration_data['camera_matrix']
            distortion_coeffs = calibration_data['distortion_coeffs']
            self.logger.info(f"Camera calibration loaded from {camera_calibration_file}")
            
            # Load robot configuration
            with open(robot_config_file, 'r') as f:
                robot_data = json.load(f)
                # TOOL->CAM offset in mm/deg
                if 'camera_tool_offset' in robot_data:
                    camera_offset = robot_data['camera_tool_offset']
                    self.T_tool_cam = pose_to_T(
                        np.array(camera_offset['translation']),
                        np.array(camera_offset['rotation'])
                    )
            
            # Load ChArUco board configuration (required)
            if not self.config.charuco_boards_config_file:
                raise ValueError("ChArUco boards configuration file is required")
                
            if not self._load_charuco_boards_config(self.config.charuco_boards_config_file, 
                                                   camera_matrix, distortion_coeffs):
                raise RuntimeError("Failed to load ChArUco boards configuration")
            
            self.logger.info("Configuration files loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration files: {e}")
            return False
    
    def _load_charuco_boards_config(self, config_file: str, 
                                  camera_matrix: np.ndarray,
                                  distortion_coeffs: np.ndarray) -> bool:
        """
        Load ChArUco boards configuration from file.
        
        Args:
            config_file: Path to ChArUco boards configuration file
            camera_matrix: Camera intrinsic matrix
            distortion_coeffs: Camera distortion coefficients
            
        Returns:
            True if successful
        """
        try:
            self.charuco_board_configs = []
            with open(config_file, 'r') as f:
                boards_data = json.load(f)

                board_image = boards_data.get('board_image', 'charuco_board.png')
                planes = boards_data.get('expected_planes', [])
                if not planes:
                    raise ValueError("expected_planes missing or empty in ChArUco config JSON")

                inferred = infer_charuco_board_config_from_image(board_image)

                # Clone per plane; identification is by index
                for plane in planes:
                    cfg = CharucoBoardConfig(
                        squares_x=inferred.squares_x,
                        squares_y=inferred.squares_y,
                        square_size=inferred.square_size,
                        marker_size=inferred.marker_size,
                        dictionary_type=inferred.dictionary_type,
                        expected_plane=plane,
                    )
                    self.charuco_board_configs.append(cfg)
        
            self.charuco_detector = CharucoBoardDetector(
                board_configs=self.charuco_board_configs,
                camera_matrix=camera_matrix,
                distortion_coeffs=distortion_coeffs
            )
            
            self.charuco_matcher = CharucoBoardMatcher(
                board_configs=self.charuco_board_configs,
                max_position_error=self.config.max_board_position_error,
                max_rotation_error=self.config.max_board_rotation_error,
                position_weight=self.config.board_position_weight,
                rotation_weight=self.config.board_rotation_weight
            )
            
            self.logger.info(f"Loaded {len(self.charuco_board_configs)} ChArUco board configurations")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load ChArUco boards config: {e}")
            return False
    
    def start_system(self) -> bool:
        """
        Start all system components.
        
        Returns:
            True if all components started successfully
        """
        try:
            # Start camera
            if not self._start_camera():
                return False
            
            # Start communication
            if not self._start_communication():
                self._stop_camera()
                return False
            
            # Start processing
            if not self._start_processing():
                self._stop_communication()
                self._stop_camera()
                return False
            
            self.logger.info("Vision correction system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.stop_system()
            return False
    
    def stop_system(self):
        """Stop all system components."""
        self.logger.info("Stopping vision correction system")
        
        self._stop_processing()
        self._stop_communication()
        self._stop_camera()
        
        self.logger.info("Vision correction system stopped")
    
    def _start_camera(self) -> bool:
        """Start camera source."""
        try:
            # Create camera configuration
            camera_config = CameraConfig(
                resolution=self.config.camera_resolution,
                fps=self.config.camera_fps,
                device_index=self.config.camera_index,
                host=self.config.camera_host,
                port=self.config.camera_port
            )
            
            # Create camera source based on configuration
            self.camera_source = create_camera_source(
                self.config.camera_source_type, 
                camera_config
            )
            
            if not self.camera_source.start():
                self.logger.error(f"Failed to start {self.config.camera_source_type} camera source")
                return False
            
            self.logger.info(f"Started {self.config.camera_source_type} camera source successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera source: {e}")
            return False
    
    def _stop_camera(self):
        """Stop camera source."""
        if self.camera_source:
            self.camera_source.stop()
            self.camera_source = None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame from camera source."""
        if self.camera_source:
            return self.camera_source.get_frame()
        return None
    
    def _start_communication(self) -> bool:
        """Start TCP communication to controller."""
        try:
            # TCP client for corrections
            self._connect_tcp()
            
            self.communication_running = True
            self.communication_thread = threading.Thread(target=self._communication_loop)
            self.communication_thread.start()
            
            self.logger.info(
                f"TCP communication started to {self.config.controller_ip}:{self.config.controller_port}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start communication: {e}")
            return False
    
    def _stop_communication(self):
        """Stop TCP communication."""
        self.communication_running = False
        
        if hasattr(self, 'communication_thread'):
            self.communication_thread.join()
        
        if self.tcp_socket:
            try:
                self.tcp_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.tcp_socket.close()
            self.tcp_socket = None
    
    def _communication_loop(self):
        """Communication loop: send corrections and receive controller state."""
        while self.communication_running:
            try:
                # Send queued messages
                while True:
                    message = self.message_queue.get_nowait()
                    self._send_tcp_json(message)
                    self.logger.debug(f"Sent correction: {message.get('sequence_id', -1)}")
            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Failed to send message: {e}")
                self.system_statistics['communication_errors'] += 1
            # Non-blocking receive for controller state
            try:
                if self.tcp_socket:
                    self.tcp_socket.settimeout(0.0)
                    data = self.tcp_socket.recv(8192)
                    if data:
                        for line in data.decode('utf-8', errors='ignore').split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                msg = json.loads(line)
                                if msg.get('type') == 'CONTROLLER_STATE':
                                    with self.controller_state_lock:
                                        self.latest_controller_state = msg
                                        self.controller_state_buffer.append(msg)
                            except Exception:
                                # ignore partial/garbage
                                pass
            except BlockingIOError:
                pass
            except Exception:
                # Ignore receive errors, sending continues
                pass
            
            time.sleep(0.01)  # Small sleep to prevent CPU spinning

    def _connect_tcp(self):
        """Connect TCP client to controller helper (with quick retry)."""
        try:
            if self.tcp_socket:
                return
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((self.config.controller_ip, self.config.controller_port))
            s.settimeout(0.5)
            self.tcp_socket = s
        except Exception as e:
            self.logger.warning(f"TCP connect failed: {e}")
            if self.tcp_socket:
                try:
                    self.tcp_socket.close()
                except Exception:
                    pass
                self.tcp_socket = None

    def _send_tcp_json(self, payload: Dict[str, Any]):
        data = (json.dumps(payload) + "\n").encode('utf-8')
        if not self.tcp_socket:
            self._connect_tcp()
        if not self.tcp_socket:
            raise RuntimeError("TCP not connected")
        try:
            self.tcp_socket.sendall(data)
        except Exception:
            # reconnect and retry once
            try:
                if self.tcp_socket:
                    self.tcp_socket.close()
            except Exception:
                pass
            self.tcp_socket = None
            self._connect_tcp()
            if not self.tcp_socket:
                raise
            self.tcp_socket.sendall(data)
    
    def _start_processing(self) -> bool:
        """Start vision processing."""
        try:
            self.processing_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.start()
            
            self.logger.info("Processing started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            return False
    
    def _stop_processing(self):
        """Stop vision processing."""
        self.processing_running = False
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
    
    def _processing_loop(self):
        """Main processing loop."""
        processing_interval = 1.0 / self.config.processing_rate
        
        while self.processing_running:
            start_time = time.time()
            
            try:
                # Process current frame for continuous correction
                correction = self._process_current_frame()
                
                if correction:
                    self._send_correction(correction)
                    self.system_statistics['corrections_sent'] += 1
                
                self.system_statistics['frames_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
            
            # Maintain processing rate
            elapsed = time.time() - start_time
            sleep_time = max(0, processing_interval - elapsed)
            time.sleep(sleep_time)
    
    def _parse_move_command(self, command_data: Dict[str, Any]) -> Optional[MoveCommand]:
        """Parse received move command."""
        try:
            cmd = command_data['command']
            target_pos = cmd['target_position']
            
            target_pose = RobotPose(
                translation=np.array([target_pos['x'], target_pos['y'], target_pos['z']]),
                rotation=np.array([target_pos['a'], target_pos['b'], target_pos['c']]),
                rotation_matrix=np.eye(3)  # Will be calculated in kinematics
            )
            
            return MoveCommand(
                command_type=cmd['type'],
                target_pose=target_pose,
                velocity=cmd.get('velocity', 100),
                acceleration=cmd.get('acceleration', 50)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse move command: {e}")
            return None
    
    def _process_current_frame(self) -> Optional[CorrectionData]:
        """Process current frame for continuous correction without move commands."""
        frame = self.get_current_frame()
        if frame is None:
            return None
            
        frame_ts = time.time()
        return self._process_charuco_boards(frame, frame_ts)
    
    def _process_charuco_boards(self, frame: np.ndarray, frame_ts: float) -> Optional[CorrectionData]:
        """Process frame using ChArUco boards."""
        # Get current camera pose from robot TCP
        with self.controller_state_lock:
            if not self.latest_controller_state:
                self.logger.debug("No controller state available")
                return None
            tcp = self.latest_controller_state.get('tcp', {})
        
        try:
            # Extract current camera pose in base frame
            t_mm = np.array([float(tcp.get('X', 0.0)), float(tcp.get('Y', 0.0)), float(tcp.get('Z', 0.0))])
            r_deg = np.array([float(tcp.get('A', 0.0)), float(tcp.get('B', 0.0)), float(tcp.get('C', 0.0))])
        except Exception:
            self.logger.debug("Invalid controller state format")
            return None

        # Calculate current camera pose from TCP + tool->cam transform
        T_base_tcp = pose_to_T(t_mm, r_deg)
        T_base_cam = T_base_tcp @ self.T_tool_cam
        camera_pose_base = np.concatenate([
            T_base_cam[:3, 3],  # translation in mm
            rotation_matrix_to_kuka_abc(T_base_cam[:3, :3])  # rotation in degrees
        ])
        
        # Detect ChArUco boards
        detected_boards = self.charuco_detector.detect_boards(frame)
        self.system_statistics['markers_detected'] += len(detected_boards)  # Reuse marker count for boards
        
        if not detected_boards:
            self.logger.debug("No ChArUco boards detected")
            return None
        
        # Match detected boards with expected configurations
        matched_results = self.charuco_matcher.match_boards(detected_boards, camera_pose_base)
        
        if not matched_results:
            self.logger.debug("No boards successfully matched")
            return None
        
        # Log successful matches
        for result in matched_results:
            self.logger.info(
                f"Matched board config[{result.config_index}] with detected[{result.detected_index}]: "
                f"pos_err={result.position_error:.1f}mm, rot_err={result.rotation_error:.1f}deg, "
                f"conf={result.detected_board.confidence:.2f}"
            )
        
        # Calculate correction using best matched board
        best_match = matched_results[0]  # Already sorted by error
        return self._calculate_charuco_correction(best_match, camera_pose_base)
    
    def _get_controller_state_for_timestamp(self, ts: Optional[float]) -> Optional[Dict[str, Any]]:
        """Return the controller state closest in time to ts within max_time_skew."""
        with self.controller_state_lock:
            if not self.controller_state_buffer:
                return None
            
            # If ts is None, return the latest controller state
            if ts is None:
                return dict(self.latest_controller_state) if self.latest_controller_state else None
            
            # Find nearest by absolute delta
            best = None
            best_dt = None
            for item in self.controller_state_buffer:
                st = item.get('timestamp')
                if st is None:
                    continue
                dt = abs(ts - float(st))
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best = item
            if best is not None and best_dt is not None and best_dt <= self.config.max_time_skew:
                return dict(best)
            return None
    
    def _calculate_correction(
        self,
        expected_pose: RobotPose,
        actual_pose: DetectedCharucoBoard,
    ) -> Optional[CorrectionData]:
        """Calculate position correction (BASE frame)."""
        # Normalize units and orientations
        actual_translation_mm = actual_pose.translation * 1000.0  # meters -> mm
        actual_abc_deg = rotation_matrix_to_kuka_abc(actual_pose.rotation_matrix)

        # Position error (mm) and rotation error (deg)
        position_error = actual_translation_mm - expected_pose.translation
        position_error_magnitude = np.linalg.norm(position_error)
        rotation_error = actual_abc_deg - expected_pose.rotation
        rotation_error_magnitude = np.linalg.norm(rotation_error)

        # Thresholds
        if (
            position_error_magnitude < self.config.position_threshold
            and rotation_error_magnitude < self.config.rotation_threshold
        ):
            return None

        # Clamp translation correction
        if position_error_magnitude > self.config.max_correction:
            scale = self.config.max_correction / position_error_magnitude
            position_error = position_error * scale

        self.sequence_counter += 1
        return CorrectionData(
            translation_correction=position_error,
            rotation_correction=rotation_error,
            confidence=actual_pose.confidence,
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
        )
    
    def _calculate_charuco_correction(self, 
                                    match_result,
                                    current_camera_pose: np.ndarray) -> Optional[CorrectionData]:
        """
        Calculate position correction using ChArUco board match.
        
        Args:
            match_result: Matched board result
            current_camera_pose: Current camera pose in base frame [x, y, z, a, b, c]
            
        Returns:
            Correction data or None if no correction needed
        """
        detected_board = match_result.detected_board
        expected_config = match_result.matched_config
        
        # Calculate expected board pose in camera frame from current camera pose
        T_base_cam = pose_to_T(current_camera_pose[:3], current_camera_pose[3:])
        T_cam_base = np.linalg.inv(T_base_cam)
        T_base_board_expected = pose_to_T(
            np.array(expected_config.expected_plane[:3]),
            np.array(expected_config.expected_plane[3:])
        )
        T_cam_board_expected = T_cam_base @ T_base_board_expected
        
        # Expected board pose in camera frame
        expected_translation = T_cam_board_expected[:3, 3] / 1000.0  # mm to meters
        expected_rotation_matrix = T_cam_board_expected[:3, :3]
        
        # Actual detected board pose in camera frame (already in meters)
        actual_translation = detected_board.translation
        actual_rotation_matrix = detected_board.rotation_matrix
        
        # Calculate position error (in mm)
        position_error_m = actual_translation - expected_translation
        position_error_mm = position_error_m * 1000.0
        position_error_magnitude = np.linalg.norm(position_error_mm)
        
        # Calculate rotation error (in degrees)
        rotation_error_rad = self._calculate_rotation_error_matrix(
            actual_rotation_matrix, expected_rotation_matrix
        )
        rotation_error_deg = np.degrees(rotation_error_rad)
        
        # Create rotation error vector (simplified - just magnitude)
        rotation_error_vector = np.array([0, 0, rotation_error_deg])
        rotation_error_magnitude = rotation_error_deg
        
        # Check if correction is needed
        if (position_error_magnitude < self.config.position_threshold and 
            rotation_error_magnitude < self.config.rotation_threshold):
            return None
        
        # Clamp translation correction
        if position_error_magnitude > self.config.max_correction:
            scale = self.config.max_correction / position_error_magnitude
            position_error_mm = position_error_mm * scale
        
        self.sequence_counter += 1
        return CorrectionData(
            translation_correction=position_error_mm,
            rotation_correction=rotation_error_vector,
            confidence=detected_board.confidence,
            timestamp=time.time(),
            sequence_id=self.sequence_counter,
        )
    
    def _calculate_rotation_error_matrix(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """
        Calculate rotation error between two rotation matrices in radians.
        
        Args:
            R1: Actual rotation matrix (3x3)
            R2: Expected rotation matrix (3x3)
            
        Returns:
            Rotation error angle in radians
        """
        try:
            # Relative rotation
            R_rel = R1.T @ R2
            
            # Calculate angle of rotation
            trace = np.trace(R_rel)
            # Clamp to valid range for acos
            trace = np.clip(trace, -1.0, 3.0)
            angle_rad = np.arccos((trace - 1) / 2)
            
            return angle_rad
        except Exception:
            # In case of numerical issues, return large error
            return np.pi
    
    def _send_correction(self, correction: CorrectionData):
        """Send correction to KUKA controller via TCP."""
        message = {
            'type': 'BASE_CORRECTION',
            'timestamp': correction.timestamp,
            'sequence_id': correction.sequence_id,
            'correction': {
                'translation': {
                    'x': float(correction.translation_correction[0]),
                    'y': float(correction.translation_correction[1]),
                    'z': float(correction.translation_correction[2])
                },
                'rotation': {
                    'rx': float(correction.rotation_correction[0]),
                    'ry': float(correction.rotation_correction[1]),
                    'rz': float(correction.rotation_correction[2])
                }
            },
            'confidence': correction.confidence
        }
        
        self.message_queue.put(message)
        self.last_correction_time = correction.timestamp
        
        self.logger.info(f"Correction calculated: pos_error={np.linalg.norm(correction.translation_correction):.2f}mm")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'camera_running': self.camera_running,
            'communication_running': self.communication_running,
            'processing_running': self.processing_running,
            'statistics': self.system_statistics.copy(),
            'last_correction_time': self.last_correction_time,
            'sequence_counter': self.sequence_counter
        }
    
    def get_current_frame_with_markers(self) -> Optional[np.ndarray]:
        """Get current frame with detected ChArUco boards drawn."""
        frame = self.get_current_frame()
        if frame is None:
            return None
        
        # Detect and draw ChArUco boards
        if self.charuco_detector:
            detected_boards = self.charuco_detector.detect_boards(frame)
            return self.charuco_detector.draw_detected_boards(frame, detected_boards)
        else:
            # Return original frame if detector not initialized
            return frame


def main():
    """Main function for standalone operation."""
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration - ChArUco boards config is required
    config = SystemConfig(charuco_boards_config_file='charuco_boards_config.json')
    
    # Create and start system
    system = VisionCorrectionSystem(config)
    
    # Load configuration files (camera calibration and robot config required)
    # if not system.load_configuration_files(
    #     'camera_calibration.npz',
    #     'robot_config.json'
    # ):
    #     print("Failed to load configuration files")
    #     return
    
    # Start system
    if not system.start_system():
        print("Failed to start system")
        return
    
    try:
        print("Vision correction system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            status = system.get_system_status()
            print(f"Status: {status['statistics']}")
    
    except KeyboardInterrupt:
        print("Stopping system...")
        system.stop_system()


if __name__ == "__main__":
    main()
