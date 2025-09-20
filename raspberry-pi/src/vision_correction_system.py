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

from aruco_detector import ArucoDetector, CameraPose
from pose_utils import MoveCommand, RobotPose, pose_to_T, rotation_matrix_to_kuka_abc


@dataclass
class SystemConfig:
    """Configuration for the vision correction system."""
    # Camera settings
    camera_index: int = 0
    camera_resolution: tuple = (640, 480)
    camera_fps: int = 30
    
    # ArUco settings
    aruco_dictionary: int = cv2.aruco.DICT_6X6_250
    marker_size: float = 0.05  # meters
    
    # Robot settings
    robot_model: str = "KR6_R900"
    
    # Communication settings
    controller_ip: str = "127.0.0.1"     # Helper TCP server IP (helper runs locally on the Pi)
    controller_port: int = 7001           # Helper TCP port for corrections
    
    # Correction settings
    position_threshold: float = 0.1  # mm
    rotation_threshold: float = 0.01  # degrees
    confidence_threshold: float = 0.8
    max_correction: float = 5.0  # mm
    
    # Timing settings
    processing_rate: float = 30.0  # Hz
    communication_timeout: float = 1.0  # seconds


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
        
        # Initialize components
        self.aruco_detector = ArucoDetector(
            dictionary_type=config.aruco_dictionary,
            marker_size=config.marker_size
        )
        
        # TOOL->CAM transform (to be configured)
        self.T_tool_cam = np.eye(4)
        
        # Camera setup
        self.camera = None
        self.camera_running = False
        
        # Communication
        self.tcp_socket = None            # TCP client to controller helper
        self.communication_running = False
        self.message_queue = Queue()
        
        # Processing
        self.processing_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # State tracking
        self.sequence_counter = 0
        self.last_correction_time = 0
        self.system_statistics = {
            'frames_processed': 0,
            'markers_detected': 0,
            'corrections_sent': 0,
            'communication_errors': 0
        }
        
        self.logger.info("Vision correction system initialized")
    
    def load_configuration_files(self, 
                               camera_calibration_file: str,
                               marker_positions_file: str,
                               robot_config_file: str) -> bool:
        """
        Load configuration files for the system.
        
        Args:
            camera_calibration_file: Path to camera calibration file
            marker_positions_file: Path to marker positions file
            robot_config_file: Path to robot configuration file
            
        Returns:
            True if all files loaded successfully
        """
        try:
            # Load camera calibration
            if not self.aruco_detector.load_camera_calibration(camera_calibration_file):
                return False
            
            # Load marker positions
            with open(marker_positions_file, 'r') as f:
                marker_data = json.load(f)
                self.aruco_detector.load_marker_positions(marker_data['markers'])
            
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
            
            self.logger.info("Configuration files loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration files: {e}")
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
        """Start camera capture."""
        try:
            self.camera = cv2.VideoCapture(self.config.camera_index)
            
            if not self.camera.isOpened():
                self.logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.start()
            
            self.logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            return False
    
    def _stop_camera(self):
        """Stop camera capture."""
        self.camera_running = False
        
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join()
        
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def _camera_loop(self):
        """Camera capture loop."""
        while self.camera_running:
            ret, frame = self.camera.read()
            
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            else:
                self.logger.warning("Failed to capture frame")
            
            time.sleep(1.0 / self.config.camera_fps)
    
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
        """Communication loop: send corrections via TCP to KUKA controller."""
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
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        # Detect ArUco markers
        detected_markers = self.aruco_detector.detect_markers(frame)
        self.system_statistics['markers_detected'] += len(detected_markers)
        
        if not detected_markers:
            self.logger.debug("No markers detected")
            return None
            
        # Calculate correction from detected markers (continuous mode)
        correction = self._calculate_base_correction(detected_markers)
        return correction
    
    def _process_frame_with_command(self, move_command: MoveCommand) -> Optional[CorrectionData]:
        """Process current frame with move command."""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        # Detect ArUco markers
        detected_markers = self.aruco_detector.detect_markers(frame)
        self.system_statistics['markers_detected'] += len(detected_markers)
        
        if not detected_markers:
            self.logger.debug("No markers detected")
            return None
        
        # Estimate actual camera pose from markers
        actual_camera_pose = self.aruco_detector.estimate_camera_pose(detected_markers)
        
        if not actual_camera_pose or actual_camera_pose.confidence < self.config.confidence_threshold:
            self.logger.debug("Low confidence camera pose estimation")
            return None
        
        # Estimate expected camera pose from move command
        T_base_tcp = pose_to_T(move_command.target_pose.translation, move_command.target_pose.rotation)
        T_base_cam_expected = T_base_tcp @ self.T_tool_cam
        expected_camera_pose = RobotPose(
            translation=T_base_cam_expected[:3, 3],
            rotation=rotation_matrix_to_kuka_abc(T_base_cam_expected[:3, :3]),
            rotation_matrix=T_base_cam_expected[:3, :3],
            frame="BASE",
        )
        
        # Calculate correction
        return self._calculate_correction(expected_camera_pose, actual_camera_pose)
    
    def _calculate_correction(
        self,
        expected_pose: RobotPose,
        actual_pose: CameraPose,
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
        """Get current frame with detected markers drawn."""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        # Detect and draw markers
        detected_markers = self.aruco_detector.detect_markers(frame)
        return self.aruco_detector.draw_detected_markers(frame, detected_markers)


def main():
    """Main function for standalone operation."""
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = SystemConfig()
    
    # Create and start system
    system = VisionCorrectionSystem(config)
    
    # Load configuration files (optional - can be hardcoded in SystemConfig)
    # if not system.load_configuration_files(
    #     'camera_calibration.npz',
    #     'marker_positions.json', 
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
