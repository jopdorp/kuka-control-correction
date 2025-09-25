"""
Camera Source Abstractions

This module provides different camera source implementations that can be used
interchangeably with the vision correction system.
"""

import cv2
import numpy as np
import abc
import threading
import time
import socket
import json
import os
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from queue import Queue, Empty
from pathlib import Path


@dataclass 
class CameraConfig:
    """Configuration for camera sources."""
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    
    # For network sources
    host: str = "127.0.0.1"
    port: int = 8080
    
    # For physical cameras
    device_index: int = 0


class CameraSource(abc.ABC):
    """Abstract base class for all camera sources."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        self._frame_lock = threading.Lock()
        self._current_frame = None
        
    @abc.abstractmethod
    def start(self) -> bool:
        """Start the camera source."""
        pass
    
    @abc.abstractmethod 
    def stop(self):
        """Stop the camera source."""
        pass
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        with self._frame_lock:
            return self._current_frame.copy() if self._current_frame is not None else None
    
    def is_running(self) -> bool:
        """Check if camera source is running."""
        return self._running
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution."""
        return self.config.resolution


class PhysicalCameraSource(CameraSource):
    """Physical camera source using OpenCV VideoCapture."""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self.camera = None
        self.camera_thread = None
        
    def start(self) -> bool:
        """Start physical camera capture."""
        try:
            self.camera = cv2.VideoCapture(self.config.device_index)
            
            if not self.camera.isOpened():
                self.logger.error(f"Failed to open camera {self.config.device_index}")
                return False
            
            # Set camera properties
            width, height = self.config.resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Physical camera started: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            self._running = True
            self.camera_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.camera_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start physical camera: {e}")
            return False
    
    def stop(self):
        """Stop physical camera capture."""
        self._running = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
            
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def _capture_loop(self):
        """Camera capture loop."""
        while self._running and self.camera:
            ret, frame = self.camera.read()
            
            if ret:
                with self._frame_lock:
                    self._current_frame = frame
            else:
                self.logger.warning("Failed to capture frame from physical camera")
                
            time.sleep(1.0 / self.config.fps)


class TCPStreamCameraSource(CameraSource):
    """Camera source that receives frames via TCP connection from Rhino3D."""


class FileBatchCameraSource(CameraSource):
    """Camera source that watches a directory for new images from Rhino."""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self.watch_thread = None
        self.last_modified = 0
        
        # Create watch directory if it doesn't exist
        Path(self.config.watch_directory).mkdir(parents=True, exist_ok=True)
        
    def start(self) -> bool:
        """Start file watching."""
        try:
            self._running = True
            self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self.watch_thread.start()
            
            self.logger.info(f"File watcher started, watching: {self.config.watch_directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start file watcher: {e}")
            return False
    
    def stop(self):
        """Stop file watching."""
        self._running = False
        
        if self.watch_thread:
            self.watch_thread.join(timeout=2.0)
    
    def _watch_loop(self):
        """File watching loop."""
        while self._running:
            try:
                # Look for latest image file
                watch_path = Path(self.config.watch_directory)
                image_files = list(watch_path.glob(f"*{self.config.image_format}"))
                
                if image_files:
                    # Get the most recently modified image
                    latest_file = max(image_files, key=lambda f: f.stat().st_mtime)
                    latest_modified = latest_file.stat().st_mtime
                    
                    if latest_modified > self.last_modified:
                        # Load new image
                        frame = cv2.imread(str(latest_file))
                        if frame is not None:
                            with self._frame_lock:
                                self._current_frame = frame
                            self.last_modified = latest_modified
                            self.logger.debug(f"Loaded new image: {latest_file.name}")
                
                time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                if self._running:
                    self.logger.error(f"File watcher error: {e}")


class TCPStreamCameraSource(CameraSource):
    """Camera source that receives frames via TCP connection."""
    
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self.server_socket = None
        self.server_thread = None
        
    def start(self) -> bool:
        """Start TCP stream server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.config.host, self.config.port))
            self.server_socket.listen(1)
            
            self._running = True
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            
            self.logger.info(f"TCP stream server listening on {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start TCP stream server: {e}")
            return False
    
    def stop(self):
        """Stop TCP stream server."""
        self._running = False
        
        if self.server_socket:
            self.server_socket.close()
            
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
    
    def _server_loop(self):
        """TCP server loop to receive images."""
        while self._running:
            try:
                client_socket, addr = self.server_socket.accept()
                self.logger.info(f"TCP connection from {addr}")
                
                while self._running:
                    try:
                        # Read message header (length of image data)
                        header = client_socket.recv(8)
                        if not header:
                            break
                            
                        data_length = int.from_bytes(header, byteorder='big')
                        
                        # Read image data
                        image_data = b''
                        while len(image_data) < data_length:
                            chunk = client_socket.recv(min(4096, data_length - len(image_data)))
                            if not chunk:
                                break
                            image_data += chunk
                        
                        if len(image_data) == data_length:
                            # Decode image
                            nparr = np.frombuffer(image_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                with self._frame_lock:
                                    self._current_frame = frame
                        
                    except Exception as e:
                        self.logger.error(f"TCP receive error: {e}")
                        break
                
                client_socket.close()
                
            except Exception as e:
                if self._running:
                    self.logger.error(f"TCP server error: {e}")


def create_camera_source(source_type: str, config: CameraConfig) -> CameraSource:
    """Factory function to create camera sources."""
    
    sources = {
        'physical': PhysicalCameraSource,
        'tcp_stream': TCPStreamCameraSource,
    }
    
    if source_type not in sources:
        raise ValueError(f"Unknown camera source type: {source_type}. Available: {list(sources.keys())}")
    
    return sources[source_type](config)
