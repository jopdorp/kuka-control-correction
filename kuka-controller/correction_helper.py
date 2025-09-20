"""
KUKA Vision Correction Helper for Windows

TCP server that receives vision corrections from Raspberry Pi and
writes KRL global variables via KUKAVARPROXY for continuous base correction.

Requirements:
- KUKAVARPROXY running on KRC2 Windows PC
- Python 3.x with socket support
- Network connectivity to Raspberry Pi

Usage:
    python correction_helper.py [--port 7001] [--kuka-ip 127.0.0.1]
"""

import socket
import json
import logging
import time
import threading
import argparse
from typing import Dict, Any, Optional
import struct


class KukaVarProxy:
    """Simple client for KUKAVARPROXY variable read/write."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7000):
        self.host = host
        self.port = port
        self.socket = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """Connect to KUKAVARPROXY."""
        try:
            if self.socket:
                self.socket.close()
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2.0)
            self.socket.connect((self.host, self.port))
            self.logger.info(f"Connected to KUKAVARPROXY at {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to KUKAVARPROXY: {e}")
            self.socket = None
            return False
    
    def disconnect(self):
        """Disconnect from KUKAVARPROXY."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
    
    def write_frame(self, var_name: str, x: float, y: float, z: float, 
                   a: float, b: float, c: float) -> bool:
        """Write a FRAME variable to KRL."""
        frame_str = f"{{X {x:.3f},Y {y:.3f},Z {z:.3f},A {a:.3f},B {b:.3f},C {c:.3f}}}"
        return self.write_variable(var_name, frame_str)
    
    def write_bool(self, var_name: str, value: bool) -> bool:
        """Write a BOOL variable to KRL."""
        bool_str = "TRUE" if value else "FALSE"
        return self.write_variable(var_name, bool_str)
    
    def write_int(self, var_name: str, value: int) -> bool:
        """Write an INT variable to KRL."""
        return self.write_variable(var_name, str(value))
    
    def write_real(self, var_name: str, value: float) -> bool:
        """Write a REAL variable to KRL."""
        return self.write_variable(var_name, f"{value:.6f}")
    
    def write_variable(self, var_name: str, value_str: str) -> bool:
        """Write a variable to KRL via KUKAVARPROXY protocol."""
        if not self.socket:
            if not self.connect():
                return False
        
        try:
            # KUKAVARPROXY protocol: 2-byte length + variable_name + value
            message = f"{var_name} {value_str}"
            msg_bytes = message.encode('ascii')
            length_bytes = struct.pack('<H', len(msg_bytes))  # Little-endian 16-bit length
            
            self.socket.sendall(length_bytes + msg_bytes)
            
            # Read response (2-byte length + response)
            resp_len_bytes = self.socket.recv(2)
            if len(resp_len_bytes) != 2:
                raise Exception("Failed to read response length")
            
            resp_len = struct.unpack('<H', resp_len_bytes)[0]
            if resp_len > 0:
                response = self.socket.recv(resp_len).decode('ascii')
                if "OK" not in response:
                    self.logger.warning(f"KUKAVARPROXY response: {response}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write {var_name}: {e}")
            self.disconnect()
            return False


class VisionCorrectionHelper:
    """TCP server for vision corrections."""
    
    def __init__(self, listen_port: int = 7001, kuka_ip: str = "127.0.0.1"):
        self.listen_port = listen_port
        self.kuka_proxy = KukaVarProxy(kuka_ip, 7000)
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.server_socket = None
        self.stats = {
            'corrections_received': 0,
            'kuka_writes_ok': 0,
            'kuka_writes_failed': 0,
            'last_correction_time': 0
        }
    
    def start_server(self):
        """Start TCP server for vision corrections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', self.listen_port))
            self.server_socket.listen(5)
            self.running = True
            
            self.logger.info(f"Vision correction helper listening on port {self.listen_port}")
            
            while self.running:
                try:
                    client_socket, client_addr = self.server_socket.accept()
                    self.logger.info(f"Client connected from {client_addr}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        self.logger.error(f"Server socket error: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop TCP server."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None
        self.kuka_proxy.disconnect()
    
    def handle_client(self, client_socket: socket.socket, client_addr):
        """Handle connected client (Raspberry Pi)."""
        buffer = ""
        
        try:
            client_socket.settimeout(5.0)
            
            while self.running:
                try:
                    data = client_socket.recv(4096).decode('utf-8')
                    if not data:
                        break
                    
                    buffer += data
                    
                    # Process complete JSON lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self.process_correction_message(line)
                            
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"Client handling error: {e}")
                    break
                    
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            self.logger.info(f"Client {client_addr} disconnected")
    
    def process_correction_message(self, json_line: str):
        """Process a vision correction message and update KRL variables."""
        try:
            message = json.loads(json_line)
            
            if message.get('type') != 'BASE_CORRECTION':
                return
            
            correction = message.get('correction', {})
            translation = correction.get('translation', {})
            rotation = correction.get('rotation', {})
            
            # Extract correction values
            x = float(translation.get('x', 0.0))
            y = float(translation.get('y', 0.0))
            z = float(translation.get('z', 0.0))
            rx = float(rotation.get('rx', 0.0))
            ry = float(rotation.get('ry', 0.0))
            rz = float(rotation.get('rz', 0.0))
            
            confidence = float(message.get('confidence', 0.0))
            sequence_id = int(message.get('sequence_id', 0))
            
            self.logger.debug(f"Correction #{sequence_id}: "
                            f"T=[{x:.2f},{y:.2f},{z:.2f}] "
                            f"R=[{rx:.3f},{ry:.3f},{rz:.3f}] "
                            f"conf={confidence:.2f}")
            
            # Write to KRL variables via KUKAVARPROXY
            success = True
            success &= self.kuka_proxy.write_frame("G_CORR_RAW", x, y, z, rx, ry, rz)
            success &= self.kuka_proxy.write_bool("G_CORR_VALID", True)
            
            # Update statistics
            self.stats['corrections_received'] += 1
            if success:
                self.stats['kuka_writes_ok'] += 1
            else:
                self.stats['kuka_writes_failed'] += 1
            self.stats['last_correction_time'] = time.time()
            
            if success:
                self.logger.info(f"Applied correction #{sequence_id} "
                               f"(pos_mag={(x**2 + y**2 + z**2)**0.5:.2f}mm)")
            else:
                self.logger.error(f"Failed to write correction #{sequence_id} to KUKA")
                
        except Exception as e:
            self.logger.error(f"Failed to process correction message: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.stats.copy()
        stats['uptime'] = time.time() - self.stats.get('start_time', time.time())
        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="KUKA Vision Correction Helper")
    parser.add_argument('--port', type=int, default=7001,
                       help='TCP port to listen for corrections (default: 7001)')
    parser.add_argument('--kuka-ip', type=str, default='127.0.0.1',
                       help='IP address of KUKAVARPROXY (default: 127.0.0.1)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create and start helper
    helper = VisionCorrectionHelper(args.port, args.kuka_ip)
    
    try:
        logger.info("Starting KUKA Vision Correction Helper")
        logger.info(f"Listening on port {args.port}")
        logger.info(f"KUKAVARPROXY at {args.kuka_ip}:7000")
        logger.info("Press Ctrl+C to stop")
        
        helper.start_server()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        helper.stop_server()
        logger.info("Helper stopped")


if __name__ == "__main__":
    main()
