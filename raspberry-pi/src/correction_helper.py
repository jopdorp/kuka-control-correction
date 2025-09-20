"""
KUKA Vision Correction Helper (Pi-only)

Runs on the Raspberry Pi. Receives vision corrections (JSON lines) and writes
KRL global variables via KUKAVARPROXY running on the KUKA controller PC.

Usage:
    python correction_helper.py --port 7001 --kuka-ip <KUKA_PC_IP>
"""

import socket
import json
import logging
import time
import threading
import argparse
from typing import Dict, Any, Optional
import struct
import re


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

    def write_variable(self, var_name: str, value_str: str) -> bool:
        """Write a variable to KRL via KUKAVARPROXY protocol."""
        if not self.socket:
            if not self.connect():
                return False

        try:
            # KUKAVARPROXY protocol: 2-byte little-endian length + payload (ASCII)
            message = f"{var_name} {value_str}"
            msg_bytes = message.encode('ascii')
            length_bytes = struct.pack('<H', len(msg_bytes))

            self.socket.sendall(length_bytes + msg_bytes)

            # Read response (2-byte length + response ASCII)
            resp_len_bytes = self.socket.recv(2)
            if len(resp_len_bytes) != 2:
                raise Exception("Failed to read response length")

            resp_len = struct.unpack('<H', resp_len_bytes)[0]
            if resp_len > 0:
                response = self.socket.recv(resp_len).decode('ascii', errors='ignore')
                if "OK" not in response:
                    self.logger.warning(f"KUKAVARPROXY response: {response}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to write {var_name}: {e}")
            self.disconnect()
            return False

    def read_variable(self, var_name: str) -> Optional[str]:
        """Read a variable via KUKAVARPROXY protocol and return raw ASCII value string.

        Note: Variable must be GLOBAL in $CONFIG.DAT for reliable access.
        """
        if not self.socket:
            if not self.connect():
                return None

        try:
            # Reading: send length + var name only
            msg_bytes = var_name.encode('ascii')
            length_bytes = struct.pack('<H', len(msg_bytes))
            self.socket.sendall(length_bytes + msg_bytes)

            # Response: 2-byte length + ASCII payload
            resp_len_bytes = self.socket.recv(2)
            if len(resp_len_bytes) != 2:
                raise Exception("Failed to read response length")

            resp_len = struct.unpack('<H', resp_len_bytes)[0]
            if resp_len == 0:
                return ""
            response = self.socket.recv(resp_len).decode('ascii', errors='ignore')
            return response

        except Exception as e:
            self.logger.error(f"Failed to read {var_name}: {e}")
            self.disconnect()
            return None

    def read_frame_parsed(self, var_name: str) -> Optional[Dict[str, float]]:
        """Read a FRAME/E6POS-like string and parse X,Y,Z,A,B,C."""
        raw = self.read_variable(var_name)
        if raw is None:
            return None
        # Expect format like: {X 0.000,Y 0.000,Z 0.000,A 0.000,B 0.000,C 0.000,...}
        vals: Dict[str, float] = {}
        for key in ("X", "Y", "Z", "A", "B", "C"):
            m = re.search(rf"{key}\s+([-+]?\d+(?:\.\d+)?)", raw)
            if m:
                vals[key] = float(m.group(1))
        # Only return if we got at least X..C
        if all(k in vals for k in ("X","Y","Z","A","B","C")):
            return vals
        return None


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
            'last_correction_time': 0,
            'start_time': time.time(),
        }

    def start_server(self):
        """Start TCP server for vision corrections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', self.listen_port))
            self.server_socket.listen(5)
            self.running = True

            self.logger.info(f"Vision correction helper listening on 0.0.0.0:{self.listen_port}")

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
        """Handle connected client (vision system)."""
        buffer = ""
        stop_event = threading.Event()

        try:
            client_socket.settimeout(5.0)

            # Start background streamer of controller state
            streamer = threading.Thread(
                target=self._stream_controller_state,
                args=(client_socket, stop_event),
                daemon=True,
            )
            streamer.start()

            while self.running:
                try:
                    data = client_socket.recv(4096).decode('utf-8', errors='ignore')
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
            stop_event.set()
            self.logger.info(f"Client {client_addr} disconnected")

    def _stream_controller_state(self, client_socket: socket.socket, stop_event: threading.Event):
        """Periodically read controller state via KVP and send to the client as JSON lines."""
        while self.running and not stop_event.is_set():
            try:
                base = self.kuka_proxy.read_frame_parsed("G_BASE_ACTIVE")
                tcp = self.kuka_proxy.read_frame_parsed("G_TCP_ACT")
                if base and tcp:
                    payload = {
                        "type": "CONTROLLER_STATE",
                        "timestamp": time.time(),  # Pi time when sampled
                        "base": base,
                        "tcp": tcp,
                    }
                    line = (json.dumps(payload) + "\n").encode('utf-8')
                    client_socket.sendall(line)
            except Exception as e:
                # Non-fatal; keep trying
                self.logger.debug(f"State stream error: {e}")
            time.sleep(0.05)  # ~20 Hz

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

            self.logger.debug(
                f"Correction #{sequence_id}: T=[{x:.2f},{y:.2f},{z:.2f}] R=[{rx:.3f},{ry:.3f},{rz:.3f}] conf={confidence:.2f}"
            )

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
                self.logger.info(
                    f"Applied correction #{sequence_id} (pos_mag={(x**2 + y**2 + z**2)**0.5:.2f}mm)"
                )
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
    parser.add_argument('--kuka-ip', type=str, default='192.168.1.50',
                        help='IP address of KUKAVARPROXY on controller (default: 192.168.1.50)')
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
