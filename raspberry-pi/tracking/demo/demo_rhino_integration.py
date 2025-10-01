#!/usr/bin/env python3
"""
Rhino3D TCP Streaming Integration Demo

This script demonstrates how to integrate Rhino3D viewport streaming
with the KUKA vision correction system using TCP streaming.
"""

import json
import time
import socket
import threading
from typing import Optional


class DemoTCPServer:
    """Demo TCP server that simulates receiving images from Rhino."""
    
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        self.client_count = 0
        
    def start(self):
        """Start TCP server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            self.running = True
            server_thread = threading.Thread(target=self._server_loop, daemon=True)
            server_thread.start()
            
            print(f"TCP server listening on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to start TCP server: {e}")
            return False
    
    def stop(self):
        """Stop TCP server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
    
    def _server_loop(self):
        """TCP server loop."""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                self.client_count += 1
                print(f"Rhino connection #{self.client_count} from {addr}")
                
                self._handle_client(client_socket)
                client_socket.close()
                
            except Exception as e:
                if self.running:
                    print(f"Server error: {e}")
    
    def _handle_client(self, client_socket):
        """Handle client connection."""
        frame_count = 0
        
        while self.running:
            try:
                # Read message header (8 bytes - image length)
                header = client_socket.recv(8)
                if not header or len(header) != 8:
                    break
                    
                data_length = int.from_bytes(header, byteorder='big')
                print(f"Receiving image: {data_length} bytes")
                
                # Read image data
                received = 0
                while received < data_length and self.running:
                    chunk_size = min(4096, data_length - received)
                    chunk = client_socket.recv(chunk_size)
                    if not chunk:
                        break
                    received += len(chunk)
                
                if received == data_length:
                    frame_count += 1
                    print(f"Frame {frame_count}: Received {data_length} bytes from Rhino")
                else:
                    print(f"Incomplete frame: {received}/{data_length} bytes")
                    break
                    
            except Exception as e:
                print(f"Client handling error: {e}")
                break
        
        print(f"Client disconnected after {frame_count} frames")


def demo_tcp_integration():
    """Demonstrate TCP integration with Rhino3D."""
    print("=" * 60)
    print("RHINO3D TCP STREAMING INTEGRATION DEMO")
    print("=" * 60)
    
    # Start demo server
    server = DemoTCPServer(host='127.0.0.1', port=8081)  # Use different port to avoid conflicts
    if not server.start():
        return
    
    print("TCP server started. Now:")
    print("1. Open Rhino3D with your robot model")
    print("2. Open Grasshopper")
    print("3. Add C# Script component with rhino_viewport_stream.cs code")
    print("4. Set server_host to '127.0.0.1' and server_port to 8080")
    print("5. Enable streaming")
    print("6. Watch this console for incoming frames")
    print()
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping demo...")
        server.stop()


def create_grasshopper_setup_guide():
    """Create setup guide for Grasshopper integration."""
    guide = """
# RHINO3D GRASSHOPPER TCP STREAMING SETUP

## Step 1: Create C# Script Component
1. In Grasshopper, add a C# Script component
2. Set inputs:
   - `enabled` (Boolean) - connect a Boolean Toggle
   - `server_host` (String) - set to "127.0.0.1"
   - `server_port` (Integer) - set to 8080
   - `fps` (Number) - set to 20.0
   - `viewport_name` (String) - optional, leave empty for active viewport
   - `quality` (Integer) - set to 85 (JPEG quality)

## Step 2: Copy C# Code
Copy the content from: `rhino_integration/rhino_viewport_stream.cs`
Paste into the C# Script component

## Step 3: Test Connection
1. Run the Python demo: `python3 demo_rhino_integration.py`
2. Enable the Grasshopper toggle
3. Watch for "Rhino connection" messages in Python console

## Step 4: Integration with Vision System
Update your system config to use TCP streaming:
```json
{
  "camera": {
    "source_type": "tcp_stream",
    "fps": 20,
    "host": "127.0.0.1",
    "port": 8080
  }
}
```

Then run: `python3 raspberry-pi/tracking/correction/vision_correction_system.py system_config_rhino.json`

## Performance Notes
- TCP streaming provides best performance (~20-30 FPS)
- Adjust FPS and JPEG quality based on your needs
- Higher quality = larger data, lower FPS

## Troubleshooting
- Connection refused: Start Python server before Grasshopper
- Poor performance: Reduce FPS or JPEG quality
- Detection issues: Check coordinate system alignment
"""
    
    with open('rhino_tcp_setup.md', 'w') as f:
        f.write(guide)
    
    print("Created rhino_tcp_setup.md with detailed setup instructions")


def main():
    print("KUKA Vision System - Rhino3D TCP Streaming Integration")
    print("This demo shows TCP streaming from Rhino3D to the vision system.")
    print()
    
    # Create setup guide
    create_grasshopper_setup_guide()
    
    # Run demo
    try:
        demo_tcp_integration()
    except Exception as e:
        print(f"Demo error: {e}")
    
    print("\nDemo completed!")
    print("Next steps:")
    print("1. Follow rhino_tcp_setup.md for complete setup")
    print("2. Test with: python3 raspberry-pi/tracking/correction/vision_correction_system.py system_config_rhino.json")


if __name__ == "__main__":
    main()
