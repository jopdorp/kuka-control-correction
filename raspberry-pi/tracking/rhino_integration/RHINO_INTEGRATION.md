# Rhino3D TCP Viewport Streaming Integration for KUKA Vision System

This integration allows you to replace your physical camera with a TCP video stream from Rhino3D's viewport camera, enabling integration testing and virtual robot programming validation.

## Overview

Instead of using a physical camera (webcam/ArUCam), the vision correction system can now receive images via **TCP streaming from Rhino3D viewport**.

Benefits:
- **High Performance**: 20-30 FPS streaming
- **Low Latency**: Direct TCP connection
- **Integration Testing**: Test vision corrections virtually
- **KUKAPRC Compatible**: Works with your Grasshopper workflow

## Quick Start

### 1. Run the Demo

```bash
cd /home/jopdorp/development/kuka-control-correction
python3 demo_rhino_integration.py
```

This will:
- Start a TCP server on port 8080
- Show connection status when Rhino connects
- Generate setup guide (`rhino_tcp_setup.md`)

### 2. Set up Grasshopper

1. Open Rhino3D with your robot model
2. Open Grasshopper
3. Add a C# Script component
4. Set inputs:
   - `enabled` (Boolean) - connect a toggle
   - `server_host` (String) - set to `"127.0.0.1"`
   - `server_port` (Integer) - set to `8080`
   - `fps` (Number) - set to `20.0`
   - `quality` (Integer) - set to `85` (JPEG quality)
5. Copy code from `rhino_integration/rhino_viewport_stream.cs`
6. Enable the toggle

### 3. Test Integration

Run your vision correction system with Rhino config:

```bash
python3 raspberry-pi/src/vision_correction_system.py system_config_rhino.json
```

## TCP Streaming Method

- **Performance**: 20-30 FPS  
- **Complexity**: Medium
- **Script**: `rhino_viewport_stream.cs`
- **Config**: `system_config_rhino.json`

## Configuration

The system configuration is simplified to just physical or TCP streaming:

```json
{
  "camera": {
    "source_type": "tcp_stream",
    "resolution": [1920, 1080],
    "fps": 20,
    "host": "127.0.0.1",
    "port": 8080
  }
}
```

## Camera Source Architecture

The system uses a clean camera abstraction:

```python
from camera_sources import create_camera_source, CameraConfig

# Create TCP camera source
config = CameraConfig(
    resolution=(1920, 1080),
    fps=20,
    host="127.0.0.1",
    port=8080
)

camera = create_camera_source('tcp_stream', config)
camera.start()

# Get frames
frame = camera.get_frame()
```

Available sources:
- `'physical'` - Physical camera (OpenCV VideoCapture)
- `'tcp_stream'` - TCP streaming from Rhino3D

## Coordinate System Alignment

**Critical**: Ensure your Rhino model coordinate system matches your physical robot setup:

### 1. Units
- Use **millimeters** in Rhino (KUKA standard)

### 2. Coordinate System
- **X-axis**: Same orientation as physical setup
- **Y-axis**: Same orientation as physical setup  
- **Z-axis**: Same orientation as physical setup
- **Origin**: Place at robot base coordinate system

### 3. Camera Setup
- Position Rhino viewport camera to match physical camera mount
- Match field of view to physical camera
- Use perspective view (not parallel projection)

### 4. CharUco Board Placement
- Place boards in Rhino model at same locations as physical boards
- Ensure same size and orientation
- Verify board coordinate systems match

## Camera Calibration

You'll need to calibrate the "virtual camera" (Rhino viewport):

### Option 1: Use Physical Camera Matrix
If your Rhino viewport closely matches the physical camera, use the existing calibration:

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coeffs": [k1, k2, p1, p2, k3]
}
```

### Option 2: Generate New Calibration  
Capture calibration images from Rhino viewport and run standard calibration.

## Testing and Validation

### 1. Connection Test
```bash
python3 demo_rhino_integration.py
```

### 2. Geometric Validation
- Place known geometry in both Rhino and physical setup
- Verify measurements match
- Check coordinate transformations

### 3. CharUco Board Detection
- Ensure boards are detected correctly in Rhino images
- Verify pose estimation accuracy
- Compare with physical board poses

### 4. Integration Testing
- Run full vision correction pipeline with Rhino feed
- Verify correction commands are reasonable
- Test edge cases (board occlusion, lighting, etc.)

## Troubleshooting

### Common Issues

**Connection refused**
- Start Python server before Grasshopper script
- Check host/port settings match
- Verify firewall allows connection

**Poor performance**
- Reduce FPS setting
- Lower JPEG quality
- Check network bandwidth

**Poor detection accuracy**
- Verify coordinate system alignment
- Check camera calibration matrix
- Ensure adequate lighting in Rhino viewport
- Verify CharUco board size and placement

### Debug Mode

Enable debug logging:
```bash
python3 -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python3 raspberry-pi/src/vision_correction_system.py system_config_rhino.json
```

## Integration with KUKAPRC

Since your robot programs come from Grasshopper/KUKAPRC:

1. **Use same Grasshopper definition** for both robot programming and vision streaming
2. **Sync robot positions** between KUKAPRC output and vision system
3. **Coordinate timing** between robot moves and vision corrections
4. **Validate toolpath** using vision feedback before physical execution

## Performance Optimization

### Rhino Viewport Optimization
- Use appropriate display mode (shaded vs rendered)
- Disable unnecessary visual effects
- Optimize model complexity in viewport
- Use GPU acceleration if available

### Streaming Optimization
- Match FPS to vision system requirements (typically 10-30 FPS)
- Use appropriate JPEG quality (70-90)
- Monitor CPU usage during streaming
- Test network bandwidth if using remote connections

## Files Created

- `raspberry-pi/src/camera_sources.py` - Camera abstraction classes
- `rhino_integration/rhino_viewport_stream.cs` - Grasshopper C# script
- `system_config_rhino.json` - Configuration for Rhino streaming
- `demo_rhino_integration.py` - TCP server demo
- `rhino_tcp_setup.md` - Detailed setup instructions

## Example Workflow

1. **Design robot path in Grasshopper/KUKAPRC**
2. **Place CharUco boards in Rhino model at physical positions**
3. **Start vision system**: `python3 raspberry-pi/src/vision_correction_system.py system_config_rhino.json`
4. **Start Rhino streaming**: Enable Grasshopper toggle
5. **Test vision corrections**: Verify detection and corrections work
6. **Deploy to physical robot**: Transfer validated program

This integration enables powerful virtual testing and validation of your KUKA vision correction system using your existing Rhino3D/Grasshopper workflow!
