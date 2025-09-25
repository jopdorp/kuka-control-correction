"""
Rhino3D TCP Viewport Streaming for KUKA Vision System

This directory contains the Grasshopper C# script for streaming Rhino viewport 
images to the KUKA vision correction system using TCP connection.

## Setup Instructions:

### 1. Grasshopper C# Script Setup

Use the `rhino_viewport_stream.cs` script in a C# component in Grasshopper.

Configuration:
- Set server host/port to match Python system
- Set capture rate (FPS)
- Enable/disable streaming with boolean input
- Adjust JPEG quality as needed

Python side configuration:
```python
camera_config = CameraConfig(
    host="127.0.0.1",
    port=8080,
    fps=20
)
camera_source = create_camera_source('tcp_stream', camera_config)
```

### 2. Usage Workflow

1. Start Python vision system first (starts TCP server)
2. Start Grasshopper with Rhino viewport streaming script  
3. Enable streaming toggle
4. Monitor connection and frame reception

## Performance

TCP streaming provides optimal performance:
- **Frame Rate**: 20-30 FPS
- **Latency**: Low
- **Reliability**: High
- **Quality**: Adjustable JPEG compression

## Coordinate System Alignment

Ensure your Rhino model coordinate system matches your physical robot setup:
- X, Y, Z axes orientation  
- Units (mm for KUKA)
- Origin placement
- Camera position and orientation relative to robot base

## Configuration

Update your system config JSON:
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

## Troubleshooting

- **Connection refused**: Start Python server before Grasshopper
- **Poor performance**: Reduce FPS or JPEG quality  
- **Detection issues**: Check coordinate system alignment
- **Network errors**: Check firewall settings
"""
