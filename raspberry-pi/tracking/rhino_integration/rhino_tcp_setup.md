
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

Then run: `python3 raspberry-pi/src/vision_correction_system.py system_config_rhino.json`

## Performance Notes
- TCP streaming provides best performance (~20-30 FPS)
- Adjust FPS and JPEG quality based on your needs
- Higher quality = larger data, lower FPS

## Troubleshooting
- Connection refused: Start Python server before Grasshopper
- Poor performance: Reduce FPS or JPEG quality
- Detection issues: Check coordinate system alignment
