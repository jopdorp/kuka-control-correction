# KUKA Vision-Based Position Correction System

This project implements a real-time vision correction system for KUKA industrial robots using ArUco markers. The system automatically corrects robot positioning errors by tracking the robot's actual position relative to known reference markers and continuously adjusting the robot's coordinate system in real-time.

## System Overview

The system consists of two main components:

1. **Raspberry Pi Vision System** - Mounted on robot tool, tracks ArUco markers
2. **KUKA Controller Integration** - Receives corrections via TCP and adjusts robot coordinate system

```
┌─────────────────┐    TCP/Ethernet    ┌─────────────────┐
│   Raspberry Pi  │◄──────────────────►│ KUKA Controller │
│   + Camera      │                    │   Windows PC    │
│   (Tool Frame)  │                    │   + KRL System  │
└─────────────────┘                    └─────────────────┘
        │                                       │
        ▼                                       ▼
┌─────────────────┐                    ┌─────────────────┐
│ ArUco Detection │                    │ KUKAVARPROXY    │
│ Pose Estimation │                    │ + SPS Filtering │
│ Error Calculation│                    │ + Base Update   │
└─────────────────┘                    └─────────────────┘
```

## Key Features

- **Continuous position correction** at device speed (~12ms SPS rate) with low-pass filtering
- **ArUco marker-based** visual tracking system  
- **Direct TCP communication** eliminating intermediate hardware
- **VKRC2 compatible** KRL implementation using SPS background tasks
- **Configurable safety limits** and filter tuning
- **Multi-marker support** for improved accuracy and redundancy
- **Non-intrusive integration** with existing KUKA programs

## Directory Structure

```
├── README.md
├── docs/                      # Documentation
│   └── architecture.md
├── raspberry-pi/              # Vision processing system
│   └── src/
│       ├── aruco_detector.py          # ArUco marker detection
│       ├── pose_utils.py              # SE3 utilities (no kinematics)
│       └── vision_correction_system.py # Main coordination system
└── kuka-controller/           # KUKA KRL programs and integration
    ├── config_additions.dat          # KRL global variables
    ├── sps_additions.src             # SPS continuous correction
    ├── VisionRun.src                 # Sample program
    ├── VisionRun.dat                 # Sample points
    ├── correction_helper.py          # Windows TCP helper
    └── README.md                     # Controller setup guide
```

## Quick Start

### 1. Hardware Setup

1. **Mount camera** on robot tool (recommend USB3 camera for low latency)
2. **Place ArUco markers** in robot workspace at known positions
3. **Connect Raspberry Pi** to network with access to KUKA controller Windows PC

### 2. Raspberry Pi Setup

```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip python3-opencv
pip3 install numpy opencv-contrib-python

# Clone and configure
git clone <this-repo>
cd kuka-control-correction/raspberry-pi

# Edit configuration for your setup
nano src/vision_correction_system.py  # Set controller IP, camera params, marker positions

# Test camera and detection
python3 src/aruco_detector.py

# Run vision correction system
python3 src/vision_correction_system.py
```

### 3. KUKA Controller Setup

1. **Install KUKAVARPROXY** on Windows PC (if not present)
2. **Add KRL variables** from `kuka-controller/config_additions.dat` to `$CONFIG.DAT`
3. **Modify SPS.SUB** with code from `kuka-controller/sps_additions.src`
4. **Install sample program** files to test integration
5. **Run Windows helper**: `python correction_helper.py`

See detailed setup instructions in `kuka-controller/README.md`

## Configuration

### Camera Calibration

Generate camera intrinsics using OpenCV:

```python
# Use standard checkerboard calibration
camera_matrix = [[fx, 0, cx],
                 [0, fy, cy], 
                 [0, 0, 1]]
distortion_coeffs = [k1, k2, p1, p2, k3]
```

### ArUco Marker Setup

```python
# Define marker positions in robot base frame
marker_positions = {
    23: {  # Marker ID
        'position': [100.0, 200.0, 0.0],      # X,Y,Z in mm
        'orientation': [0.0, 0.0, 0.0]        # A,B,C in degrees
    }
}
```

### System Tuning

- **Filter response**: Adjust `G_ALPHA` in KRL (0.05-0.5 range)
- **Safety limits**: Set `G_MAX_MM` and `G_MAX_DEG` appropriately
- **Update rate**: Balance vision processing vs network load

## Architecture Details

### Vision Processing Pipeline

1. **Frame capture** from tool-mounted camera
2. **ArUco detection** using OpenCV with sub-pixel refinement
3. **Pose estimation** using solvePnP with known marker geometry
4. **Error calculation** relative to expected tool position in base frame
5. **Coordinate transform** to robot ABC rotation format
6. **TCP transmission** to KUKA controller with JSON protocol

### KUKA Integration

1. **Windows helper** receives TCP corrections and writes to KUKAVARPROXY  
2. **SPS background task** applies low-pass filtering at ~12ms rate
3. **Base coordinate adjustment** modifies `$BASE` for automatic correction
4. **Reference frame capture** allows per-program coordinate system setup

## Supported Controllers

- **VKRC2** (tested) - Uses KUKAVARPROXY and SPS.SUB modifications
- **VKRC4** (adaptable) - May require updated communication protocols
- **Other KRL systems** - Core approach applicable with minor modifications

## Theory of Operation

### Coordinate Systems

The system operates with three key coordinate frames:

- **Base Frame** (`$BASE`) - Robot's working coordinate system
- **Tool Frame** (`$TOOL`) - Camera mounting position on robot tool
- **Marker Frame** - ArUco marker positions in world coordinates

### Position Correction

Position errors are calculated as:

```
Error = (Marker_Expected - Marker_Detected) 
Correction = Error × Calibration_Transform
$BASE_new = $BASE_old + Filtered(Correction)
```

### Real-time Processing

- **Vision processing**: 20-50 FPS depending on image size and markers
- **Correction filtering**: 80 Hz (12ms SPS cycle) with configurable low-pass filter
- **Network latency**: 1-5ms on local Ethernet network
- **Overall latency**: 50-100ms from detection to robot correction

## Troubleshooting

### Common Issues

1. **No marker detection**:
   - Check lighting conditions and marker quality
   - Verify camera focus and exposure settings
   - Ensure markers are within camera field of view

2. **Large correction errors**:
   - Verify marker positions are accurate in configuration
   - Check camera calibration parameters
   - Ensure tool-to-camera transform is correct

3. **Unstable motion**:
   - Reduce filter coefficient `G_ALPHA`
   - Increase safety limits `G_MAX_MM` and `G_MAX_DEG`
   - Check for mechanical vibrations affecting camera

4. **Network connection failures**:
   - Verify IP addresses and port configurations
   - Check firewall settings on Windows PC
   - Test network connectivity with ping

## Performance Optimization

### Vision Processing

- Use **smaller image regions** around expected marker locations
- **Reduce camera resolution** if processing speed is limiting
- **Multi-threading** for parallel marker detection
- **GPU acceleration** with OpenCV CUDA support

### Network Communication

- **Compress correction data** for slower networks
- **Batch multiple corrections** to reduce packet overhead
- **Use UDP** for faster transmission (with reliability handling)

### Robot Control

- **Tune $ADVANCE** settings for smoother motion with base corrections
- **Reduce correction rate** during critical motion segments
- **Coordinate multiple robots** sharing the same marker references

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with actual hardware setup
4. Submit pull request with detailed description

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- KUKA for KRL programming documentation and KUKAVARPROXY
- OpenCV community for ArUco marker detection libraries
- Industrial automation community for testing and feedback
