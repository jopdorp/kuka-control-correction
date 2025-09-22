# KUKA Vision-Based Position Correction System

**ALWAYS follow these instructions first. Only fallback to additional search and context gathering if the information here is incomplete or found to be in error.**

## Working Effectively

### Bootstrap and Setup Commands
Execute these commands in order to set up the development environment:

```bash
# Install required system packages
sudo apt-get update
sudo apt-get install -y python3-opencv python3-scipy python3-pytest python3-flake8 black mypy

# Verify Python version (should be 3.12+)
python3 --version

# Verify OpenCV installation
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

**IMPORTANT**: The system uses OpenCV 4.6.0 (system package) which lacks some newer features. Camera calibration scripts may fail due to API changes - this is a known limitation.

### Testing
Run these commands to validate code changes:

```bash
# ALWAYS run these tests - they are fast and reliable
# Takes ~0.5 seconds total - NEVER CANCEL
python3 -m pytest tests/test_pose_utils.py tests/test_pose_utils_roundtrip.py -v

# Camera-related tests will SEGFAULT in headless environments - this is expected
# DO NOT attempt to run: pytest tests/test_aruco_detector_unit.py
# DO NOT attempt to run: pytest tests/test_vision_system_*.py
```

### Linting and Code Quality
Run these before committing any changes:

```bash
# Check code style - takes ~0.2 seconds - NEVER CANCEL
python3 -m flake8 raspberry-pi/src/ --max-line-length=100

# Check code formatting - takes ~0.7 seconds - NEVER CANCEL  
python3 -m black --check raspberry-pi/src/ --line-length=100

# Auto-format code (if needed)
python3 -m black raspberry-pi/src/ --line-length=100
```

**EXPECTED LINTING ISSUES**: The codebase currently has ~206 flake8 issues (mostly whitespace) and would reformat 4 files with black. This is normal.

### Running Applications

#### Correction Helper (Network Component)
```bash
cd raspberry-pi/src
python3 correction_helper.py --kuka-ip 127.0.0.1 --port 7001
# Starts TCP server listening on port 7001
# Connects to KUKAVARPROXY on target IP:7000
# Ctrl+C to stop
```

#### Vision System (Camera Required)
```bash
cd raspberry-pi/src
python3 vision_correction_system.py
# WILL SEGFAULT in headless environments - camera required
# Only run on systems with physical cameras attached
```

#### Camera Testing Scripts
```bash
# Basic ArUco marker detection test
./test_webcam_aruco.py
# WILL SEGFAULT in headless environments - camera required

# Camera calibration (has OpenCV version compatibility issues)
python3 simple_calibration.py
# MAY FAIL due to OpenCV 4.6.0 vs newer API requirements
```

## Validation Scenarios

### Manual Testing Requirements
After making code changes, ALWAYS test these scenarios:

1. **Pose Utilities (REQUIRED)**:
   ```bash
   python3 -m pytest tests/test_pose_utils*.py -v
   ```
   Must pass all 29 tests in <1 second.

2. **Network Components**:
   ```bash
   cd raspberry-pi/src
   timeout 5 python3 correction_helper.py --kuka-ip 127.0.0.1
   ```
   Should start and log "Vision correction helper listening on 0.0.0.0:7001" without errors.

3. **JSON Message Validation**:
   ```bash
   echo '{"translation_correction": [1.0, 2.0, 3.0], "rotation_correction": [0.1, 0.2, 0.3], "confidence": 0.95, "timestamp": 1234567890.0, "sequence_id": 1}' | python3 -c "import json, sys; print('Valid:', json.load(sys.stdin))"
   ```
   Should parse and print the JSON structure.

4. **Code Quality Checks**:
   ```bash
   python3 -m flake8 raspberry-pi/src/ --max-line-length=100
   python3 -m black --check raspberry-pi/src/ --line-length=100
   ```
   Document any new linting issues introduced by your changes.

### Hardware-Dependent Testing (Cannot Test in Headless Environment)
These scenarios require physical hardware and will fail in CI/sandboxed environments:

- Camera detection and ArUco marker processing
- Full vision correction system execution
- Camera calibration workflows
- KUKA robot controller integration

## Architecture and Navigation

### Key Projects in Codebase
1. **raspberry-pi/src/** - Main Python vision processing system
   - `aruco_detector.py` - ArUco marker detection and pose estimation
   - `pose_utils.py` - SE3 utilities and coordinate transformations (CORE - always test this)
   - `vision_correction_system.py` - Main coordination system
   - `correction_helper.py` - TCP bridge to KUKA controller

2. **kuka-controller/** - KUKA KRL integration files
   - `config_additions.dat` - KRL global variables to add to $CONFIG.DAT
   - `sps_additions.src` - Continuous correction code for SPS.SUB
   - `VisionRun.src/dat` - Sample program for testing
   - `README.md` - Detailed controller setup instructions

3. **tests/** - Unit test suite
   - Only `test_pose_utils*.py` work reliably in headless environments
   - Other tests require camera hardware

### Critical Dependencies
- **Python 3.12+** with system OpenCV 4.6.0
- **Network connectivity** for KUKA controller communication (port 7000/7001)
- **Physical camera** for vision processing (USB3 recommended)
- **ArUco markers** placed at known positions in robot workspace

### Common Debugging Steps
1. **Import errors**: Check if system packages are installed (opencv, scipy, pytest)
2. **Camera issues**: Verify physical camera connection and permissions
3. **Network issues**: Check IP addresses and firewall settings for ports 7000/7001
4. **Coordinate transform errors**: Always test pose_utils functions with unit tests
5. **KUKA connection**: Verify KUKAVARPROXY is running on controller PC port 7000

### Known Limitations and Workarounds
- **Headless environments**: Camera-related code will segfault - this is expected
- **OpenCV version**: Some calibration features require newer OpenCV than system provides
- **Network timeouts**: KUKAVARPROXY connections may fail if controller not reachable
- **Linting issues**: Codebase has many whitespace issues - focus on functional changes

### File Locations and Frequent Tasks
- **Always check `pose_utils.py`** after making coordinate transformation changes
- **Configuration files** in `kuka-controller/` need manual installation on KUKA PC
- **Test outputs** go to `tests/` - only pose_utils tests are reliable
- **Camera calibration files** (.npz format) go in repository root
- **Log files** and debug output appear in console - no persistent logging configured

## Build Information
- **No build step required** - Python interpreted language
- **No compilation** needed
- **Dependency installation** via system packages only (pip packages cause compatibility issues)
- **No CI/CD configured** - manual testing required

## Important Notes
- **NEVER CANCEL** any command - all operations complete in <1 second except network timeouts
- **Camera access required** for vision components - will fail in sandboxed environments
- **Always test pose utilities** - they are the mathematical foundation of the system
- **Linting is advisory** - focus on functional correctness over style
- **Network components can be tested** without physical robot using localhost