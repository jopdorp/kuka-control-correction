# Multiple ChArUco Board Support

This document describes the new functionality for supporting multiple ChArUco boards simultaneously in the KUKA vision correction system.

## Overview

The system now supports detecting and tracking multiple ChArUco boards at the same time, using a least-error algorithm to match detected boards with expected board configurations based on the current camera pose.

## Key Features

- **Multiple Board Detection**: Detect several different ChArUco boards in a single image
- **Intelligent Matching**: Use Hungarian algorithm to match detected boards with expected configurations
- **Flexible Configuration**: JSON-based configuration for board specifications and expected positions
- **Backwards Compatibility**: Existing ArUco marker functionality is preserved

## Configuration

### Enabling ChArUco Board Mode

In your `SystemConfig`, set:

```python
config = SystemConfig()
config.use_charuco_boards = True
config.charuco_boards_config_file = "charuco_boards_config.json"
```

### Board Configuration File

Create a JSON file defining your boards:

```json
{
  "boards": [
    {
      "board_id": "board_1",
      "squares_x": 5,
      "squares_y": 4,
      "square_size": 0.05,
      "marker_size": 0.035,
      "dictionary_type": 10,
      "expected_plane": [100.0, 200.0, 0.0, 0.0, 0.0, 0.0]
    },
    {
      "board_id": "board_2",
      "squares_x": 4,
      "squares_y": 3,
      "square_size": 0.04,
      "marker_size": 0.028,
      "dictionary_type": 10,
      "expected_plane": [300.0, 150.0, 50.0, 0.0, 0.0, 90.0]
    }
  ]
}
```

### Configuration Parameters

- **board_id**: Unique identifier for the board
- **squares_x, squares_y**: Number of squares in each direction
- **square_size**: Physical size of each square in meters
- **marker_size**: Physical size of ArUco markers in meters
- **dictionary_type**: ArUco dictionary type (e.g., cv2.aruco.DICT_6X6_250)
- **expected_plane**: Expected board position and orientation [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg]

## System Configuration Options

Additional configuration parameters for board matching:

```python
config.max_board_position_error = 50.0  # Maximum position error in mm
config.max_board_rotation_error = 30.0  # Maximum rotation error in degrees
config.board_position_weight = 1.0      # Weight for position errors in matching
config.board_rotation_weight = 0.1      # Weight for rotation errors in matching
```

## How It Works

### 1. Detection Phase

The system detects all visible ChArUco boards in the camera image using OpenCV's ChArUco detection algorithms.

### 2. Expected Pose Calculation

For each configured board, the system calculates where that board should appear in the camera image based on:
- Current robot TCP position (from controller)
- Known tool-to-camera transformation
- Board's expected position in base frame

### 3. Matching Algorithm

The system uses the Hungarian algorithm to optimally match detected boards with expected configurations:

- Calculates position and rotation errors between detected and expected poses
- Builds a cost matrix weighing position and rotation errors
- Finds the optimal assignment minimizing total error
- Rejects matches with errors exceeding configured thresholds

### 4. Correction Calculation

Using the best matched board, the system calculates position corrections by comparing:
- Actual detected board pose in camera frame
- Expected board pose calculated from current robot position

## Testing

### Generate Test Boards

Run the test script to generate ChArUco board images:

```bash
python test_charuco_boards.py
```

This will:
- Generate test board images for each configured board
- Test the detection and matching algorithms
- Verify system initialization

### Unit Tests

Run the test suite:

```bash
python -m pytest tests/test_charuco_boards.py -v
```

## Usage Examples

### Basic Setup

```python
from vision_correction_system import VisionCorrectionSystem, SystemConfig

# Configure for ChArUco boards
config = SystemConfig()
config.use_charuco_boards = True
config.charuco_boards_config_file = "my_boards.json"

# Initialize system
system = VisionCorrectionSystem(config)

# Load calibration and start
system.load_configuration_files(
    "camera_calibration.npz",
    "marker_positions.json",  # Still needed for ArUco fallback
    "robot_config.json"
)
system.start_system()
```

### Direct Detection

```python
from charuco_board_detector import CharucoBoardDetector, CharucoBoardConfig
from charuco_board_matcher import CharucoBoardMatcher

# Create board configurations
configs = [
    CharucoBoardConfig(
        board_id="test_board",
        squares_x=5, squares_y=4,
        square_size=0.05, marker_size=0.035,
        dictionary_type=cv2.aruco.DICT_6X6_250,
        expected_plane=[100.0, 200.0, 0.0, 0.0, 0.0, 0.0]
    )
]

# Initialize detector and matcher
detector = CharucoBoardDetector(configs, camera_matrix, distortion_coeffs)
matcher = CharucoBoardMatcher(configs)

# Process frame
detected_boards = detector.detect_boards(image)
camera_pose = [x, y, z, a, b, c]  # Current camera pose
matches = matcher.match_boards(detected_boards, camera_pose)
```

## Advantages Over Individual Markers

1. **Higher Accuracy**: ChArUco boards provide more stable pose estimation
2. **Better Occlusion Handling**: Partial board visibility still allows pose estimation
3. **Reduced Ambiguity**: Board matching eliminates uncertainty about marker identity
4. **Scalable**: Easy to add more boards without ID conflicts

## Performance Considerations

- **Processing Time**: Increases with number of board configurations
- **Memory Usage**: Each board type requires separate detector instance
- **Matching Complexity**: O(n³) Hungarian algorithm for n boards
- **Recommended**: Use 3-5 boards maximum for real-time performance

## Troubleshooting

### No Boards Detected
- Check camera calibration quality
- Verify board visibility and lighting
- Ensure board configurations match physical boards
- Lower `min_corners_for_pose` if needed

### Poor Matching
- Increase `max_board_position_error` and `max_board_rotation_error`
- Check expected_plane coordinates in configuration
- Verify tool-to-camera transformation accuracy
- Adjust matching weights if position/rotation priorities differ

### System Fallback
If ChArUco board detection fails, the system automatically falls back to traditional ArUco marker mode.

## Integration with KUKA System

The ChArUco board system integrates seamlessly with the existing KUKA correction pipeline:

1. Robot reports current TCP position
2. System calculates expected board poses
3. Camera detects and matches boards
4. Correction calculated from best match
5. Correction sent to robot controller

The correction calculation and communication protocols remain unchanged, ensuring compatibility with existing robot programs.