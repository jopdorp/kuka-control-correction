# Camera Calibration Guide

This guide explains how to calibrate your camera for accurate ArUco marker detection and pose estimation.

## Why Camera Calibration is Important

Camera calibration determines:
- **Focal length** (fx, fy): How the camera magnifies objects
- **Principal point** (cx, cy): The camera's optical center
- **Distortion coefficients**: Correction for lens distortion (barrel/pincushion)

Without proper calibration, pose estimation will be inaccurate, especially for:
- Distance measurements
- 3D position calculations
- Angle estimations

## Quick Start

### 1. Generate Calibration Board
```bash
python camera_calibration.py
```
This will:
- Generate `aruco_calibration_board.png`
- Print this file at **actual size** (not "fit to page")

### 2. Calibration Process
1. Run the calibration script
2. Hold the printed board in front of your camera
3. Move it to different positions and angles
4. Press 'c' to capture images when the board is detected (need 15-30 images)
5. Press 'k' to perform calibration
6. Results are saved to `camera_calibration.npz`

### 3. Use Calibration
Your calibration is automatically loaded by `test_webcam_aruco.py` if the file exists.

## Detailed Instructions

### Printing the Calibration Board

1. **Print at actual size**: Make sure your printer is set to 100% scale
2. **Use quality paper**: Matte paper reduces reflections
3. **Verify size**: Measure the printed squares - they should be exactly 5cm

### Capturing Calibration Images

For best results:

1. **Good lighting**: Even, diffuse lighting without shadows
2. **Various angles**: Capture from different viewpoints:
   - Straight on
   - Tilted left/right
   - Tilted up/down
   - Different distances
3. **Cover the field of view**: Make sure to capture images with the board in all areas of the camera's view
4. **Keep it flat**: The board should be as flat as possible
5. **Avoid motion blur**: Hold steady when capturing

### Quality Indicators

- **Good calibration**: Reprojection error < 1.0 pixels
- **Acceptable**: Reprojection error < 2.0 pixels  
- **Poor**: Reprojection error > 3.0 pixels (recalibrate)

## Alternative: Generate Test Markers

If you want to test with individual markers:

```bash
python generate_aruco_markers.py
```

This generates various test markers you can print individually.

## Troubleshooting

### "Board not detected"
- Check lighting (avoid shadows and reflections)
- Ensure the entire board is visible
- Try different angles
- Print quality might be poor

### "Calibration failed"
- Need more images (capture 20-30)
- Images too similar (capture from more varied angles)
- Poor image quality (check focus and lighting)

### "High reprojection error"
- Recapture with better lighting
- Ensure board was flat during capture
- Check print quality and size
- Capture more images from varied angles

## Using Calibration Results

The calibration creates two files:
- `camera_calibration.npz`: Binary data for loading into code
- `camera_calibration_info.json`: Human-readable calibration info

Example usage in your code:
```python
import numpy as np

# Load calibration
calib_data = np.load('camera_calibration.npz')
camera_matrix = calib_data['camera_matrix']
distortion_coeffs = calib_data['distortion_coeffs']

# Use with ArUco detector
detector = ArucoDetector()
detector.camera_matrix = camera_matrix
detector.distortion_coeffs = distortion_coeffs
```

## Expected Calibration Values

Typical values for a webcam:
- **Focal length**: 800-2000 pixels (depends on resolution and lens)
- **Principal point**: Near image center
- **Distortion**: Small values (-0.5 to 0.5 for k1, k2, k3)

If your values are very different, double-check:
- Print size (markers should be exactly 5cm squares)
- Image resolution settings
- Capture quality
