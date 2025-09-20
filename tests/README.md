# Tests for KUKA Vision Correction System

This directory contains unit tests and integration tests for the vision correction system.

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest

# Run tests with coverage
pytest --cov=raspberry-pi/src --cov-report=html

# Run specific test file
pytest tests/test_aruco_detector.py

# Run tests with verbose output
pytest -v
```

## Test Structure

- `test_aruco_detector.py` - Tests for ArUco marker detection
- `test_pose_utils.py` - Tests for pose utilities and coordinate transforms
- `test_vision_system.py` - Tests for the main vision correction system
- `test_integration.py` - Integration tests for the complete system
- `conftest.py` - Shared test fixtures and configuration

## Test Data

The `test_data/` directory contains sample images and calibration data for testing:
- Sample ArUco marker images
- Camera calibration parameters
- Known marker positions for validation

## Mocking

Tests use mocking for:
- Camera hardware (no physical camera required)
- Network connections (TCP sockets)
- File I/O operations
- OpenCV operations where appropriate
