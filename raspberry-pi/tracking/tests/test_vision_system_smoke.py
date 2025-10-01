"""
Lightweight smoke tests for VisionCorrectionSystem that avoid starting threads.
Tests basic functionality of the ChArUco-based vision correction system.
"""
import os
import sys
import time
import numpy as np

# Ensure correction is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'correction'))

from vision_correction_system import VisionCorrectionSystem, SystemConfig, CorrectionData


def test_system_construction():
    """Test basic system construction with ChArUco configuration."""
    cfg = SystemConfig(charuco_boards_config_file="test_boards.json")  # Required field
    
    system = VisionCorrectionSystem(cfg)
    assert system.config is cfg
    assert system.camera_source is None
    assert system.tcp_socket is None
    assert system.processing_running is False
    assert system.communication_running is False
    
    # ChArUco-specific components should be initialized to None
    assert system.charuco_detector is None
    assert system.charuco_matcher is None
    assert system.charuco_board_configs == []
    
    status = system.get_system_status()
    assert 'statistics' in status


def test_send_correction_enqueues_message():
    """Test that correction data is properly enqueued for transmission."""
    cfg = SystemConfig(charuco_boards_config_file="test_boards.json")  # Required field
    
    system = VisionCorrectionSystem(cfg)
    cd = CorrectionData(
        translation_correction=np.array([1.0, -2.0, 3.0]),
        rotation_correction=np.array([0.1, -0.2, 0.3]),
        confidence=0.9,
        timestamp=time.time(),
        sequence_id=1,
    )
    # Should enqueue a message without needing a TCP socket
    system._send_correction(cd)
    msg = system.message_queue.get(timeout=1.0)
    assert msg['type'] == 'BASE_CORRECTION'
    assert msg['sequence_id'] == 1
    assert 'correction' in msg
    assert 'confidence' in msg
    assert msg['confidence'] == 0.9


def test_system_statistics_initialization():
    """Test that system statistics are properly initialized."""
    cfg = SystemConfig(charuco_boards_config_file="test_boards.json")
    
    system = VisionCorrectionSystem(cfg)
    status = system.get_system_status()
    
    assert 'statistics' in status
    stats = status['statistics']
    
    # Check that expected statistics fields exist
    expected_fields = ['frames_processed', 'corrections_sent', 'markers_detected']
    for field in expected_fields:
        assert field in stats
