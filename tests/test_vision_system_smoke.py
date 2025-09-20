"""
Lightweight smoke tests for VisionCorrectionSystem that avoid starting threads.
"""
import os
import sys
import time
import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'raspberry-pi', 'src'))

from vision_correction_system import VisionCorrectionSystem, SystemConfig, CorrectionData


def test_system_construction():
    cfg = SystemConfig()
    system = VisionCorrectionSystem(cfg)
    assert system.config is cfg
    assert system.camera is None
    assert system.tcp_socket is None
    assert system.processing_running is False
    assert system.communication_running is False
    status = system.get_system_status()
    assert 'statistics' in status


def test_send_correction_enqueues_message():
    cfg = SystemConfig()
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
