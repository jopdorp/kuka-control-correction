#!/usr/bin/env python3
"""
Example integration of multiple ChArUco boards with KUKA vision correction system.

This example shows how to configure and use the multi-board functionality
in a real robot application.
"""

import json
import logging
from vision_correction_system import VisionCorrectionSystem, SystemConfig

def create_production_config():
    """Create production configuration for multi-board system."""
    config = SystemConfig()
    
    # Enable ChArUco boards
    config.use_charuco_boards = True
    config.charuco_boards_config_file = "production_boards.json"
    
    # Camera settings for production
    config.camera_index = 0
    config.camera_resolution = (1920, 1080)
    config.camera_fps = 30
    
    # Board matching parameters (tuned for production environment)
    config.max_board_position_error = 30.0   # mm - tighter tolerance
    config.max_board_rotation_error = 15.0   # degrees - tighter tolerance
    config.board_position_weight = 1.0       # Position is critical
    config.board_rotation_weight = 0.2       # Rotation less critical
    
    # Vision correction settings
    config.position_threshold = 0.5          # mm
    config.rotation_threshold = 0.5          # degrees
    config.confidence_threshold = 0.7        # Require good confidence
    config.max_correction = 3.0              # mm - limit correction magnitude
    
    return config

def create_production_board_config():
    """Create production board configuration file."""
    boards_config = {
        "boards": [
            {
                "board_id": "station_1_board",
                "squares_x": 7,
                "squares_y": 5,
                "square_size": 0.04,      # 40mm squares
                "marker_size": 0.03,      # 30mm markers
                "dictionary_type": 10,    # DICT_6X6_250
                "expected_plane": [450.0, 200.0, 10.0, 0.0, 0.0, 0.0]  # Station 1 position
            },
            {
                "board_id": "station_2_board", 
                "squares_x": 6,
                "squares_y": 4,
                "square_size": 0.05,      # 50mm squares
                "marker_size": 0.035,     # 35mm markers
                "dictionary_type": 10,
                "expected_plane": [200.0, 450.0, 15.0, 0.0, 0.0, 90.0]  # Station 2 position
            },
            {
                "board_id": "quality_check_board",
                "squares_x": 5,
                "squares_y": 4,
                "square_size": 0.03,      # 30mm squares
                "marker_size": 0.021,     # 21mm markers  
                "dictionary_type": 10,
                "expected_plane": [350.0, 350.0, 20.0, 0.0, 30.0, 45.0]  # QC station
            }
        ]
    }
    
    with open("production_boards.json", 'w') as f:
        json.dump(boards_config, f, indent=2)
    
    return boards_config

def main():
    """Main production example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("KUKA Multi-Board Vision Correction System")
    print("=" * 50)
    
    # Create production configuration
    print("Creating production configuration...")
    board_config = create_production_board_config()
    config = create_production_config()
    
    print(f"✓ Configured {len(board_config['boards'])} boards:")
    for board in board_config['boards']:
        print(f"  - {board['board_id']}: {board['squares_x']}×{board['squares_y']} at {board['expected_plane'][:3]}")
    
    # Initialize vision system
    print("\nInitializing vision system...")
    system = VisionCorrectionSystem(config)
    
    # Load configuration files
    print("Loading calibration and configuration files...")
    if system.load_configuration_files(
        'camera_calibration.npz',     # Camera intrinsics
        'marker_positions.json',      # Fallback ArUco markers (can be empty)
        'robot_config.json'           # Robot kinematics and tool offset
    ):
        print("✓ Configuration loaded successfully")
        print(f"✓ ChArUco mode enabled with {len(system.charuco_board_configs)} boards")
    else:
        print("✗ Failed to load configuration - check file paths")
        return
    
    # Start system
    print("\nStarting vision correction system...")
    if system.start_system():
        print("✓ System started successfully")
        
        try:
            print("\n" + "=" * 50)
            print("System running. Monitoring for corrections...")
            print("Press Ctrl+C to stop.\n")
            
            # Main monitoring loop
            import time
            while True:
                time.sleep(1)
                
                # Get system status
                status = system.get_system_status()
                stats = status['statistics']
                
                # Print periodic status
                print(f"Status: Camera={status['camera_running']}, "
                      f"Comm={status['communication_running']}, "
                      f"Proc={status['processing_running']}")
                print(f"Stats: Frames={stats['frames_processed']}, "
                      f"Corrections={stats['corrections_sent']}, "
                      f"Seq={status['sequence_counter']}")
                
                # Show detected boards info
                if hasattr(system, 'charuco_detector') and system.charuco_detector:
                    with system.frame_lock:
                        if system.current_frame is not None:
                            frame = system.current_frame.copy()
                            detected_boards = system.charuco_detector.detect_boards(frame)
                            if detected_boards:
                                print(f"Detected: {len(detected_boards)} boards")
                                for board in detected_boards:
                                    if board.board_id:
                                        print(f"  - {board.board_id}: {board.num_corners} corners, "
                                              f"conf={board.confidence:.2f}")
                
                print("-" * 30)
                
        except KeyboardInterrupt:
            print("\nShutting down system...")
            system.stop_system()
            print("✓ System stopped")
    
    else:
        print("✗ Failed to start system")

if __name__ == "__main__":
    main()