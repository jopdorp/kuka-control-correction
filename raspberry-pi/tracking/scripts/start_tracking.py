#!/usr/bin/env python3
"""
KUKA Vision Correction System Startup Script

This script starts the vision correction system using configuration files.
All configuration is loaded from JSON files, no hardcoded values.
"""

import sys
import os
import json
import logging
import time
import argparse

# Add correction to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'correction'))

from vision_correction_system import VisionCorrectionSystem, SystemConfig


def validate_config_files(config_file: str, system_config: SystemConfig) -> bool:
    """
    Validate that all required configuration files exist.
    
    Args:
        config_file: Path to system configuration file
        system_config: Loaded system configuration
        
    Returns:
        True if all files exist
    """
    config_dir = os.path.dirname(config_file)
    
    required_files = [
        system_config.charuco_boards_config_file,
        # Camera calibration and robot config are loaded by the system
    ]
    
    missing_files = []
    for file_path in required_files:
        # Handle relative paths relative to config file directory
        if not os.path.isabs(file_path):
            file_path = os.path.join(config_dir, file_path)
        
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required configuration files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="KUKA Vision Correction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration files:
  The system requires a JSON configuration file that specifies all system
  parameters and paths to other required files.

Examples:
  %(prog)s --config system_config.json
  %(prog)s --config system_config.json --verbose
  %(prog)s --help
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default=os.path.join(os.path.dirname(__file__), '..', 'config', 'system_config.json'),
        help='Path to system configuration file (default: config/system_config.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration files, do not start system'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("KUKA Vision Correction System")
    print("=" * 50)
    
    # Load system configuration
    print(f"Loading system configuration from: {args.config}")
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    
    try:
        config = SystemConfig.from_json(args.config)
        print("✓ System configuration loaded successfully")
        
        # Display key configuration
        print(f"  Camera: {config.camera_resolution[0]}x{config.camera_resolution[1]} @ {config.camera_fps}fps")
        print(f"  ChArUco config: {config.charuco_boards_config_file}")
        print(f"  Controller: {config.controller_ip}:{config.controller_port}")
        print(f"  Thresholds: pos={config.position_threshold}mm, rot={config.rotation_threshold}°")
        
    except Exception as e:
        print(f"Error loading system configuration: {e}")
        return 1
    
    # Validate configuration files
    print("\nValidating configuration files...")
    if not validate_config_files(args.config, config):
        print("✗ Configuration validation failed")
        return 1
    
    print("✓ Configuration validation passed")
    
    if args.validate_only:
        print("\nValidation complete (--validate-only specified)")
        return 0
    
    # Initialize vision system
    print("\nInitializing vision system...")
    system = VisionCorrectionSystem(config)
    
    # Load configuration files (camera calibration, robot config, etc.)
    print("Loading calibration and configuration files...")
    
    # Get file paths from config
    config_dir = os.path.dirname(args.config)
    camera_cal_file = os.path.join(config_dir, 'camera_calibration.npz')
    robot_config_file = os.path.join(config_dir, 'robot_config.json')
    
    if system.load_configuration_files(camera_cal_file, robot_config_file):
        print("✓ Configuration loaded successfully")
        if hasattr(system, 'charuco_board_configs') and system.charuco_board_configs:
            print(f"✓ ChArUco board system enabled with {len(system.charuco_board_configs)} boards")
    else:
        print("✗ Failed to load configuration - check file paths")
        print(f"  Camera calibration: {camera_cal_file}")
        print(f"  Robot config: {robot_config_file}")
        return 1
    
    # Start system
    print("\nStarting vision correction system...")
    if system.start_system():
        print("✓ System started successfully")
        
        try:
            print("\n" + "=" * 50)
            print("System running. Monitoring for corrections...")
            print("Press Ctrl+C to stop.\n")
            
            # Main monitoring loop
            last_status_time = 0
            status_interval = 5.0  # Print status every 5 seconds
            
            while True:
                time.sleep(0.1)
                
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
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
                                    for i, board in enumerate(detected_boards):
                                        print(
                                            f"  - Board_{i}: {board.num_corners} corners, "
                                            f"conf={board.confidence:.2f}"
                                        )
                    
                    print("-" * 30)
                    last_status_time = current_time
                
        except KeyboardInterrupt:
            print("\nShutting down system...")
            system.stop_system()
            print("✓ System stopped")
            return 0
    
    else:
        print("✗ Failed to start system")
        return 1


if __name__ == "__main__":
    sys.exit(main())
