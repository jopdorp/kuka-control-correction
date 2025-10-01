#!/usr/bin/env python3
"""
KUKA File Upload Web Service

A Flask-based web service that allows drag-and-drop file uploads to the KUKA controller
via SMB/CIFS connection using smbclient.

Features:
- Web interface with drag-and-drop file upload
- File validation and security checks
- Direct transfer to KUKA controller via SMB
- Upload progress and status feedback
- Support for multiple file types (KRL, DAT, etc.)
- Logging and error handling

Usage:
    python3 kuka_file_upload_service.py --kuka-ip 192.168.1.50 --port 8000

Requirements:
    - smbclient (system package)
    - Flask
    - Werkzeug
"""

import os
import sys
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class KukaFileUploadService:
    """Web service for uploading files to KUKA controller."""
    
    # Allowed file extensions for KUKA programs
    ALLOWED_EXTENSIONS = {
        'src', 'dat', 'sub', 'krl', 'xml', 'txt', 'cfg', 'ini'
    }

    # Maximum file size (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    def __init__(self, kuka_ip: str = "192.168.1.50", web_port: int = 8000):
        self.kuka_ip = kuka_ip
        self.web_port = web_port
        self.logger = logging.getLogger(__name__)
        
        # Flask app configuration
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = self.MAX_FILE_SIZE
        self.app.secret_key = 'kuka-file-upload-service-2024'
        
        # Upload statistics
        self.stats = {
            'uploads_attempted': 0,
            'uploads_successful': 0,
            'uploads_failed': 0,
            'total_bytes_transferred': 0,
            'start_time': time.time()
        }
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main upload page."""
            return render_template('index.html', kuka_ip=self.kuka_ip)
            
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file upload."""
            try:
                if 'file' not in request.files:
                    return jsonify({'success': False, 'error': 'No file selected'})
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'})
                
                if not self._allowed_file(file.filename):
                    return jsonify({
                        'success': False, 
                        'error': f'File type not allowed. Allowed: {", ".join(self.ALLOWED_EXTENSIONS)}'
                    })
                
                # Secure the filename
                filename = secure_filename(file.filename)
                if not filename:
                    return jsonify({'success': False, 'error': 'Invalid filename'})
                
                # Save to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
                    file.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                try:
                    # Upload to KUKA
                    self.stats['uploads_attempted'] += 1
                    result = self._upload_to_kuka(tmp_path, filename)
                    
                    if result['success']:
                        self.stats['uploads_successful'] += 1
                        self.stats['total_bytes_transferred'] += os.path.getsize(tmp_path)
                        self.logger.info(f"Successfully uploaded {filename} to KUKA controller")
                    else:
                        self.stats['uploads_failed'] += 1
                        self.logger.error(f"Failed to upload {filename}: {result.get('error', 'Unknown error')}")
                    
                    return jsonify(result)
                    
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                        
            except RequestEntityTooLarge:
                return jsonify({'success': False, 'error': 'File too large (max 10MB)'})
            except Exception as e:
                self.logger.error(f"Upload error: {e}")
                self.stats['uploads_failed'] += 1
                return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})
        
        @self.app.route('/status')
        def status():
            """Get service status and statistics."""
            uptime = time.time() - self.stats['start_time']
            return jsonify({
                'kuka_ip': self.kuka_ip,
                'uptime_seconds': uptime,
                'uptime_formatted': f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
                'stats': self.stats,
                'allowed_extensions': list(self.ALLOWED_EXTENSIONS),
                'max_file_size_mb': self.MAX_FILE_SIZE / (1024 * 1024)
            })
        
        @self.app.route('/test-connection')
        def test_connection():
            """Test connection to KUKA controller."""
            try:
                result = self._test_kuka_connection()
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS)
    
    def _upload_to_kuka(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Upload file to KUKA controller using smbclient.
        
        Args:
            file_path: Local path to file to upload
            filename: Name for file on KUKA controller
            
        Returns:
            Dict with success status and details
        """
        try:
            # Build smbclient command - use hostname (resolved via extra_hosts in docker-compose)
            smb_command = [
                'smbclient',
                f'//krcpc/PROGRAM',
                '-U', '%kuka',  # Anonymous login with username 'kuka'
                '--option=client min protocol=NT1',
                '--option=client max protocol=NT1',
                '--option=client lanman auth=yes',
                '--option=client ntlmv2 auth=no',
                '--option=client plaintext auth=yes',
                '-c', f'put "{file_path}" "{filename}"'
            ]
            
            # Execute smbclient command
            self.logger.info(f"Uploading {filename} to {self.kuka_ip}/PROGRAM")
            
            result = subprocess.run(
                smb_command,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': f'Successfully uploaded {filename}',
                    'filename': filename,
                    'size_bytes': os.path.getsize(file_path)
                }
            else:
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                return {
                    'success': False,
                    'error': f'SMB upload failed: {error_msg}',
                    'returncode': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Upload timeout - KUKA controller may be unreachable'
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': 'smbclient not found - please install samba-client package'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Upload error: {str(e)}'
            }
    
    def _test_kuka_connection(self) -> Dict[str, Any]:
        """Test connection to KUKA controller SMB share."""
        try:
            # Test with simple directory listing - use hostname (resolved via extra_hosts in docker-compose)
            smb_command = [
                'smbclient',
                f'//krcpc/PROGRAM',
                '-U', '%kuka',
                '--option=client min protocol=NT1',
                '--option=client max protocol=NT1', 
                '--option=client lanman auth=yes',
                '--option=client ntlmv2 auth=no',
                '--option=client plaintext auth=yes',
                '-c', 'ls'
            ]
            
            result = subprocess.run(
                smb_command,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': f'Successfully connected to {self.kuka_ip}/PROGRAM',
                    'directory_listing': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': f'Connection failed: {result.stderr.strip() or result.stdout.strip()}',
                    'returncode': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Connection timeout - KUKA controller unreachable'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Connection test failed: {str(e)}'
            }

    
    def run(self, debug: bool = False):
        """Run the web service."""
        try:
            self.logger.info(f"Starting KUKA File Upload Service")
            self.logger.info(f"Web interface: http://0.0.0.0:{self.web_port}")
            self.logger.info(f"KUKA Controller: {self.kuka_ip}")
            self.logger.info(f"SMB Share: //{self.kuka_ip}/PROGRAM")
            self.logger.info("Press Ctrl+C to stop")
            
            self.app.run(
                host='0.0.0.0',
                port=self.web_port,
                debug=debug,
                threaded=True
            )
            
        except KeyboardInterrupt:
            self.logger.info("Service stopped by user")
        except Exception as e:
            self.logger.error(f"Service error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='KUKA File Upload Web Service')
    parser.add_argument('--kuka-ip', default='192.168.1.50', 
                       help='KUKA controller IP address (default: 192.168.1.50)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Web service port (default: 8000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check if smbclient is available
    try:
        subprocess.run(['smbclient', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: smbclient not found!")
        print("Please install samba client:")
        print("  sudo apt-get install samba-client")
        sys.exit(1)
    
    # Create and run service
    service = KukaFileUploadService(kuka_ip=args.kuka_ip, web_port=args.port)
    service.run(debug=args.debug)


if __name__ == "__main__":
    main()
