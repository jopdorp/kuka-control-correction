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
from flask import Flask, request, render_template_string, jsonify, redirect, url_for
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
            return render_template_string(self._get_html_template())
            
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
            # Build smbclient command
            smb_command = [
                'smbclient',
                f'//{self.kuka_ip}/PROGRAM',
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
            # Test with simple directory listing
            smb_command = [
                'smbclient',
                f'//{self.kuka_ip}/PROGRAM',
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
    
    def _get_html_template(self) -> str:
        """Get HTML template for upload interface."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KUKA File Upload Service</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .header {
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.8;
            font-size: 1.1em;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-area {
            border: 3px dashed #3498db;
            border-radius: 10px;
            padding: 60px 20px;
            text-align: center;
            transition: all 0.3s ease;
            background: #f8f9fa;
            margin-bottom: 30px;
            cursor: pointer;
        }
        
        .upload-area.dragover {
            border-color: #2ecc71;
            background: #e8f8f0;
            transform: scale(1.02);
        }
        
        .upload-area:hover {
            border-color: #2980b9;
            background: #ecf0f1;
        }
        
        .upload-icon {
            font-size: 4em;
            color: #3498db;
            margin-bottom: 20px;
        }
        
        .upload-text {
            font-size: 1.2em;
            color: #34495e;
            margin-bottom: 15px;
        }
        
        .upload-hint {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
            text-decoration: none;
            display: inline-block;
        }
        
        .btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
        }
        
        .btn.secondary {
            background: #95a5a6;
        }
        
        .btn.secondary:hover {
            background: #7f8c8d;
        }
        
        .status-area {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            display: none;
        }
        
        .status-area.success {
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }
        
        .status-area.error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }
        
        .status-area.loading {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
            display: none;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 30px;
        }
        
        .info-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }
        
        .info-card h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        
        .connection-status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .status-unknown { background: #f39c12; color: white; }
        .status-connected { background: #27ae60; color: white; }
        .status-disconnected { background: #e74c3c; color: white; }
        
        @media (max-width: 768px) {
            .container { margin: 10px; }
            .info-grid { grid-template-columns: 1fr; }
            .header { padding: 20px; }
            .header h1 { font-size: 2em; }
            .content { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 KUKA File Upload</h1>
            <p>Drag and drop files to upload to KUKA controller</p>
        </div>
        
        <div class="content">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div class="upload-text">Click here or drag files to upload</div>
                <div class="upload-hint">Supported: .src, .dat, .sub, .krl, .xml, .txt, .cfg, .ini (max 10MB)</div>
            </div>
            
            <input type="file" id="fileInput" class="file-input" multiple accept=".src,.dat,.sub,.krl,.xml,.txt,.cfg,.ini">
            
            <div class="text-center">
                <button class="btn" onclick="testConnection()">🔗 Test Connection</button>
                <button class="btn secondary" onclick="showStatus()">📊 View Status</button>
            </div>
            
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div class="status-area" id="statusArea"></div>
            
            <div class="info-grid">
                <div class="info-card">
                    <h3>Connection Info</h3>
                    <p><strong>KUKA IP:</strong> ''' + str(self.kuka_ip) + '''</p>
                    <p><strong>SMB Share:</strong> //''' + str(self.kuka_ip) + '''/PROGRAM</p>
                    <p><strong>Status:</strong> <span class="connection-status status-unknown" id="connectionStatus">Unknown</span></p>
                </div>
                
                <div class="info-card">
                    <h3>Upload Guidelines</h3>
                    <p>• Upload KRL programs (.src) and data files (.dat)</p>
                    <p>• Files are transferred directly to KUKA PROGRAM folder</p>
                    <p>• Ensure KUKA controller is powered and network accessible</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // File upload handling
        const uploadArea = document.querySelector('.upload-area');
        const fileInput = document.getElementById('fileInput');
        const statusArea = document.getElementById('statusArea');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        
        // Drag and drop events
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            handleFiles(files);
        });
        
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
        
        function handleFiles(files) {
            for (let file of files) {
                uploadFile(file);
            }
        }
        
        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            showStatus('Uploading ' + file.name + '...', 'loading');
            showProgress(0);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideProgress();
                
                if (data.success) {
                    showStatus('✅ Successfully uploaded: ' + file.name, 'success');
                } else {
                    showStatus('❌ Upload failed: ' + data.error, 'error');
                }
            })
            .catch(error => {
                hideProgress();
                showStatus('❌ Network error: ' + error.message, 'error');
            });
        }
        
        function testConnection() {
            showStatus('Testing connection...', 'loading');
            
            fetch('/test-connection')
            .then(response => response.json())
            .then(data => {
                const statusEl = document.getElementById('connectionStatus');
                
                if (data.success) {
                    showStatus('✅ Connection test successful!', 'success');
                    statusEl.textContent = 'Connected';
                    statusEl.className = 'connection-status status-connected';
                } else {
                    showStatus('❌ Connection failed: ' + data.error, 'error');
                    statusEl.textContent = 'Disconnected';
                    statusEl.className = 'connection-status status-disconnected';
                }
            })
            .catch(error => {
                showStatus('❌ Connection test error: ' + error.message, 'error');
                document.getElementById('connectionStatus').textContent = 'Error';
            });
        }
        
        function showStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                const info = `
                    📊 Service Statistics:
                    • Uptime: ${data.uptime_formatted}
                    • Uploads attempted: ${data.stats.uploads_attempted}
                    • Successful: ${data.stats.uploads_successful}
                    • Failed: ${data.stats.uploads_failed}
                    • Data transferred: ${(data.stats.total_bytes_transferred / 1024).toFixed(1)} KB
                `;
                showStatus(info, 'success');
            })
            .catch(error => {
                showStatus('❌ Failed to get status: ' + error.message, 'error');
            });
        }
        
        function showStatus(message, type) {
            statusArea.innerHTML = message.replace(/\\n/g, '<br>');
            statusArea.className = 'status-area ' + type;
            statusArea.style.display = 'block';
        }
        
        function showProgress(percent) {
            progressBar.style.display = 'block';
            progressFill.style.width = percent + '%';
        }
        
        function hideProgress() {
            progressBar.style.display = 'none';
            progressFill.style.width = '0%';
        }
        
        // Auto-test connection on page load
        document.addEventListener('DOMContentLoaded', testConnection);
    </script>
</body>
</html>
        '''
    
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
