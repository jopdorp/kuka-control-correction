# KUKA File Upload Web Service

A web-based file upload service that allows drag-and-drop upload of KUKA program files directly to the KUKA controller via SMB/CIFS network share.

## Features

- **Web Interface**: Modern, responsive web UI with drag-and-drop functionality
- **Direct SMB Upload**: Files are uploaded directly to the KUKA controller's PROGRAM folder
- **File Validation**: Supports KRL program files (.src, .dat, .sub, .krl, etc.)
- **Connection Testing**: Built-in connection test to KUKA controller
- **Upload Statistics**: Track upload success/failure rates and data transfer
- **Security**: File type validation and size limits (10MB max)
- **Logging**: Comprehensive logging for troubleshooting

## Quick Start

### 1. Installation

```bash
# Install and start the service (replace IP with your KUKA controller IP)
./setup-file-upload-service.sh 192.168.1.50 8000 install
```

### 2. Access Web Interface

Open your web browser and navigate to:
```
http://[RASPBERRY_PI_IP]:8000
```

### 3. Upload Files

- Drag and drop KRL files onto the upload area, OR
- Click the upload area to browse and select files
- Supported file types: `.src`, `.dat`, `.sub`, `.krl`, `.xml`, `.txt`, `.cfg`, `.ini`

## Manual Installation

### Prerequisites

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y samba-client python3-pip

# Install Python dependencies
pip3 install Flask==2.3.2 Werkzeug==2.3.6
```

### Running the Service

```bash
# Navigate to the source directory
cd raspberry-pi/src

# Run the service manually
python3 kuka_file_upload_service.py --kuka-ip 192.168.1.50 --port 8000
```

## Configuration

### Command Line Options

```bash
python3 kuka_file_upload_service.py --help

Options:
  --kuka-ip IP      KUKA controller IP address (default: 192.168.1.50)
  --port PORT       Web service port (default: 8000)
  --debug           Enable debug mode
```

### Network Configuration

The service connects to the KUKA controller using SMB with these parameters:
- **SMB Share**: `//[KUKA_IP]/PROGRAM`
- **Username**: `kuka` (anonymous)
- **Protocol**: NT1 (legacy SMB for compatibility)
- **Authentication**: Plain text (required for older KUKA systems)

### KUKA Controller Setup

Ensure your KUKA controller has:
1. **Network connectivity** to the Raspberry Pi
2. **SMB/CIFS sharing enabled** for the PROGRAM folder
3. **Anonymous access** configured (username: kuka, no password)

## Service Management

### Using the Setup Script

```bash
# Start service
./setup-file-upload-service.sh start

# Stop service
./setup-file-upload-service.sh stop

# Check status
./setup-file-upload-service.sh status

# View live logs
./setup-file-upload-service.sh logs

# Test manually
./setup-file-upload-service.sh test
```

### Using systemctl (if installed as service)

```bash
# Start/stop service
sudo systemctl start kuka-file-upload
sudo systemctl stop kuka-file-upload

# Check status
sudo systemctl status kuka-file-upload

# View logs
journalctl -u kuka-file-upload -f
```

## API Endpoints

### File Upload
- **URL**: `POST /upload`
- **Content-Type**: `multipart/form-data`
- **Parameters**: `file` (file to upload)
- **Response**: JSON with success status and details

### Connection Test
- **URL**: `GET /test-connection`
- **Response**: JSON with connection status

### Service Status
- **URL**: `GET /status`
- **Response**: JSON with service statistics and configuration

## Supported File Types

| Extension | Description |
|-----------|-------------|
| `.src` | KUKA Robot Language (KRL) source files |
| `.dat` | KUKA data files (points, configurations) |
| `.sub` | KUKA subroutine files |
| `.krl` | Additional KRL files |
| `.xml` | Configuration XML files |
| `.txt` | Text documentation files |
| `.cfg` | Configuration files |
| `.ini` | Initialization files |

## Security Considerations

- **File Type Validation**: Only allowed file extensions are accepted
- **File Size Limit**: Maximum 10MB per file
- **Filename Sanitization**: Filenames are sanitized to prevent path traversal
- **Network Security**: Uses SMB protocol over local network only
- **No Authentication**: Service intended for trusted network environments

## Troubleshooting

### Common Issues

1. **"smbclient not found"**
   ```bash
   sudo apt-get install samba-client
   ```

2. **"Connection failed"**
   - Verify KUKA controller IP address
   - Check network connectivity: `ping [KUKA_IP]`
   - Verify SMB share is accessible: `smbclient //[KUKA_IP]/PROGRAM -U %kuka -c ls`

3. **"Upload timeout"**
   - Check network stability
   - Verify KUKA controller is powered on
   - Check firewall settings

4. **"Permission denied"**
   - Verify KUKA SMB share allows write access
   - Check KUKA user permissions

### Testing SMB Connection

```bash
# Test SMB connection manually
smbclient //192.168.1.50/PROGRAM -U "%kuka" \
  --option="client min protocol=NT1" \
  --option="client max protocol=NT1" \
  --option="client lanman auth=yes" \
  --option="client ntlmv2 auth=no" \
  --option="client plaintext auth=yes" \
  -c "ls"
```

### Debug Mode

Run the service with debug mode for detailed logging:
```bash
python3 kuka_file_upload_service.py --debug --kuka-ip 192.168.1.50
```

## Integration with Existing System

This service can run alongside other KUKA integration components:

- **Vision Correction System**: Runs on different ports, no conflicts
- **Correction Helper**: Independent TCP service, can run simultaneously
- **Network Resources**: Uses different ports (8000 for web, vs 7000/7001 for correction system)

## Example Usage

### From Web Browser
1. Navigate to `http://raspberry-pi-ip:8000`
2. Test connection to KUKA controller
3. Drag and drop `.src` and `.dat` files
4. Monitor upload progress and status

### Programmatic Upload
```javascript
// JavaScript example for programmatic upload
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('Upload successful:', data.message);
    } else {
        console.error('Upload failed:', data.error);
    }
});
```

## Logs and Monitoring

### Log Locations
- **Service logs**: `journalctl -u kuka-file-upload`
- **Manual run logs**: Console output
- **Web server logs**: Included in service logs

### Key Log Messages
- `Successfully uploaded [filename] to KUKA controller` - Upload success
- `SMB upload failed: [error]` - SMB connection or transfer error
- `Connection test successful` - KUKA controller reachable
- `File type not allowed` - Invalid file extension

## License

This software is part of the KUKA Vision-Based Position Correction System and follows the same licensing terms as the parent project.
