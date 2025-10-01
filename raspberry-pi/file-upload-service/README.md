# KUKA File Upload Service - Docker Setup

A containerized web service for uploading files to KUKA controllers via SMB/CIFS.

## Quick Start
3. **Start the service:**
   ```bash
   docker-compose up -d
   ```
4. **Access the web interface:**
   - Direct access: http://localhost:8000
   - Via nginx: http://localhost (if nginx service is enabled)

## Configuration

### Environment Variables

- `KUKA_IP`: Hostname/IP for KUKA controller connection (default: `krcpc`)
- `WEB_PORT`: Port for web service (default: `8000`)
- `KUKA_CONTROLLER_IP`: Actual IP address mapped to `krcpc` hostname

### Docker Compose Services

- **kuka-file-upload**: Main Flask application
- **nginx** (optional): Reverse proxy with SSL support

## Usage

### Basic Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f kuka-file-upload

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Access container shell
docker-compose exec kuka-file-upload sh
```

### File Upload

1. Open web interface in browser
2. Drag and drop files or use file selector
3. Files are automatically transferred to KUKA controller via SMB

### Supported File Types

- `.src` - KUKA source files
- `.dat` - KUKA data files
- `.sub` - KUKA subroutines
- `.krl` - KUKA Robot Language files
- `.xml`, `.txt`, `.cfg`, `.ini` - Configuration files

## Network Configuration

The service uses `/etc/hosts` mapping to resolve `krcpc` to your KUKA controller IP.
Adjust the `extra_hosts` section in `docker-compose.yml` or use the `.env` file.

## SSL/HTTPS (Optional)

To enable HTTPS via nginx:

1. Place SSL certificates in `./ssl/` directory
2. Uncomment SSL configuration in `docker-compose.yml`
3. Update nginx configuration as needed

## Troubleshooting

### Check service status:
```bash
docker-compose ps
```

### View detailed logs:
```bash
docker-compose logs kuka-file-upload
```

### Test SMB connection:
```bash
docker-compose exec kuka-file-upload smbclient -L //krcpc -U ''
```

### Check network connectivity:
```bash
docker-compose exec kuka-file-upload ping krcpc
```

## Development

### Local testing without Docker:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 kuka_file_upload_service.py --kuka-ip krcpc --port 8000
```

## Migration from systemd service

If migrating from the previous systemd-based setup:

1. Stop old service: `sudo systemctl stop kuka-file-upload`
2. Disable old service: `sudo systemctl disable kuka-file-upload`
3. Remove service file: `sudo rm /etc/systemd/system/kuka-file-upload.service`
4. Start Docker containers: `docker-compose up -d`
