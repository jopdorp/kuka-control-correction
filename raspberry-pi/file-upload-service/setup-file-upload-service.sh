#!/bin/bash
"""
KUKA File Upload Service - Installation and Management Script

This script helps install and manage the KUKA File Upload Web Service on Raspberry Pi.
"""

set -e

SERVICE_NAME="kuka-file-upload"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUKA_IP="${1:-192.168.1.50}"
WEB_PORT="${2:-8000}"

echo "KUKA File Upload Service Manager"
echo "==============================="
echo "Project Directory: $PROJECT_DIR"
echo "KUKA IP: $KUKA_IP"
echo "Web Port: $WEB_PORT"
echo

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        echo "Error: Don't run this script as root. Use regular user account."
        exit 1
    fi
}

# Function to install system dependencies
install_dependencies() {
    echo "Installing system dependencies..."
    
    # Check if samba-client is installed
    if ! command -v smbclient &> /dev/null; then
        echo "Installing samba-client..."
        sudo apt-get update
        sudo apt-get install -y samba-client
    else
        echo "samba-client already installed"
    fi
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        pip3 install --user -r "$PROJECT_DIR/requirements.txt"
    else
        pip3 install --user Flask==2.3.2 Werkzeug==2.3.6
    fi
    
    echo "Dependencies installed successfully!"
}

# Function to create and install systemd service
install_service() {
    echo "Installing systemd service..."
    
    # Create service file content
    cat > /tmp/${SERVICE_NAME}.service << EOF
[Unit]
Description=KUKA File Upload Web Service
After=network.target
Requires=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR/raspberry-pi/src
ExecStart=/usr/bin/python3 $PROJECT_DIR/raspberry-pi/src/kuka_file_upload_service.py --kuka-ip $KUKA_IP --port $WEB_PORT
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment variables
Environment=PYTHONPATH=$PROJECT_DIR/raspberry-pi/src

[Install]
WantedBy=multi-user.target
EOF

    # Install service file
    sudo cp /tmp/${SERVICE_NAME}.service "$SERVICE_FILE"
    sudo systemctl daemon-reload
    
    echo "Service installed successfully!"
}

# Function to start service
start_service() {
    echo "Starting KUKA File Upload Service..."
    sudo systemctl start "$SERVICE_NAME"
    sudo systemctl enable "$SERVICE_NAME"
    
    sleep 2
    
    if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
        echo "✅ Service started successfully!"
        echo "🌐 Web interface available at: http://$(hostname -I | awk '{print $1}'):$WEB_PORT"
    else
        echo "❌ Service failed to start. Check logs with: journalctl -u $SERVICE_NAME -f"
        exit 1
    fi
}

# Function to stop service
stop_service() {
    echo "Stopping KUKA File Upload Service..."
    sudo systemctl stop "$SERVICE_NAME"
    sudo systemctl disable "$SERVICE_NAME"
    echo "Service stopped."
}

# Function to check service status
check_status() {
    echo "Service Status:"
    echo "==============="
    sudo systemctl status "$SERVICE_NAME" --no-pager || true
    echo
    echo "Recent Logs:"
    echo "============"
    sudo journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true
}

# Function to show usage
usage() {
    echo "Usage: $0 [KUKA_IP] [WEB_PORT] [COMMAND]"
    echo
    echo "Arguments:"
    echo "  KUKA_IP   KUKA controller IP address (default: 192.168.1.50)"
    echo "  WEB_PORT  Web service port (default: 8000)"
    echo
    echo "Commands:"
    echo "  install   Install dependencies and service"
    echo "  start     Start the service"
    echo "  stop      Stop the service"
    echo "  restart   Restart the service"
    echo "  status    Show service status and logs"
    echo "  logs      Show recent logs"
    echo "  test      Test the service manually"
    echo
    echo "Examples:"
    echo "  $0 192.168.1.50 8000 install"
    echo "  $0 start"
    echo "  $0 status"
}

# Main script logic
main() {
    check_root
    
    case "${3:-install}" in
        "install")
            install_dependencies
            install_service
            start_service
            ;;
        "start")
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            stop_service
            sleep 2
            start_service
            ;;
        "status")
            check_status
            ;;
        "logs")
            echo "Live logs (Ctrl+C to exit):"
            sudo journalctl -u "$SERVICE_NAME" -f
            ;;
        "test")
            echo "Testing service manually..."
            cd "$PROJECT_DIR/raspberry-pi/src"
            python3 kuka_file_upload_service.py --kuka-ip "$KUKA_IP" --port "$WEB_PORT"
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            echo "Unknown command: ${3:-install}"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
