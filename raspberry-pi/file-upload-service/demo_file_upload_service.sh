#!/bin/bash
"""
KUKA File Upload Service - Demo and Setup Script

This script demonstrates the complete file upload service setup and provides
examples of all functionality.
"""

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUKA_IP="${1:-192.168.1.50}"
WEB_PORT="${2:-8000}"

echo "🤖 KUKA File Upload Service - Complete Demo"
echo "==========================================="
echo "Project: $PROJECT_DIR"
echo "KUKA IP: $KUKA_IP" 
echo "Web Port: $WEB_PORT"
echo

# Function to create demo files
create_demo_files() {
    echo "📁 Creating demo KRL files..."
    
    # Create demo directory
    mkdir -p "$PROJECT_DIR/demo_files"
    
    # Demo KRL source file
    cat > "$PROJECT_DIR/demo_files/DemoProgram.src" << 'EOF'
&ACCESS RVP
&REL 2
&PARAM TEMPLATE = C:\KRC\Roboter\Template\vorgabe
&PARAM EDITMASK = *
DEF DemoProgram ( )
;FOLD INI
  ;FOLD BASISTECH INI
    GLOBAL INTERRUPT DECL 3 WHEN $STOPMESS==TRUE DO IR_STOPM ( )
    INTERRUPT ON 3 
    BAS (#INITMOV,0 )
  ;ENDFOLD (BASISTECH INI)
  ;FOLD USER INI
    ;Make your modifications here
  ;ENDFOLD (USER INI)
;ENDFOLD (INI)

; Demo program uploaded via web service
; This program demonstrates basic robot movements

PTP HOME Vel= 100 % DEFAULT
WAIT SEC 1.0

; Move to demonstration points
PTP P1 Vel= 50 % PDAT1 Tool[1] Base[1]
LIN P2 Vel= 0.5 m/s CPDAT1 Tool[1] Base[1]
CIRC P3, P4 Vel= 0.3 m/s CPDAT1 Tool[1] Base[1]

; Return home
PTP HOME Vel= 100 % DEFAULT

; End of demo program
END
EOF

    # Demo data file
    cat > "$PROJECT_DIR/demo_files/DemoProgram.dat" << 'EOF'
&ACCESS RVP
&REL 2
&PARAM TEMPLATE = C:\KRC\Roboter\Template\vorgabe
DEFDAT DemoProgram
;FOLD EXTERNAL DECLARATIONS
;ENDFOLD (EXTERNAL DECLARATIONS)

;FOLD DECLARATION
DECL E6POS P1={X 1000.0, Y 0.0, Z 1200.0, A 0.0, B 90.0, C 0.0, S 2, T 35}
DECL E6POS P2={X 1200.0, Y 200.0, Z 1200.0, A 0.0, B 90.0, C 0.0, S 2, T 35}
DECL E6POS P3={X 1200.0, Y 400.0, Z 1000.0, A 0.0, B 90.0, C 0.0, S 2, T 35}
DECL E6POS P4={X 1000.0, Y 400.0, Z 1200.0, A 0.0, B 90.0, C 0.0, S 2, T 35}
DECL E6POS HOME={X 1000.0, Y 0.0, Z 1500.0, A 0.0, B 90.0, C 0.0, S 2, T 35}

DECL PDAT PDAT1={VEL 100.0, ACC 100.0, APO_DIST 50.0}
DECL CPDAT CPDAT1={VEL_CP 1.0, ACC_CP 100.0, RADIUS_CP 10.0}
;ENDFOLD (DECLARATION)

ENDDAT
EOF

    # Demo configuration file
    cat > "$PROJECT_DIR/demo_files/demo_config.ini" << 'EOF'
[DEMO_SETTINGS]
; Demo program configuration
TOOL_NUMBER=1
BASE_NUMBER=1
VELOCITY_OVERRIDE=75
SAFETY_MODE=T1

[POSITIONS]
; Demo position parameters
HOME_HEIGHT=1500
WORK_HEIGHT=1200
SAFETY_HEIGHT=1000
WORK_RADIUS=400

[TIMING]
; Movement timing
DWELL_TIME=1.0
APPROACH_VELOCITY=50
WORK_VELOCITY=25
EOF

    echo "✅ Demo files created in demo_files/"
    ls -la "$PROJECT_DIR/demo_files/"
}

# Function to show web interface
demo_web_interface() {
    echo "🌐 Starting web service demo..."
    
    cd "$PROJECT_DIR/raspberry-pi/src"
    
    # Start service in background
    echo "Starting KUKA File Upload Service on port $WEB_PORT..."
    /home/jopdorp/development/kuka-control-correction/venv/bin/python kuka_file_upload_service.py \
        --kuka-ip "$KUKA_IP" --port "$WEB_PORT" &
    SERVICE_PID=$!
    
    # Wait for service to start
    sleep 3
    
    echo "🎯 Service running at:"
    echo "   Web Interface: http://$(hostname -I | awk '{print $1}'):$WEB_PORT"
    echo "   Local: http://localhost:$WEB_PORT"
    echo
    echo "📋 Available endpoints:"
    echo "   GET  /           - Main upload interface"
    echo "   POST /upload     - File upload endpoint" 
    echo "   GET  /status     - Service status and stats"
    echo "   GET  /test-connection - Test KUKA connection"
    echo
    echo "Press any key to stop the demo service..."
    read -n 1 -s
    
    # Stop service
    kill $SERVICE_PID 2>/dev/null || true
    wait $SERVICE_PID 2>/dev/null || true
    echo "Service stopped."
}

# Function to test upload functionality
test_upload_functionality() {
    echo "🧪 Testing upload functionality..."
    
    if command -v curl &> /dev/null; then
        echo "Testing with curl..."
        
        # Test status endpoint
        echo "1. Testing status endpoint..."
        curl -s "http://localhost:$WEB_PORT/status" | \
            python3 -m json.tool 2>/dev/null || echo "Service not running"
        
        # Test connection endpoint
        echo -e "\n2. Testing connection endpoint..."
        curl -s "http://localhost:$WEB_PORT/test-connection" | \
            python3 -m json.tool 2>/dev/null || echo "Service not running"
        
        echo -e "\n3. To test file upload:"
        echo "   curl -X POST -F 'file=@demo_files/DemoProgram.src' http://localhost:$WEB_PORT/upload"
        
    else
        echo "curl not found - install curl to test API endpoints"
    fi
}

# Function to show integration examples
show_integration_examples() {
    echo "🔧 Integration Examples"
    echo "====================="
    
    echo "1. Systemd Service:"
    echo "   ./setup-file-upload-service.sh $KUKA_IP $WEB_PORT install"
    echo
    
    echo "2. Manual Python Execution:"
    echo "   cd raspberry-pi/src"
    echo "   python3 kuka_file_upload_service.py --kuka-ip $KUKA_IP --port $WEB_PORT"
    echo
    
    echo "3. Nginx Reverse Proxy:"
    echo "   sudo cp nginx-kuka-upload.conf /etc/nginx/sites-available/kuka-upload"
    echo "   sudo ln -s /etc/nginx/sites-available/kuka-upload /etc/nginx/sites-enabled/"
    echo "   sudo systemctl reload nginx"
    echo
    
    echo "4. Test Script:"
    echo "   python3 test_file_upload_service.py --kuka-ip $KUKA_IP"
    echo
    
    echo "5. SMB Client Direct Test:"
    echo "   smbclient //$KUKA_IP/PROGRAM -U '%kuka' \\"
    echo "     --option='client min protocol=NT1' \\"
    echo "     --option='client max protocol=NT1' \\"
    echo "     --option='client lanman auth=yes' \\"
    echo "     --option='client ntlmv2 auth=no' \\"
    echo "     --option='client plaintext auth=yes' \\"
    echo "     -c 'ls'"
}

# Function to display usage
usage() {
    echo "Usage: $0 [KUKA_IP] [WEB_PORT] [COMMAND]"
    echo
    echo "Arguments:"
    echo "  KUKA_IP   KUKA controller IP (default: 192.168.1.50)"
    echo "  WEB_PORT  Web service port (default: 8000)"
    echo
    echo "Commands:"
    echo "  demo      Run complete demo (default)"
    echo "  files     Create demo files only"
    echo "  web       Start web interface demo"
    echo "  test      Test upload functionality"
    echo "  examples  Show integration examples"
    echo "  help      Show this help"
}

# Main demo function
main() {
    case "${3:-demo}" in
        "demo")
            create_demo_files
            demo_web_interface
            test_upload_functionality
            show_integration_examples
            ;;
        "files")
            create_demo_files
            ;;
        "web")
            demo_web_interface
            ;;
        "test")
            test_upload_functionality
            ;;
        "examples")
            show_integration_examples
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            echo "Unknown command: ${3:-demo}"
            usage
            exit 1
            ;;
    esac
}

# Check dependencies
echo "🔍 Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

# Check Flask
if ! /home/jopdorp/development/kuka-control-correction/venv/bin/python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not installed - run: pip3 install Flask"
    echo "Continuing with limited functionality..."
fi

# Check smbclient
if ! command -v smbclient &> /dev/null; then
    echo "⚠️  smbclient not found - install with: sudo apt-get install samba-client"
    echo "Upload functionality will not work without smbclient"
fi

echo "✅ Dependency check complete"
echo

# Run main function
main "$@"

echo
echo "🎉 Demo complete!"
echo
echo "📚 Next Steps:"
echo "  1. Review FILE_UPLOAD_SERVICE.md for detailed documentation"
echo "  2. Configure your KUKA controller IP in scripts"
echo "  3. Install as system service: ./setup-file-upload-service.sh install"
echo "  4. Access web interface and start uploading KRL files!"
