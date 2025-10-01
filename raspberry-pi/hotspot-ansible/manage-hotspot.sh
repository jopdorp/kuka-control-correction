#!/bin/bash
"""
WiFi Hotspot Ansible Management Script
Handles encrypted secrets and playbook execution
"""

set -e

PLAYBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_FILE="$PLAYBOOK_DIR/secrets.yml"
INVENTORY_FILE="$PLAYBOOK_DIR/inventory.ini"
PLAYBOOK_FILE="$PLAYBOOK_DIR/hotspot-playbook.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  WiFi Hotspot Ansible Manager${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo
}

print_usage() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  encrypt-secrets  - Encrypt the secrets.yml file"
    echo "  edit-secrets     - Edit encrypted secrets (requires password)"
    echo "  decrypt-secrets  - Decrypt secrets.yml (for viewing)"
    echo "  setup-hotspot    - Run the hotspot setup playbook"
    echo "  test-connection  - Test connectivity to the Pi"
    echo "  status           - Check hotspot services status"
    echo "  help             - Show this help message"
    echo
    echo "Examples:"
    echo "  $0 encrypt-secrets   # First time setup"
    echo "  $0 edit-secrets      # Change WiFi password"
    echo "  $0 setup-hotspot     # Deploy configuration"
}

check_requirements() {
    if ! command -v ansible-playbook &> /dev/null; then
        echo -e "${RED}Error: ansible-playbook not found${NC}"
        echo "Install with: sudo apt install ansible"
        exit 1
    fi
    
    if ! command -v ansible-vault &> /dev/null; then
        echo -e "${RED}Error: ansible-vault not found${NC}"
        exit 1
    fi
}

encrypt_secrets() {
    print_header
    echo -e "${YELLOW}Encrypting secrets.yml...${NC}"
    echo "You will be prompted to create a vault password."
    echo "Remember this password - you'll need it to run the playbook!"
    echo
    
    if [ -f "$SECRETS_FILE.encrypted" ]; then
        echo -e "${RED}Warning: $SECRETS_FILE.encrypted already exists${NC}"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
    fi
    
    ansible-vault encrypt "$SECRETS_FILE"
    echo -e "${GREEN}✅ Secrets encrypted successfully!${NC}"
    echo "File encrypted as: $SECRETS_FILE"
}

edit_secrets() {
    print_header
    echo -e "${YELLOW}Editing encrypted secrets...${NC}"
    
    if [ ! -f "$SECRETS_FILE" ]; then
        echo -e "${RED}Error: $SECRETS_FILE not found${NC}"
        echo "Run '$0 encrypt-secrets' first"
        exit 1
    fi
    
    ansible-vault edit "$SECRETS_FILE"
    echo -e "${GREEN}✅ Secrets updated!${NC}"
}

decrypt_secrets() {
    print_header
    echo -e "${YELLOW}Decrypting secrets for viewing...${NC}"
    echo -e "${RED}Warning: This will show passwords in plaintext!${NC}"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    ansible-vault view "$SECRETS_FILE"
}

setup_hotspot() {
    print_header
    echo -e "${YELLOW}Running hotspot setup playbook...${NC}"
    echo "You will be prompted for the vault password."
    echo
    
    if [ ! -f "$INVENTORY_FILE" ]; then
        echo -e "${RED}Error: $INVENTORY_FILE not found${NC}"
        echo "Please ensure inventory.ini exists with your Pi's IP address"
        exit 1
    fi
    
    if [ ! -f "$SECRETS_FILE" ]; then
        echo -e "${RED}Error: $SECRETS_FILE not found or not encrypted${NC}"
        echo "Run '$0 encrypt-secrets' first"
        exit 1
    fi
    
    ansible-playbook -i "$INVENTORY_FILE" "$PLAYBOOK_FILE" --ask-vault-pass
    
    echo
    echo -e "${GREEN}✅ Hotspot setup complete!${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Reboot the Pi: ansible raspberry_pi -i inventory.ini -m reboot -b"
    echo "2. Test connection from your device"
}

test_connection() {
    print_header
    echo -e "${YELLOW}Testing connection to Pi...${NC}"
    
    ansible raspberry_pi -i "$INVENTORY_FILE" -m ping
    
    echo
    echo -e "${YELLOW}Checking hotspot services...${NC}"
    ansible raspberry_pi -i "$INVENTORY_FILE" -m shell -a "systemctl is-active hostapd dnsmasq" -b
}

check_status() {
    print_header
    echo -e "${YELLOW}Checking hotspot services status...${NC}"
    
    ansible raspberry_pi -i "$INVENTORY_FILE" -m shell -a "
        echo 'Hostapd:' && systemctl is-active hostapd
        echo 'Dnsmasq:' && systemctl is-active dnsmasq
        echo 'WiFi Interface:'
        ip addr show wlan0 | grep -E 'inet |UP'
        echo 'Connected Clients:'
        iw dev wlan0 station dump | grep -c Station || echo 'No clients connected'
    " -b 2>/dev/null || echo -e "${RED}Cannot connect to Pi${NC}"
}

# Main script logic
case "${1:-help}" in
    encrypt-secrets)
        check_requirements
        encrypt_secrets
        ;;
    edit-secrets)
        check_requirements
        edit_secrets
        ;;
    decrypt-secrets)
        check_requirements
        decrypt_secrets
        ;;
    setup-hotspot)
        check_requirements
        setup_hotspot
        ;;
    test-connection)
        check_requirements
        test_connection
        ;;
    status)
        check_requirements
        check_status
        ;;
    help|*)
        print_usage
        ;;
esac
