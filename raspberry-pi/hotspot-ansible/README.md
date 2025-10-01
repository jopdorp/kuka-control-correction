# WiFi Hotspot Setup with Ansible

This Ansible playbook sets up a WiFi hotspot on a Raspberry Pi with encrypted secrets and custom DNS for easy access to KUKA services.

## Features

- 🔒 **Encrypted secrets** using Ansible Vault
- 🌐 **Custom DNS** - Pi responds to `kuka.drop` and controller to `krcpc` domain
- 📡 **WiFi Hotspot** with configurable SSID/password
- 🔧 **Service management** scripts included
- 🔄 **Mode Switching** - Easy switch between internet mode (phone hotspot) and hotspot mode (Pi broadcasts)
- 🐳 **Docker Integration** - Automatically installs Docker and starts the file upload service
- 📂 **Auto-deployment** - Clones and updates the kuka-control-correction repository

## Prerequisites

### On your control machine (laptop/desktop):
```bash
# Install Ansible
sudo apt update
sudo apt install ansible

# Or using pip
pip3 install ansible
```

### On the Raspberry Pi:
```bash
# Enable SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# Make sure you can SSH to the Pi
ssh pi@kukacam  # Replace with your Pi's IP
```

## Quick Setup

2. **Encrypt the secrets (first time only):**
   ```bash
   ./manage-hotspot.sh encrypt-secrets
   ```
   You'll be prompted to create a vault password. Remember this password!

3. **Edit WiFi credentials (optional):**
   ```bash
   ./manage-hotspot.sh edit-secrets
   ```
   
   Configure these settings in secrets.yml:
   - `wifi_hotspot.ssid`: Your Pi's hotspot name (default: KUKA-Control)
   - `wifi_hotspot.password`: Password for connecting to Pi's hotspot
   - `external_hotspot.name`: Your phone's hotspot name (for internet access)
   - `external_hotspot.password`: Your phone's hotspot password

4. **Deploy the hotspot:**
   ```bash
   ./manage-hotspot.sh setup-hotspot
   ```
   You'll be prompted for the vault password.

5. **Test connectivity:**
   ```bash
   ./manage-hotspot.sh test-connection
   ```

## Usage

After the playbook runs successfully:

### Start the hotspot:
```bash
# On the Raspberry Pi
sudo hotspot-control start
```

### Connect to the hotspot:
- **SSID:** `KUKA-Control` (or custom from secrets.yml)
- **Password:** From encrypted secrets.yml
- **Pi IP:** `192.168.4.1`

### Access Services via Custom Domains:
- **File Upload:** http://kuka.drop:8000
- **SSH to Pi:** `ssh pi@kuka.drop`
- **KUKA Controller:** `ping krcpc` or `ssh krcpc`
- **Direct IP:** http://192.168.4.1:8000

> 💡 **Note:** The `.drop` TLD is a special-use domain that provides better browser compatibility than custom TLDs like `.local`

### Management Commands:
```bash
# Encrypt secrets for first time
./manage-hotspot.sh encrypt-secrets

# Edit encrypted secrets (change WiFi password, etc.)
./manage-hotspot.sh edit-secrets

# View decrypted secrets (careful - shows passwords!)
./manage-hotspot.sh decrypt-secrets

# Deploy hotspot configuration
./manage-hotspot.sh setup-hotspot

# Test connection to Pi
./manage-hotspot.sh test-connection

# Check hotspot service status
./manage-hotspot.sh status
```

### On the Pi (after setup):

#### Switch Between Modes

The Pi can operate in two modes (both use wlan0, so only one at a time):

**Hotspot Mode** (default): Pi broadcasts its own WiFi network for the robot system
```bash
sudo wifi-mode hotspot
```

**Internet Mode**: Pi connects to your phone's hotspot for updates/downloads
```bash
sudo wifi-mode internet
```

**Check Current Mode**:
```bash
sudo wifi-mode status
```

#### Basic Service Management
```bash
# Check service status
sudo systemctl status hostapd dnsmasq

# Manual service control (for advanced users)
sudo hotspot-control start
sudo hotspot-control stop
sudo hotspot-control status

# View connected clients
sudo iw dev wlan0 station dump
```

## Typical Workflow

### 1. During Development (need internet for updates):
```bash
# On the Pi
sudo wifi-mode internet

# Now you have internet access for:
sudo apt update && sudo apt upgrade
pip3 install some-package
git pull
```

### 2. During Operation (robot needs to connect):
```bash
# On the Pi
sudo wifi-mode hotspot

# Now your laptop can connect to KUKA-Control WiFi
# Access services at http://kuka.drop:8000
```

## Customization

You can customize the hotspot settings by editing the variables in `hotspot-playbook.yml`:

```yaml
vars:
  hotspot_ssid: "MyKUKA-Network"        # Change network name
  hotspot_password: "my-secure-pass"     # Change password
  hotspot_channel: 6                     # Change WiFi channel
  hotspot_ip: "192.168.5.1"            # Change IP address
```

## Troubleshooting

### Check services:
```bash
systemctl status hostapd
systemctl status dnsmasq
systemctl status kuka-hotspot
```

### Check logs:
```bash
journalctl -u hostapd -f
journalctl -u dnsmasq -f
```

### Check network interface:
```bash
ip addr show wlan0
iwconfig wlan0
```

### Re-run playbook:
```bash
ansible-playbook hotspot-playbook.yml --ask-become-pass
```

## Files Created

The playbook creates these files on the Raspberry Pi:

- `/etc/hostapd/hostapd.conf` - WiFi access point configuration
- `/etc/dnsmasq.conf` - DHCP server configuration
- `/usr/local/bin/hotspot-control` - Control script
- `/etc/systemd/system/kuka-hotspot.service` - Systemd service
- `/usr/local/bin/hotspot-status` - Status checking script

## Integration with File Upload Service

To use with the Docker file upload service:

```bash
# Start hotspot first
sudo hotspot-control start

# Then start the file upload service
cd /path/to/file-upload-service
docker-compose up -d

# Service will be available at http://192.168.4.1:8000
```

## Security Notes

- The hotspot provides internet access if the Pi has ethernet connection
- Default password is `kukafiles123` - change it in the playbook variables
- The hotspot creates a bridge to your network - use in trusted environments only
- Consider disabling internet sharing if not needed by removing the NAT rules
