# WiFi Hotspot Setup with Ansible

This Ansible playbook sets up a WiFi hotspot on a Raspberry Pi with encrypted secrets and custom DNS for easy access to KUKA services.

## Features

- 🔒 **Encrypted secrets** using Ansible Vault
- 🌐 **Custom DNS** - Pi responds to `kukacam`, `kukacontrol`, and `krcpc` domains
- 📡 **WiFi Hotspot** with configurable SSID/password
- 🔧 **Service management** scripts included

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
- **File Upload:** http://kukacam:8000 or http://kukacontrol:8000
- **SSH to Pi:** `ssh pi@kukacam`
- **KUKA Controller:** `ping krcpc`
- **Direct IP:** http://192.168.4.1:8000

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
```bash
# Check service status
sudo systemctl status hostapd dnsmasq

# Restart services
sudo systemctl restart hostapd dnsmasq

# View connected clients
sudo iw dev wlan0 station dump
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
