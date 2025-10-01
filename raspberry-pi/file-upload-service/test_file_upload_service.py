#!/usr/bin/env python3
"""
Test script for KUKA File Upload Service

This script tests various components of the file upload service including:
- SMB connection to KUKA controller
- File upload functionality
- Web service endpoints
- Error handling

Usage:
    python3 test_file_upload_service.py --kuka-ip 192.168.1.50
"""

import sys
import os
import argparse
import tempfile
import subprocess
import requests
import time
import json
from pathlib import Path

def test_smbclient():
    """Test if smbclient is available."""
    print("1. Testing smbclient availability...")
    try:
        result = subprocess.run(['smbclient', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"   ✅ smbclient found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ smbclient not found - please install samba-client")
        return False

def test_kuka_connection(kuka_ip: str):
    """Test SMB connection to KUKA controller."""
    print(f"2. Testing SMB connection to {kuka_ip}...")
    
    try:
        cmd = [
            'smbclient', f'//{kuka_ip}/PROGRAM',
            '-U', '%kuka',
            '--option=client min protocol=NT1',
            '--option=client max protocol=NT1',
            '--option=client lanman auth=yes',
            '--option=client ntlmv2 auth=no',
            '--option=client plaintext auth=yes',
            '-c', 'ls'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"   ✅ Connection successful")
            print(f"   📁 Directory listing preview:")
            lines = result.stdout.strip().split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"      {line}")
            return True
        else:
            print(f"   ❌ Connection failed: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Connection timeout - KUKA controller unreachable")
        return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False

def create_test_files():
    """Create test files for upload."""
    print("3. Creating test files...")
    
    test_files = []
    
    # Create a test KRL source file
    src_content = '''&ACCESS RVP
&REL 2
&PARAM TEMPLATE = C:\\KRC\\Roboter\\Template\\vorgabe
&PARAM EDITMASK = *
DEF TestProgram ( )
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

; Test program uploaded via web service
PTP HOME Vel= 100 % DEFAULT
WAIT SEC 1.0

; End of test program
END
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.src', delete=False) as f:
        f.write(src_content)
        test_files.append((f.name, 'TestProgram.src'))
    
    # Create a test DAT file
    dat_content = '''&ACCESS RVP
&REL 2
&PARAM TEMPLATE = C:\\KRC\\Roboter\\Template\\vorgabe
DEFDAT TestProgram
;FOLD EXTERNAL DECLARATIONS
;ENDFOLD
;FOLD Declaration
DECL E6POS HOME={X 1000.0, Y 0.0, Z 1000.0, A 0.0, B 90.0, C 0.0, S 2, T 35}
;ENDFOLD

ENDDAT
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
        f.write(dat_content)
        test_files.append((f.name, 'TestProgram.dat'))
    
    print(f"   ✅ Created {len(test_files)} test files")
    return test_files

def test_direct_upload(test_files, kuka_ip: str):
    """Test direct SMB upload without web service."""
    print("4. Testing direct SMB file upload...")
    
    success_count = 0
    
    for local_path, remote_name in test_files:
        try:
            cmd = [
                'smbclient', f'//{kuka_ip}/PROGRAM',
                '-U', '%kuka',
                '--option=client min protocol=NT1',
                '--option=client max protocol=NT1',
                '--option=client lanman auth=yes',
                '--option=client ntlmv2 auth=no',
                '--option=client plaintext auth=yes',
                '-c', f'put "{local_path}" "TEST_{remote_name}"'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"   ✅ Uploaded TEST_{remote_name}")
                success_count += 1
            else:
                print(f"   ❌ Failed to upload TEST_{remote_name}: {result.stderr.strip()}")
                
        except Exception as e:
            print(f"   ❌ Upload error for TEST_{remote_name}: {e}")
    
    print(f"   📊 Direct upload results: {success_count}/{len(test_files)} successful")
    return success_count == len(test_files)

def test_web_service(test_files, service_url: str):
    """Test web service endpoints."""
    print(f"5. Testing web service at {service_url}...")
    
    # Test status endpoint
    try:
        response = requests.get(f"{service_url}/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            print(f"   ✅ Status endpoint working")
            print(f"      KUKA IP: {status_data.get('kuka_ip', 'Unknown')}")
            print(f"      Max file size: {status_data.get('max_file_size_mb', 'Unknown')} MB")
        else:
            print(f"   ❌ Status endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Status endpoint error: {e}")
        return False
    
    # Test connection endpoint
    try:
        response = requests.get(f"{service_url}/test-connection", timeout=10)
        if response.status_code == 200:
            conn_data = response.json()
            if conn_data.get('success'):
                print(f"   ✅ Connection test passed")
            else:
                print(f"   ⚠️  Connection test failed: {conn_data.get('error', 'Unknown')}")
        else:
            print(f"   ❌ Connection test endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection test error: {e}")
    
    # Test file uploads
    upload_success = 0
    for local_path, remote_name in test_files:
        try:
            with open(local_path, 'rb') as f:
                files = {'file': (f"WEBTEST_{remote_name}", f)}
                response = requests.post(f"{service_url}/upload", 
                                       files=files, timeout=30)
            
            if response.status_code == 200:
                upload_data = response.json()
                if upload_data.get('success'):
                    print(f"   ✅ Web upload successful: WEBTEST_{remote_name}")
                    upload_success += 1
                else:
                    print(f"   ❌ Web upload failed: {upload_data.get('error', 'Unknown')}")
            else:
                print(f"   ❌ Upload request failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Web upload error for WEBTEST_{remote_name}: {e}")
    
    print(f"   📊 Web upload results: {upload_success}/{len(test_files)} successful")
    return upload_success == len(test_files)

def cleanup_test_files(test_files, kuka_ip: str):
    """Clean up test files from local and remote."""
    print("6. Cleaning up test files...")
    
    # Clean up local files
    for local_path, _ in test_files:
        try:
            os.unlink(local_path)
        except Exception:
            pass
    
    # Clean up remote files (optional - they're just test files)
    cleanup_files = []
    for _, remote_name in test_files:
        cleanup_files.extend([f"TEST_{remote_name}", f"WEBTEST_{remote_name}"])
    
    for remote_file in cleanup_files:
        try:
            cmd = [
                'smbclient', f'//{kuka_ip}/PROGRAM',
                '-U', '%kuka',
                '--option=client min protocol=NT1',
                '--option=client max protocol=NT1',
                '--option=client lanman auth=yes',
                '--option=client ntlmv2 auth=no',
                '--option=client plaintext auth=yes',
                '-c', f'del "{remote_file}"'
            ]
            subprocess.run(cmd, capture_output=True, timeout=10)
        except Exception:
            pass  # Ignore cleanup errors
    
    print("   ✅ Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='Test KUKA File Upload Service')
    parser.add_argument('--kuka-ip', default='192.168.1.50',
                       help='KUKA controller IP address')
    parser.add_argument('--service-url', default='http://localhost:8000',
                       help='Web service URL')
    parser.add_argument('--skip-web-test', action='store_true',
                       help='Skip web service testing')
    
    args = parser.parse_args()
    
    print("KUKA File Upload Service Test")
    print("============================")
    print(f"KUKA IP: {args.kuka_ip}")
    print(f"Service URL: {args.service_url}")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Test 1: smbclient availability
    total_tests += 1
    if test_smbclient():
        tests_passed += 1
    else:
        print("Stopping tests - smbclient required")
        sys.exit(1)
    
    # Test 2: KUKA connection
    total_tests += 1
    if test_kuka_connection(args.kuka_ip):
        tests_passed += 1
    else:
        print("Warning: KUKA connection failed - some tests may not work")
    
    # Create test files
    test_files = create_test_files()
    
    # Test 3: Direct SMB upload
    total_tests += 1
    if test_direct_upload(test_files, args.kuka_ip):
        tests_passed += 1
    
    # Test 4: Web service (if not skipped)
    if not args.skip_web_test:
        total_tests += 1
        if test_web_service(test_files, args.service_url):
            tests_passed += 1
    
    # Cleanup
    cleanup_test_files(test_files, args.kuka_ip)
    
    # Summary
    print()
    print("Test Summary")
    print("============")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed - check configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
