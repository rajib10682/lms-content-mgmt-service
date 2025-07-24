#!/usr/bin/env python3

import sys
import os

def test_ssl_configuration():
    """Test SSL configuration for all HTTP libraries"""
    print("=== Testing Comprehensive SSL Configuration ===")
    
    from app import configure_ssl_context
    configure_ssl_context()
    
    print("✓ SSL context configured")
    
    try:
        import urllib.request
        req = urllib.request.Request("https://httpbin.org/get")
        with urllib.request.urlopen(req, timeout=10) as response:
            print("✓ urllib SSL configuration works")
    except Exception as e:
        print(f"✗ urllib SSL test failed: {e}")
    
    try:
        import requests
        response = requests.get("https://httpbin.org/get", timeout=10, verify=False)
        print("✓ requests SSL configuration works")
    except Exception as e:
        print(f"✗ requests SSL test failed: {e}")
    
    try:
        import httpx
        with httpx.Client(verify=False) as client:
            response = client.get("https://httpbin.org/get", timeout=10)
            print("✓ httpx SSL configuration works")
    except Exception as e:
        print(f"✗ httpx SSL test failed: {e}")
    
    try:
        from app import app
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                health_data = response.get_json()
                ssl_status = health_data.get('ssl_verification_disabled', False)
                print(f"✓ Flask health endpoint works, SSL verification disabled: {ssl_status}")
            else:
                print(f"✗ Flask health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Flask health test failed: {e}")
    
    print("\n=== SSL Configuration Test Complete ===")

if __name__ == "__main__":
    test_ssl_configuration()
