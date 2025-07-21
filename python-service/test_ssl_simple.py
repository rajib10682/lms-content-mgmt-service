#!/usr/bin/env python3

import sys
import os

def test_ssl_basic():
    """Basic SSL configuration test"""
    print("=== Testing Basic SSL Configuration ===")
    
    try:
        from app import configure_ssl_context
        configure_ssl_context()
        print("✓ SSL context configured successfully")
    except Exception as e:
        print(f"✗ SSL context configuration failed: {e}")
        return False
    
    try:
        from app import app
        print("✓ Flask app imported successfully")
    except Exception as e:
        print(f"✗ Flask app import failed: {e}")
        return False
    
    try:
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                data = response.get_json()
                ssl_disabled = data.get('ssl_verification_disabled', False)
                ssl_configured = data.get('ssl_configured', False)
                print(f"✓ Health endpoint works - SSL configured: {ssl_configured}, SSL verification disabled: {ssl_disabled}")
                return True
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Health endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ssl_basic()
    if success:
        print("\n✅ SSL configuration test passed!")
        sys.exit(0)
    else:
        print("\n❌ SSL configuration test failed!")
        sys.exit(1)
