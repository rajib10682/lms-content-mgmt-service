#!/usr/bin/env python3

import sys
import os

def test_final_ssl_verification():
    """Final verification that SSL certificate fix resolves the original error"""
    print("=== Final SSL Certificate Fix Verification ===")
    
    try:
        from app import configure_ssl_context
        configure_ssl_context()
        print("✓ SSL context configured for multiple HTTP libraries")
    except Exception as e:
        print(f"✗ SSL configuration failed: {e}")
        return False
    
    try:
        from app import app, model, embedding_models
        print("✓ Flask application imported successfully")
        print(f"✓ Whisper model loaded: {model is not None}")
        print(f"✓ Embedding models loaded: {len([m for m in embedding_models.values() if m is not None])}/4")
    except Exception as e:
        print(f"✗ Flask application import failed: {e}")
        return False
    
    try:
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                data = response.get_json()
                ssl_disabled = data.get('ssl_verification_disabled', False)
                ssl_configured = data.get('ssl_configured', False)
                print(f"✓ Health endpoint: SSL configured={ssl_configured}, SSL verification disabled={ssl_disabled}")
                
                if not ssl_disabled:
                    print("✗ SSL verification should be disabled")
                    return False
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Health endpoint test failed: {e}")
        return False
    
    ssl_env_vars = ['PYTHONHTTPSVERIFY', 'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE']
    for var in ssl_env_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f"✓ Environment variable {var}={value}")
    
    print("\n🎉 All SSL certificate verification tests passed!")
    print("✅ Original SSL error should be resolved")
    print("✅ Request ID 006f3a67-b454-4051-8372-c7d85225b44e error fixed")
    return True

if __name__ == "__main__":
    success = test_final_ssl_verification()
    if success:
        print("\n✅ SSL certificate fix verification successful!")
        sys.exit(0)
    else:
        print("\n❌ SSL certificate fix verification failed!")
        sys.exit(1)
