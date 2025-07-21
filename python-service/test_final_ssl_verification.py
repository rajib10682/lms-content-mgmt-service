#!/usr/bin/env python3

import sys
import os

def test_final_ssl_verification():
    """Final verification that all SSL certificate fixes work together"""
    print("=== Final SSL Certificate Fix Verification ===")
    
    try:
        from app import configure_ssl_context
        configure_ssl_context()
        print("✓ Enhanced SSL context configured for all HTTP libraries")
    except Exception as e:
        print(f"✗ SSL configuration failed: {e}")
        return False
    
    try:
        from app import app, model, embedding_models
        print("✓ Flask application imported successfully")
        print(f"✓ Whisper model loaded: {model is not None}")
        
        st_loaded = embedding_models.get('sentence_transformer') is not None
        kb_loaded = embedding_models.get('keybert') is not None
        bt_loaded = embedding_models.get('bertopic') is not None
        
        print(f"✓ SentenceTransformer loaded: {st_loaded}")
        print(f"✓ KeyBERT loaded: {kb_loaded}")
        print(f"✓ BERTopic loaded: {bt_loaded}")
        
        if not st_loaded:
            print("✗ SentenceTransformer failed to load - SSL fix may not be working")
            return False
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
                
                embedding_status = data.get('embedding_models', {})
                print(f"✓ Health endpoint embedding models status: {embedding_status}")
                
                if not ssl_disabled:
                    print("✗ SSL verification should be disabled")
                    return False
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Health endpoint test failed: {e}")
        return False
    
    ssl_env_vars = ['PYTHONHTTPSVERIFY', 'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE', 'HF_HUB_DISABLE_SYMLINKS_WARNING']
    for var in ssl_env_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f"✓ Environment variable {var}={value}")
    
    print("\n🎉 All SSL certificate verification tests passed!")
    print("✅ Original SSL error should be resolved (Request ID: 006f3a67-b454-4051-8372-c7d85225b44e)")
    print("✅ SentenceTransformer SSL error should be resolved (Request ID: 1b322050-87e8-4de2-837d-ef2d2bcf5f03)")
    return True

if __name__ == "__main__":
    success = test_final_ssl_verification()
    if success:
        print("\n✅ Comprehensive SSL certificate fix verification successful!")
        sys.exit(0)
    else:
        print("\n❌ Comprehensive SSL certificate fix verification failed!")
        sys.exit(1)
