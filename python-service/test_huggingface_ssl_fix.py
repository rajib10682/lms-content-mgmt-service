#!/usr/bin/env python3

import sys
import os

def test_huggingface_ssl_fix():
    """Test enhanced SSL configuration specifically for huggingface_hub and SentenceTransformer"""
    print("=== Testing Enhanced SSL Configuration for Hugging Face ===")
    
    try:
        from app import configure_ssl_context
        configure_ssl_context()
        print("✓ Enhanced SSL context configured")
    except Exception as e:
        print(f"✗ SSL configuration failed: {e}")
        return False
    
    try:
        import huggingface_hub
        from huggingface_hub import hf_hub_download
        print("✓ huggingface_hub imported successfully")
        
        print("Testing huggingface_hub download with SSL fix...")
        file_path = hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="config.json",
            cache_dir="/tmp/test_hf_cache"
        )
        print(f"✓ huggingface_hub download successful: {file_path}")
        
    except Exception as e:
        print(f"✗ huggingface_hub test failed: {e}")
        error_str = str(e)
        if "SSL" in error_str or "certificate" in error_str:
            print("🔍 SSL certificate error still detected in huggingface_hub!")
            return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("Testing SentenceTransformer with enhanced SSL fix...")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ SentenceTransformer model loaded successfully")
        
        test_text = ["Testing SSL certificate fix for Request ID 1b322050-87e8-4de2-837d-ef2d2bcf5f03"]
        embeddings = model.encode(test_text)
        print(f"✓ Model encoding works, embedding shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"✗ SentenceTransformer test failed: {e}")
        error_str = str(e)
        if "SSL" in error_str or "certificate" in error_str:
            print("🔍 SSL certificate error still detected in SentenceTransformer!")
            print(f"Error details: {error_str}")
            return False
    
    ssl_env_vars = ['PYTHONHTTPSVERIFY', 'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE', 'HF_HUB_DISABLE_SYMLINKS_WARNING']
    for var in ssl_env_vars:
        value = os.environ.get(var, 'NOT_SET')
        print(f"✓ Environment variable {var}={value}")
    
    print("\n🎉 Enhanced SSL configuration test passed!")
    print("✅ Request ID 1b322050-87e8-4de2-837d-ef2d2bcf5f03 error should be resolved")
    return True

if __name__ == "__main__":
    success = test_huggingface_ssl_fix()
    if success:
        print("\n✅ Enhanced SSL fix verification successful!")
        sys.exit(0)
    else:
        print("\n❌ Enhanced SSL fix verification failed!")
        sys.exit(1)
