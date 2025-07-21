#!/usr/bin/env python3

import sys
import os

def investigate_sentence_transformers():
    """Investigate sentence-transformers HTTP client and SSL configuration"""
    print("=== Investigating SentenceTransformer HTTP Client ===")
    
    try:
        import sentence_transformers
        print(f"✓ SentenceTransformer version: {sentence_transformers.__version__}")
        print(f"✓ SentenceTransformer location: {sentence_transformers.__file__}")
    except Exception as e:
        print(f"✗ Failed to import sentence_transformers: {e}")
        return
    
    try:
        import sentence_transformers.util
        print("✓ Found sentence_transformers.util module")
        
        import inspect
        util_source = inspect.getsource(sentence_transformers.util)
        
        http_libs = []
        if 'requests' in util_source:
            http_libs.append('requests')
        if 'urllib' in util_source:
            http_libs.append('urllib')
        if 'httpx' in util_source:
            http_libs.append('httpx')
        
        print(f"✓ HTTP libraries found in sentence_transformers.util: {http_libs}")
        
    except Exception as e:
        print(f"✗ Failed to inspect sentence_transformers.util: {e}")
    
    try:
        import huggingface_hub
        print(f"✓ huggingface_hub version: {huggingface_hub.__version__}")
        
        import huggingface_hub.utils
        print("✓ Found huggingface_hub.utils module")
        
    except Exception as e:
        print(f"✗ Failed to inspect huggingface_hub: {e}")
    
    print("\n=== Testing Different Model Downloads ===")
    
    try:
        from app import configure_ssl_context
        configure_ssl_context()
        print("✓ SSL context configured")
        
        from sentence_transformers import SentenceTransformer
        
        print("Testing model download with potential SSL issues...")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("✓ Alternative model loaded successfully")
        
    except Exception as e:
        print(f"✗ Alternative model test failed: {e}")
        error_str = str(e)
        if "SSL" in error_str or "certificate" in error_str:
            print("🔍 SSL certificate error detected!")
            print(f"Error details: {error_str}")

if __name__ == "__main__":
    investigate_sentence_transformers()
