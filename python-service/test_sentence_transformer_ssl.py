#!/usr/bin/env python3

import sys
import os

def test_sentence_transformer_ssl():
    """Test SentenceTransformer SSL configuration specifically"""
    print("=== Testing SentenceTransformer SSL Configuration ===")
    
    try:
        from app import configure_ssl_context
        configure_ssl_context()
        print("✓ SSL context configured")
    except Exception as e:
        print(f"✗ SSL configuration failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ SentenceTransformer imported successfully")
        
        print("Attempting to load SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ SentenceTransformer model loaded successfully")
        
        test_text = ["This is a test sentence"]
        embeddings = model.encode(test_text)
        print(f"✓ Model encoding works, embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ SentenceTransformer test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        error_str = str(e)
        if "SSL" in error_str or "certificate" in error_str:
            print("🔍 This appears to be an SSL certificate verification error")
        
        return False

if __name__ == "__main__":
    success = test_sentence_transformer_ssl()
    if success:
        print("\n✅ SentenceTransformer SSL test passed!")
        sys.exit(0)
    else:
        print("\n❌ SentenceTransformer SSL test failed!")
        sys.exit(1)
