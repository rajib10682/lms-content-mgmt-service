#!/usr/bin/env python3

import sys

def test_flask_application():
    """Test Flask application startup and embedding functionality"""
    print("=== Testing Flask Application with hdbscan Fix ===")
    
    print("Testing Flask application startup...")
    
    try:
        from app import app, embedding_models
        print("✓ Flask app imports successfully")
    except Exception as e:
        print(f"✗ Flask app import failed: {e}")
        return False
    
    print("\nEmbedding models status:")
    for model_name, model in embedding_models.items():
        status = 'loaded' if model is not None else 'failed_to_load'
        print(f"  {model_name}: {status}")
    
    try:
        from app import extract_topics_from_text
        test_text = 'This is a test about machine learning and artificial intelligence in education technology.'
        topics = extract_topics_from_text(test_text)
        print(f"\n✓ Topic extraction test successful: {topics}")
    except Exception as e:
        print(f"\n✗ Topic extraction test failed: {e}")
        return False
    
    try:
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Health endpoint test successful")
                health_data = response.get_json()
                print(f"  Status: {health_data.get('status')}")
                print(f"  Whisper model: {health_data.get('whisper_model')}")
                print(f"  Embedding models: {health_data.get('embedding_models')}")
            else:
                print(f"✗ Health endpoint test failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Health endpoint test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Flask application is working correctly with hdbscan fix.")
    return True

if __name__ == "__main__":
    success = test_flask_application()
    
    if success:
        print("\n✅ hdbscan fix verification successful!")
        sys.exit(0)
    else:
        print("\n❌ hdbscan fix verification failed!")
        sys.exit(1)
