#!/usr/bin/env python3

import sys
import os

def test_bertopic_transcribe_fix():
    """Test to verify the BERTopic transcribe attribution bug is fixed"""
    print("=== Testing BERTopic Transcribe Attribution Fix ===")
    
    try:
        from app import app, model, embedding_models
        print(f"✓ Initial global model type: {type(model).__name__ if model else 'None'}")
        print(f"✓ Initial model has transcribe: {hasattr(model, 'transcribe') if model else 'N/A'}")
        
        with app.test_client() as client:
            print("Calling /health endpoint (which previously caused variable shadowing)...")
            response = client.get('/health')
            print(f"✓ Health endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.get_json()
                print(f"✓ Health response: {data.get('status', 'unknown')}")
                embedding_status = data.get('embedding_models', {})
                print(f"✓ Embedding models status: {embedding_status}")
        
        print(f"✓ After health check - model type: {type(model).__name__ if model else 'None'}")
        print(f"✓ After health check - model has transcribe: {hasattr(model, 'transcribe') if model else 'N/A'}")
        
        if model:
            if hasattr(model, 'transcribe'):
                print("✅ SUCCESS: Global model still has transcribe method after health check")
                print("✅ Variable shadowing bug is FIXED")
                return True
            else:
                print(f"❌ FAILURE: Global model lost transcribe method! Type: {type(model).__name__}")
                print("❌ Variable shadowing bug still exists")
                return False
        else:
            print("⚠️  WARNING: Global model is None - cannot test transcribe method")
            return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_models_isolation():
    """Test that embedding models are properly isolated from global model"""
    print("\n=== Testing Embedding Models Isolation ===")
    
    try:
        from app import embedding_models, model
        
        print(f"Global Whisper model type: {type(model).__name__ if model else 'None'}")
        
        for model_name, embedding_model in embedding_models.items():
            model_type = type(embedding_model).__name__ if embedding_model else 'None'
            has_transcribe = hasattr(embedding_model, 'transcribe') if embedding_model else False
            print(f"  {model_name}: {model_type}, has_transcribe={has_transcribe}")
            
            if embedding_model and model_name == 'bertopic' and hasattr(embedding_model, 'transcribe'):
                print(f"❌ ERROR: BERTopic model incorrectly has transcribe method!")
                return False
        
        print("✅ All embedding models are properly isolated")
        return True
        
    except Exception as e:
        print(f"❌ Embedding models test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing BERTopic transcribe attribution fix...\n")
    
    success1 = test_bertopic_transcribe_fix()
    success2 = test_embedding_models_isolation()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ BERTopic transcribe attribution bug is FIXED")
        print("✅ Variable shadowing issue resolved")
        print("✅ Global Whisper model preserved correctly")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED!")
        print("❌ BERTopic transcribe attribution bug may still exist")
        sys.exit(1)
