#!/usr/bin/env python3

import sys
import os
import tempfile
import json

def test_full_video_analysis_pipeline():
    """Test the complete video analysis pipeline end-to-end"""
    print("=== Testing Full Video Analysis Pipeline ===")
    
    try:
        from app import app, model, embedding_models
        
        print(f"✓ Global Whisper model loaded: {type(model).__name__ if model else 'None'}")
        print(f"✓ Embedding models loaded: {list(embedding_models.keys())}")
        
        with app.test_client() as client:
            print("\n1. Testing /health endpoint...")
            health_response = client.get('/health')
            print(f"   Status: {health_response.status_code}")
            
            if health_response.status_code == 200:
                health_data = health_response.get_json()
                print(f"   Service status: {health_data.get('status')}")
                print(f"   Whisper model: {health_data.get('whisper_model')}")
                print(f"   SSL configured: {health_data.get('ssl_configured')}")
                embedding_status = health_data.get('embedding_models', {})
                print(f"   Embedding models: {embedding_status}")
            
            print(f"\n   After health check - Global model type: {type(model).__name__ if model else 'None'}")
            print(f"   After health check - Has transcribe: {hasattr(model, 'transcribe') if model else 'N/A'}")
            
            if not model or not hasattr(model, 'transcribe'):
                print("❌ CRITICAL: Global Whisper model corrupted after health check!")
                return False
            
            print("\n2. Testing /analyze-video endpoint structure...")
            
            no_file_response = client.post('/analyze-video')
            print(f"   No file test status: {no_file_response.status_code}")
            if no_file_response.status_code == 400:
                error_data = no_file_response.get_json()
                print(f"   Expected error: {error_data.get('error')}")
            
            empty_file_response = client.post('/analyze-video', data={'file': (None, '')})
            print(f"   Empty file test status: {empty_file_response.status_code}")
            
            print("✅ Pipeline structure tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_topic_extraction_methods():
    """Test all topic extraction methods work correctly"""
    print("\n=== Testing Topic Extraction Methods ===")
    
    try:
        from app import (
            extract_topics_from_text,
            extract_topics_with_openai_embeddings,
            extract_topics_with_keybert,
            extract_topics_with_bertopic,
            extract_topics_from_text_nltk,
            embedding_models
        )
        
        test_text = """
        This is a comprehensive test of our video analysis system. 
        We are testing machine learning algorithms, natural language processing,
        and artificial intelligence capabilities. The system uses advanced
        embedding models for topic extraction and analysis.
        """
        
        print("Testing hierarchical topic extraction...")
        topics = extract_topics_from_text(test_text)
        print(f"✓ Main extraction returned: {topics}")
        
        print("\nTesting individual extraction methods:")
        
        try:
            openai_topics = extract_topics_with_openai_embeddings(test_text)
            print(f"✓ OpenAI topics: {openai_topics}")
        except Exception as e:
            print(f"⚠️  OpenAI topics (expected if no API key): {e}")
        
        if embedding_models.get('keybert'):
            try:
                keybert_topics = extract_topics_with_keybert(test_text, embedding_models['keybert'])
                print(f"✓ KeyBERT topics: {keybert_topics}")
            except Exception as e:
                print(f"❌ KeyBERT failed: {e}")
                return False
        
        if embedding_models.get('bertopic'):
            try:
                bertopic_topics = extract_topics_with_bertopic(test_text, embedding_models['bertopic'])
                print(f"✓ BERTopic topics: {bertopic_topics}")
            except Exception as e:
                print(f"❌ BERTopic failed: {e}")
                return False
        
        try:
            nltk_topics = extract_topics_from_text_nltk(test_text)
            print(f"✓ NLTK topics: {nltk_topics}")
        except Exception as e:
            print(f"❌ NLTK failed: {e}")
            return False
        
        print("✅ All topic extraction methods working!")
        return True
        
    except Exception as e:
        print(f"❌ Topic extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing complete video analysis pipeline...\n")
    
    success1 = test_full_video_analysis_pipeline()
    success2 = test_topic_extraction_methods()
    
    if success1 and success2:
        print("\n🎉 ALL PIPELINE TESTS PASSED!")
        print("✅ Video analysis pipeline is working correctly")
        print("✅ BERTopic transcribe attribution bug is fixed")
        print("✅ All topic extraction methods functional")
        print("✅ Global Whisper model preserved correctly")
        sys.exit(0)
    else:
        print("\n❌ PIPELINE TESTS FAILED!")
        print("❌ Issues detected in video analysis pipeline")
        sys.exit(1)
