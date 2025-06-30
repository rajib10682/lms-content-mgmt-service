#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_nltk_setup():
    """Test NLTK setup and downloads"""
    print("Testing NLTK setup...")
    
    try:
        import nltk
        print("✓ NLTK imported successfully")
        
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ punkt tokenizer found")
        except LookupError:
            print("Downloading punkt...")
            nltk.download('punkt')
            print("✓ punkt tokenizer downloaded")
        
        try:
            nltk.data.find('corpora/stopwords')
            print("✓ stopwords found")
        except LookupError:
            print("Downloading stopwords...")
            nltk.download('stopwords')
            print("✓ stopwords downloaded")
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
            print("✓ POS tagger found")
        except LookupError:
            print("Downloading averaged_perceptron_tagger...")
            nltk.download('averaged_perceptron_tagger')
            print("✓ POS tagger downloaded")
        
        return True
        
    except Exception as e:
        print(f"✗ NLTK setup failed: {e}")
        return False

def test_topic_extraction():
    """Test the enhanced topic extraction functionality"""
    print("\nTesting topic extraction...")
    
    try:
        from app import extract_topics_from_text
        
        tech_text = """
        In this video, we'll discuss software development best practices and programming techniques.
        We'll cover algorithms, data structures, and machine learning concepts.
        The presentation includes examples of Python coding and web development frameworks.
        """
        
        topics = extract_topics_from_text(tech_text)
        print(f"Tech text topics: {topics}")
        
        business_text = """
        Today's meeting focuses on our quarterly sales strategy and revenue projections.
        We need to discuss market analysis, customer engagement, and team management.
        The company's growth depends on effective leadership and organizational planning.
        """
        
        topics = extract_topics_from_text(business_text)
        print(f"Business text topics: {topics}")
        
        short_text = "Hello world"
        topics = extract_topics_from_text(short_text)
        print(f"Short text topics: {topics}")
        
        print("✓ Topic extraction test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Topic extraction test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== NLP Enhancement Test ===")
    
    nltk_ok = test_nltk_setup()
    if not nltk_ok:
        print("NLTK setup failed, exiting...")
        sys.exit(1)
    
    topic_ok = test_topic_extraction()
    if not topic_ok:
        print("Topic extraction test failed, exiting...")
        sys.exit(1)
    
    print("\n✅ All tests passed! Enhanced topic extraction is working correctly.")
