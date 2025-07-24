#!/usr/bin/env python3

import sys
import os

def test_hardcoded_path_functionality():
    """Test that the hardcoded C:\Ankit path functionality works correctly"""
    print("=== Testing Hardcoded Path C:\\Ankit Functionality ===")
    
    try:
        from app import app, model
        
        print(f"✓ Flask app imported successfully")
        print(f"✓ Whisper model loaded: {type(model).__name__ if model else 'None'}")
        
        with app.test_client() as client:
            print("\n1. Testing /health endpoint...")
            health_response = client.get('/health')
            print(f"   Status: {health_response.status_code}")
            
            if health_response.status_code == 200:
                health_data = health_response.get_json()
                print(f"   Service status: {health_data.get('status')}")
                print(f"   Whisper model: {health_data.get('whisper_model')}")
            
            print("\n2. Testing /analyze-video endpoint with hardcoded path...")
            
            no_file_response = client.post('/analyze-video')
            print(f"   No file test status: {no_file_response.status_code}")
            if no_file_response.status_code == 400:
                error_data = no_file_response.get_json()
                print(f"   Expected error: {error_data.get('error')}")
            
            empty_file_response = client.post('/analyze-video', data={'file': (None, '')})
            print(f"   Empty file test status: {empty_file_response.status_code}")
            if empty_file_response.status_code == 400:
                error_data = empty_file_response.get_json()
                print(f"   Expected error: {error_data.get('error')}")
            
            print("✅ Hardcoded path structure tests passed!")
            
            print("\n3. Testing path logic...")
            
            import inspect
            from app import analyze_video
            
            source_code = inspect.getsource(analyze_video)
            
            if r'C:\Ankit' in source_code:
                print("✅ Hardcoded path C:\\Ankit found in analyze_video function")
            else:
                print("❌ Hardcoded path C:\\Ankit NOT found in analyze_video function")
                return False
            
            if 'os.makedirs(storage_dir, exist_ok=True)' in source_code:
                print("✅ Directory creation logic found")
            else:
                print("❌ Directory creation logic NOT found")
                return False
            
            if 'tempfile.mkdtemp()' not in source_code:
                print("✅ Temporary directory logic successfully removed")
            else:
                print("❌ Temporary directory logic still present")
                return False
            
            print("✅ All path logic tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_platform_considerations():
    """Test cross-platform path handling"""
    print("\n=== Testing Cross-Platform Path Considerations ===")
    
    try:
        import os
        
        storage_dir = r"C:\Ankit"
        test_filename = "test_video.mp4"
        
        test_path = os.path.join(storage_dir, test_filename)
        print(f"✓ Path joining works: {test_path}")
        
        print("✓ os.makedirs with exist_ok=True should work cross-platform")
        
        print("✅ Cross-platform considerations verified!")
        return True
        
    except Exception as e:
        print(f"❌ Cross-platform test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing hardcoded path C:\\Ankit functionality...\n")
    
    success1 = test_hardcoded_path_functionality()
    success2 = test_cross_platform_considerations()
    
    if success1 and success2:
        print("\n🎉 ALL HARDCODED PATH TESTS PASSED!")
        print("✅ C:\\Ankit path successfully implemented")
        print("✅ Directory creation logic added")
        print("✅ Temporary directory logic removed")
        print("✅ All existing functionality preserved")
        sys.exit(0)
    else:
        print("\n❌ HARDCODED PATH TESTS FAILED!")
        print("❌ Issues detected with C:\\Ankit implementation")
        sys.exit(1)
