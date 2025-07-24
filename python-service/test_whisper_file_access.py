#!/usr/bin/env python3

import sys
import os
import tempfile
import wave
import struct
import shutil

def test_whisper_file_access_debug():
    """Test Whisper file access with different path approaches"""
    print("=== Testing Whisper File Access Debug ===")
    
    try:
        from app import app, model
        
        print(f"✓ Flask app imported successfully")
        print(f"✓ Whisper model loaded: {type(model).__name__ if model else 'None'}")
        
        if not model:
            print("❌ Whisper model not loaded - cannot test transcription")
            return False
        
        if os.name == 'nt':  # Windows
            storage_dir = r"C:\Ankit"
        else:  # Linux/Unix - create equivalent directory structure
            storage_dir = os.path.join(os.getcwd(), "Ankit")
        
        test_filename = "test_large_audio.wav"
        
        print(f"\n1. Creating test environment...")
        print(f"   Storage directory: {storage_dir}")
        print(f"   Operating system: {os.name}")
        
        try:
            os.makedirs(storage_dir, exist_ok=True)
            print(f"   ✓ Directory created/exists: {os.path.exists(storage_dir)}")
        except Exception as e:
            print(f"   ❌ Directory creation failed: {e}")
            return False
        
        audio_temp_path = os.path.join(storage_dir, test_filename)
        print(f"   Audio file path: {audio_temp_path}")
        
        print(f"\n2. Creating large test audio file...")
        try:
            with wave.open(audio_temp_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                
                duration = 10.0
                frames = int(duration * 16000)
                
                import math
                tone_data = []
                for i in range(frames):
                    sample = int(16000 * math.sin(2 * math.pi * 440 * i / 16000))
                    tone_data.append(sample)
                
                audio_data = struct.pack('<' + ('h' * frames), *tone_data)
                wav_file.writeframes(audio_data)
            
            print(f"   ✓ Audio file created: {os.path.exists(audio_temp_path)}")
            print(f"   ✓ Audio file size: {os.path.getsize(audio_temp_path)} bytes")
            
        except Exception as e:
            print(f"   ❌ Audio file creation failed: {e}")
            return False
        
        print(f"\n3. Testing file access methods...")
        
        try:
            with open(audio_temp_path, 'rb') as f:
                first_bytes = f.read(1024)
            print(f"   ✓ File is readable by Python ({len(first_bytes)} bytes read)")
        except Exception as e:
            print(f"   ❌ File is not readable by Python: {e}")
            return False
        
        try:
            import stat
            file_stat = os.stat(audio_temp_path)
            print(f"   ✓ File permissions: {stat.filemode(file_stat.st_mode)}")
        except Exception as e:
            print(f"   ⚠️  Could not get file stats: {e}")
        
        print(f"\n4. Testing Whisper transcription approaches...")
        
        success = False
        
        try:
            print(f"   Approach 1: Direct path")
            print(f"   Path: {audio_temp_path}")
            result = model.transcribe(audio_temp_path)
            print(f"   ✓ Direct path transcription successful!")
            print(f"   ✓ Result: '{result.get('text', 'N/A')}'")
            success = True
        except Exception as e:
            print(f"   ❌ Direct path failed: {e}")
            
            try:
                print(f"   Approach 2: Absolute path")
                abs_path = os.path.abspath(audio_temp_path)
                print(f"   Absolute path: {abs_path}")
                result = model.transcribe(abs_path)
                print(f"   ✓ Absolute path transcription successful!")
                print(f"   ✓ Result: '{result.get('text', 'N/A')}'")
                success = True
            except Exception as e2:
                print(f"   ❌ Absolute path failed: {e2}")
                
                try:
                    print(f"   Approach 3: Simple temp file copy")
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        simple_temp_path = temp_file.name
                    
                    print(f"   Copying to: {simple_temp_path}")
                    shutil.copy2(audio_temp_path, simple_temp_path)
                    
                    result = model.transcribe(simple_temp_path)
                    print(f"   ✓ Simple temp file transcription successful!")
                    print(f"   ✓ Result: '{result.get('text', 'N/A')}'")
                    success = True
                    
                    try:
                        os.unlink(simple_temp_path)
                    except:
                        pass
                        
                except Exception as e3:
                    print(f"   ❌ Simple temp file failed: {e3}")
        
        try:
            os.remove(audio_temp_path)
            print(f"   ✓ Cleanup completed")
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Whisper file access with different approaches...\n")
    
    success = test_whisper_file_access_debug()
    
    if success:
        print("\n🎉 WHISPER FILE ACCESS TEST PASSED!")
        print("✅ Found working approach for Whisper transcription")
        sys.exit(0)
    else:
        print("\n❌ WHISPER FILE ACCESS TEST FAILED!")
        print("❌ All transcription approaches failed")
        sys.exit(1)
