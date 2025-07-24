#!/usr/bin/env python3

import sys
import os
import tempfile
import wave
import struct

def test_transcription_pipeline_debug():
    """Debug the transcription pipeline step by step"""
    print("=== Testing Transcription Pipeline Debug ===")
    
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
        test_filename = "test_video.mp4"
        
        print(f"\n1. Testing directory and file creation...")
        print(f"   Storage directory: {storage_dir}")
        
        try:
            os.makedirs(storage_dir, exist_ok=True)
            print(f"   ✓ Directory created/exists: {os.path.exists(storage_dir)}")
        except Exception as e:
            print(f"   ❌ Directory creation failed: {e}")
            return False
        
        audio_temp_path = os.path.join(storage_dir, f"audio_{test_filename.rsplit('.', 1)[0]}.wav")
        print(f"   Audio file path: {audio_temp_path}")
        print(f"   Normalized audio path: {os.path.normpath(audio_temp_path)}")
        
        print(f"\n2. Testing audio file creation...")
        try:
            with wave.open(audio_temp_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                
                duration = 1.0
                frames = int(duration * 16000)
                silence = struct.pack('<' + ('h' * frames), *([0] * frames))
                wav_file.writeframes(silence)
            
            print(f"   ✓ Audio file created: {os.path.exists(audio_temp_path)}")
            print(f"   ✓ Audio file size: {os.path.getsize(audio_temp_path)} bytes")
            
        except Exception as e:
            print(f"   ❌ Audio file creation failed: {e}")
            return False
        
        print(f"\n3. Testing Whisper transcription...")
        try:
            print(f"   Testing model.transcribe() call...")
            print(f"   Original audio path: {audio_temp_path}")
            
            normalized_audio_path = os.path.normpath(audio_temp_path)
            if os.name == 'posix' and normalized_audio_path.startswith('C:'):
                normalized_audio_path = normalized_audio_path.replace('C:', '').replace('\\', '/')
                if normalized_audio_path.startswith('/'):
                    normalized_audio_path = '.' + normalized_audio_path
                else:
                    normalized_audio_path = './' + normalized_audio_path
            
            print(f"   Normalized audio path for transcription: {normalized_audio_path}")
            print(f"   Audio file exists at normalized path: {os.path.exists(normalized_audio_path)}")
            
            result = model.transcribe(normalized_audio_path)
            print(f"   ✓ Transcription successful!")
            print(f"   ✓ Result type: {type(result)}")
            print(f"   ✓ Transcribed text: '{result.get('text', 'N/A')}'")
            
            os.remove(audio_temp_path)
            print(f"   ✓ Cleanup completed")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Transcription test failed: {e}")
            print(f"   ❌ Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            if os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)
            
            return False
        
    except Exception as e:
        print(f"❌ Debug test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Debugging transcription pipeline...\n")
    
    success = test_transcription_pipeline_debug()
    
    if success:
        print("\n🎉 TRANSCRIPTION DEBUG TEST PASSED!")
        print("✅ Transcription pipeline works correctly")
        sys.exit(0)
    else:
        print("\n❌ TRANSCRIPTION DEBUG TEST FAILED!")
        print("❌ Issues detected in transcription pipeline")
        sys.exit(1)
