#!/usr/bin/env python3

import sys
import os
import tempfile
import wave
import struct
import shutil
import subprocess

def test_ffmpeg_access_methods():
    """Test different methods for ffmpeg to access audio files"""
    print("=== Testing FFmpeg Access Methods ===")
    
    if os.name == 'nt':
        storage_dir = r"C:\Ankit"
    else:
        storage_dir = os.path.join(os.getcwd(), "Ankit")
    
    os.makedirs(storage_dir, exist_ok=True)
    test_audio_path = os.path.join(storage_dir, "test_large_audio.wav")
    
    print(f"Creating test audio file: {test_audio_path}")
    with wave.open(test_audio_path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        
        duration = 30.0
        frames = int(duration * 16000)
        
        import math
        tone_data = []
        for i in range(frames):
            sample = int(16000 * math.sin(2 * math.pi * 440 * i / 16000))
            tone_data.append(sample)
        
        audio_data = struct.pack('<' + ('h' * frames), *tone_data)
        wav_file.writeframes(audio_data)
    
    print(f"Test file created: {os.path.getsize(test_audio_path)} bytes")
    
    methods = [
        ("Direct path", test_audio_path),
        ("Absolute path", os.path.abspath(test_audio_path)),
    ]
    
    if os.name == 'nt':
        methods.append(("Forward slash path", os.path.abspath(test_audio_path).replace('\\', '/')))
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"ffmpeg_test_{os.getpid()}.wav")
    shutil.copy2(test_audio_path, temp_path)
    methods.append(("Temp file", temp_path))
    
    success_count = 0
    for method_name, file_path in methods:
        print(f"\nTesting {method_name}: {file_path}")
        
        cmd = [
            "ffmpeg", "-nostdin", "-threads", "0", "-i", file_path,
            "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
            print(f"  ✓ {method_name} successful ({len(result.stdout)} bytes output)")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {method_name} failed: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            print(f"  ❌ {method_name} timed out")
        except Exception as e:
            print(f"  ❌ {method_name} error: {e}")
    
    try:
        os.remove(test_audio_path)
        os.remove(temp_path)
    except:
        pass
    
    print(f"\nFFmpeg access test results: {success_count}/{len(methods)} methods successful")
    return success_count > 0

def test_whisper_with_fallback_approaches():
    """Test Whisper transcription with the same fallback approaches as the Flask app"""
    print("\n=== Testing Whisper Transcription Fallback Approaches ===")
    
    try:
        from app import app, model
        
        print(f"✓ Flask app imported successfully")
        print(f"✓ Whisper model loaded: {type(model).__name__ if model else 'None'}")
        
        if not model:
            print("❌ Whisper model not loaded - cannot test transcription")
            return False
        
        if os.name == 'nt':
            storage_dir = r"C:\Ankit"
        else:
            storage_dir = os.path.join(os.getcwd(), "Ankit")
        
        os.makedirs(storage_dir, exist_ok=True)
        test_audio_path = os.path.join(storage_dir, "test_whisper_fallback.wav")
        
        print(f"Creating test audio file: {test_audio_path}")
        with wave.open(test_audio_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            
            duration = 5.0
            frames = int(duration * 16000)
            
            import math
            tone_data = []
            for i in range(frames):
                sample = int(8000 * math.sin(2 * math.pi * 440 * i / 16000))
                tone_data.append(sample)
            
            audio_data = struct.pack('<' + ('h' * frames), *tone_data)
            wav_file.writeframes(audio_data)
        
        print(f"Test file created: {os.path.getsize(test_audio_path)} bytes")
        
        transcription_attempts = []
        transcription_success = False
        
        try:
            print("Attempt 1: Direct path transcription")
            result = model.transcribe(test_audio_path)
            print(f"✓ Direct path successful: '{result.get('text', 'N/A')}'")
            transcription_success = True
        except Exception as e:
            transcription_attempts.append(f"Direct path failed: {str(e)}")
            print(f"❌ Direct path failed: {e}")
            
            try:
                print("Attempt 2: Absolute path normalization")
                abs_path = os.path.abspath(test_audio_path)
                result = model.transcribe(abs_path)
                print(f"✓ Absolute path successful: '{result.get('text', 'N/A')}'")
                transcription_success = True
            except Exception as e2:
                transcription_attempts.append(f"Absolute path failed: {str(e2)}")
                print(f"❌ Absolute path failed: {e2}")
                
                if not transcription_success and os.name == 'nt':
                    try:
                        print("Attempt 3: Windows forward slash path")
                        forward_slash_path = abs_path.replace('\\', '/')
                        result = model.transcribe(forward_slash_path)
                        print(f"✓ Forward slash path successful: '{result.get('text', 'N/A')}'")
                        transcription_success = True
                    except Exception as e3:
                        transcription_attempts.append(f"Forward slash path failed: {str(e3)}")
                        print(f"❌ Forward slash path failed: {e3}")
                
                if not transcription_success:
                    try:
                        print("Attempt 4: System temp file copy")
                        import tempfile
                        import shutil
                        
                        temp_dir = tempfile.gettempdir()
                        simple_temp_path = os.path.join(temp_dir, f"whisper_test_{os.getpid()}.wav")
                        
                        with open(test_audio_path, 'rb') as src, open(simple_temp_path, 'wb') as dst:
                            shutil.copyfileobj(src, dst, 1024*1024)
                        
                        result = model.transcribe(simple_temp_path)
                        print(f"✓ Temp file successful: '{result.get('text', 'N/A')}'")
                        transcription_success = True
                        
                        try:
                            os.unlink(simple_temp_path)
                        except:
                            pass
                            
                    except Exception as e4:
                        transcription_attempts.append(f"Temp file failed: {str(e4)}")
                        print(f"❌ Temp file failed: {e4}")
        
        try:
            os.remove(test_audio_path)
        except:
            pass
        
        if transcription_success:
            print("✅ At least one transcription approach succeeded")
            return True
        else:
            print(f"❌ All transcription approaches failed: {'; '.join(transcription_attempts)}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing FFmpeg access methods and Whisper transcription fallbacks...\n")
    
    ffmpeg_success = test_ffmpeg_access_methods()
    whisper_success = test_whisper_with_fallback_approaches()
    
    if ffmpeg_success and whisper_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ FFmpeg can access audio files using multiple methods")
        print("✅ Whisper transcription works with fallback approaches")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        if not ffmpeg_success:
            print("❌ FFmpeg access issues detected")
        if not whisper_success:
            print("❌ Whisper transcription issues detected")
        sys.exit(1)
