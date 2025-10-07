#!/usr/bin/env python3
"""
Test the fixed XTTS-v2 implementation
"""

import os
import sys

# Set environment variables for license acceptance
os.environ['COQUI_TTS_LICENSE_ACCEPTED'] = 'true'
os.environ['TTS_LICENSE_ACCEPTED'] = 'true'
os.environ['TTS_LICENSE_ACCEPTED_CPML'] = 'true'
os.environ['TTS_HOME'] = '/app/.cache/tts'

try:
    from TTS.api import TTS
    import tempfile
    import wave
    import struct
    
    print("✅ TTS imported successfully")
    
    # Initialize XTTS-v2
    print("🔄 Initializing XTTS-v2...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
    print("✅ XTTS-v2 initialized successfully")
    
    # Test text
    test_text = "Hello, this is a test of the Coqui XTTS version 2 model."
    test_language = "en"
    
    print(f"🔄 Testing TTS with text: '{test_text}'")
    
    # Try Method 1: Without speaker_wav
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tts.tts_to_file(
                text=test_text,
                language=test_language,
                file_path=tmp.name
            )
            if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                print(f"✅ Method 1 SUCCESS: Generated {os.path.getsize(tmp.name)} bytes")
                os.unlink(tmp.name)
                sys.exit(0)
    except TypeError as e:
        print(f"⚠️  Method 1 failed (expected): {e}")
        
        # Try Method 2: With speaker_wav reference
        if "speaker_wav" in str(e) or "missing" in str(e).lower():
            print("🔄 Creating reference speaker WAV...")
            
            reference_wav = '/tmp/reference_speaker.wav'
            with wave.open(reference_wav, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                for _ in range(22050):
                    wav_file.writeframes(struct.pack('<h', 100))
            
            print("✅ Reference WAV created")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tts.tts_to_file(
                    text=test_text,
                    speaker_wav=reference_wav,
                    language=test_language,
                    file_path=tmp.name
                )
                if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                    print(f"✅ Method 2 SUCCESS: Generated {os.path.getsize(tmp.name)} bytes")
                    os.unlink(tmp.name)
                    os.unlink(reference_wav)
                    sys.exit(0)
        else:
            raise e
    
    print("❌ Both methods failed")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
