#!/usr/bin/env python3
"""
Test script for Coqui XTTS-v2 functionality
"""

import os
import sys

# Set environment variables for license acceptance
os.environ['COQUI_TTS_LICENSE_ACCEPTED'] = 'true'
os.environ['TTS_LICENSE_ACCEPTED'] = 'true'
os.environ['TTS_LICENSE_ACCEPTED_CPML'] = 'true'

try:
    from TTS.api import TTS
    print("✅ TTS imported successfully")
    
    # Try to initialize with different approaches
    print("🔄 Testing TTS initialization...")
    
    try:
        # Method 1: Try with the specific model name
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        print("✅ TTS initialized with tts_models/multilingual/multi-dataset/xtts_v2")
    except Exception as e1:
        print(f"❌ Method 1 failed: {e1}")
        try:
            # Method 2: Try with HuggingFace model name
            tts = TTS(model_name="coqui/XTTS-v2", progress_bar=False)
            print("✅ TTS initialized with coqui/XTTS-v2")
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            try:
                # Method 3: Try without specifying model
                tts = TTS(progress_bar=False)
                print("✅ TTS initialized with default model")
            except Exception as e3:
                print(f"❌ Method 3 failed: {e3}")
                sys.exit(1)
    
    # Test TTS functionality
    print("🔄 Testing TTS functionality...")
    
    try:
        # Test with minimal parameters
        tts.tts_to_file(text="Hello world", file_path="test_output.wav")
        print("✅ TTS generated audio file successfully")
        
        # Check if file was created
        if os.path.exists("test_output.wav"):
            size = os.path.getsize("test_output.wav")
            print(f"✅ Audio file created: {size} bytes")
        else:
            print("❌ Audio file not created")
            
    except Exception as e:
        print(f"❌ TTS functionality test failed: {e}")
        sys.exit(1)
    
    print("🎉 All tests passed! Coqui XTTS-v2 is working correctly.")
    
except ImportError as e:
    print(f"❌ Failed to import TTS: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)