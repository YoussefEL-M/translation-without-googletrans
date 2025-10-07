#!/usr/bin/env python3
"""
Script to accept Coqui TTS license terms
"""

import os
import sys

# Set environment variables for license acceptance
os.environ['COQUI_TTS_LICENSE_ACCEPTED'] = 'true'
os.environ['TTS_LICENSE_ACCEPTED'] = 'true'
os.environ['TTS_LICENSE_ACCEPTED_CPML'] = 'true'

def accept_license():
    """Accept the Coqui TTS license terms"""
    try:
        print("Accepting Coqui TTS license terms...")
        
        # Create the license acceptance file
        license_file = '/app/.cache/tts/.tos_agreed'
        os.makedirs(os.path.dirname(license_file), exist_ok=True)
        
        with open(license_file, 'w') as f:
            f.write('true')
        print(f"License acceptance file created at: {license_file}")
        
        # Also create a marker file in the model directory
        model_license_file = '/app/.cache/tts/tts_models/multilingual/multi-dataset/xtts_v2/.tos_agreed'
        os.makedirs(os.path.dirname(model_license_file), exist_ok=True)
        
        with open(model_license_file, 'w') as f:
            f.write('true')
        print(f"Model license acceptance file created at: {model_license_file}")
        
        # Test TTS initialization
        print("Testing TTS initialization...")
        from TTS.api import TTS
        
        # Initialize with the specific model
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        print("✅ Coqui XTTS-v2 initialized successfully!")
        
        # Test audio generation
        print("Testing audio generation...")
        tts.tts_to_file(text="Hello, this is Coqui XTTS-v2", file_path="/tmp/test_coqui.wav")
        
        if os.path.exists("/tmp/test_coqui.wav"):
            size = os.path.getsize("/tmp/test_coqui.wav")
            print(f"✅ Audio generated successfully: {size} bytes")
            return True
        else:
            print("❌ Audio generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = accept_license()
    sys.exit(0 if success else 1)