#!/usr/bin/env python3
"""
Interactive script to set up Coqui XTTS-v2 with license acceptance
"""

import os
import sys
import subprocess

def setup_coqui_tts():
    """Set up Coqui XTTS-v2 with proper license acceptance"""
    
    print("🔧 Setting up Coqui XTTS-v2...")
    
    # Set environment variables
    os.environ['COQUI_TTS_LICENSE_ACCEPTED'] = 'true'
    os.environ['TTS_LICENSE_ACCEPTED'] = 'true'
    os.environ['TTS_LICENSE_ACCEPTED_CPML'] = 'true'
    
    # Create cache directories
    cache_dir = '/app/.cache/tts'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create license acceptance files
    license_files = [
        f'{cache_dir}/.tos_agreed',
        f'{cache_dir}/tts_models/multilingual/multi-dataset/xtts_v2/.tos_agreed',
        f'{cache_dir}/coqui/XTTS-v2/.tos_agreed'
    ]
    
    for license_file in license_files:
        os.makedirs(os.path.dirname(license_file), exist_ok=True)
        with open(license_file, 'w') as f:
            f.write('true')
        print(f"✅ Created license file: {license_file}")
    
    # Try to initialize TTS with interactive license acceptance
    print("\n🔄 Initializing Coqui XTTS-v2...")
    print("This will prompt for license acceptance. Please type 'y' when prompted.")
    
    try:
        # Use subprocess to handle interactive input
        result = subprocess.run([
            'python3', '-c', '''
import os
os.environ["COQUI_TTS_LICENSE_ACCEPTED"] = "true"
os.environ["TTS_LICENSE_ACCEPTED"] = "true"
os.environ["TTS_LICENSE_ACCEPTED_CPML"] = "true"

from TTS.api import TTS
print("Initializing XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
print("XTTS-v2 initialized successfully!")

# Test audio generation
tts.tts_to_file(text="Hello, this is Coqui XTTS-v2", file_path="/tmp/test_coqui.wav")
print("Audio generated successfully!")
'''
        ], input='y\n', text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Coqui XTTS-v2 setup completed successfully!")
            return True
        else:
            print(f"❌ Setup failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Setup timed out")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_coqui_tts()
    if success:
        print("\n🎉 Coqui XTTS-v2 is now ready to use!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
    sys.exit(0 if success else 1)