#!/usr/bin/env python3
"""
Test script to load XTTS-v2 directly from HuggingFace
"""

import os
import torch
from transformers import AutoProcessor, AutoModel

def test_xtts_direct():
    """Test loading XTTS-v2 directly from HuggingFace"""
    
    print("🔄 Testing direct XTTS-v2 loading from HuggingFace...")
    
    try:
        # Set environment variables
        os.environ['COQUI_TTS_LICENSE_ACCEPTED'] = 'true'
        os.environ['TTS_LICENSE_ACCEPTED'] = 'true'
        os.environ['TTS_LICENSE_ACCEPTED_CPML'] = 'true'
        
        # Try to load the model directly from HuggingFace
        model_name = "coqui/XTTS-v2"
        
        print(f"Loading model: {model_name}")
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print("✅ XTTS-v2 loaded successfully from HuggingFace!")
        
        # Test basic functionality
        print("Testing model properties...")
        print(f"Model type: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading XTTS-v2: {e}")
        return False

if __name__ == "__main__":
    success = test_xtts_direct()
    if success:
        print("🎉 XTTS-v2 direct loading works!")
    else:
        print("❌ XTTS-v2 direct loading failed")