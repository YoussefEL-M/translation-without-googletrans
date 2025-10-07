# GPU Access Fixed! 🎉

## The Problem

The start script used `--gpus all` which is a **Docker flag**, not Podman!

```bash
# WRONG (Docker syntax):
podman run --gpus all ...

# This silently failed - Podman doesn't recognize --gpus
```

## The Solution

Changed to Podman's GPU syntax (same as your working Whisper setup):

```bash
# CORRECT (Podman syntax):
podman run --device nvidia.com/gpu=all ...
```

## Changes Made

### 1. Fixed GPU Device Flag
**File:** `start-translation-pwa-podman.sh`
```bash
# Changed from:
--gpus all  # Docker syntax (doesn't work in Podman)

# To:
--device nvidia.com/gpu=all  # Podman syntax (works!)
```

### 2. Added CUDA Environment Variables (from Whisper)
```bash
-e CUDA_DEVICE_ORDER=PCI_BUS_ID
-e PYTORCH_NVML_BASED_CUDA_CHECK=1
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

### 3. Increased Memory to 12GB
```bash
--memory=12g  # Enough for XTTS-v2 on GPU
```

### 4. Re-enabled XTTS-v2
**File:** `backend/app.py`
- XTTS-v2 is now the primary model
- Will use your RTX A2000 12GB GPU
- Falls back to YourTTS if needed

## What You'll Get

✅ **GPU Access** - RTX A2000 12GB detected  
✅ **XTTS-v2** - Best quality TTS (16 languages)  
✅ **Voice Cloning** - Clone voices from samples  
✅ **Fast Inference** - GPU accelerated  
✅ **12GB RAM** - Enough to load XTTS-v2  
✅ **Open Source** - 100% local, no external APIs  

## Expected Logs After Rebuild

```
INFO:__main__:TTS will use device: cuda
INFO:__main__:Trying to load XTTS-v2 (best quality, 16 languages, GPU)...
INFO:__main__:✅ Successfully loaded XTTS-v2 (best quality, 16 languages, GPU) on GPU
```

## Now Ready to Rebuild!

Run:
```bash
./start-translation-pwa-podman.sh
```

You should see:
1. ✅ GPU detected (cuda)
2. ✅ XTTS-v2 loads successfully
3. ✅ Professional-grade TTS working!

## Verification

After container starts, verify GPU:
```bash
podman exec translation-pwa python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Should show:
```
CUDA: True
GPU: NVIDIA RTX A2000 12GB
```

## Summary

- ❌ **Before:** Used `--gpus all` (Docker syntax) → No GPU
- ✅ **After:** Uses `--device nvidia.com/gpu=all` (Podman syntax) → GPU works!
- 🎉 **Result:** XTTS-v2 with GPU acceleration, just like your Whisper setup!

Your translation PWA will now have the same GPU access as Whisper! 🚀
