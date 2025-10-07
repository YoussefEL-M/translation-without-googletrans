# GPU Setup for XTTS-v2 - Issue & Solution

## Current Status

❌ **GPU is NOT available in the container**
- Container shows: `CUDA available: False`
- Your system has: NVIDIA RTX A2000 12GB ✅
- Issue: Podman container doesn't have GPU access

## Why GPU Isn't Working

Podman needs the **NVIDIA Container Toolkit** to pass GPU to containers. The current setup:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia  # This doesn't work without nvidia-container-toolkit
```

## Current Workaround (Active)

Since GPU isn't available, I've switched to **CPU-friendly models**:
- ✅ **YourTTS** - Multilingual, lightweight, works on CPU
- ✅ **Tacotron2** - Good quality English, CPU-friendly
- ❌ **XTTS-v2** - Disabled (needs GPU or 12GB+ RAM)

## How to Enable GPU (Optional)

If you want XTTS-v2 with GPU, follow these steps:

### Step 1: Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure for Podman
sudo nvidia-ctk runtime configure --runtime=cdi --config=/etc/nvidia-container-runtime/config.toml
```

### Step 2: Update Podman Compose

```yaml
# In podman-compose.yml, replace the deploy section with:
services:
  translation-pwa:
    devices:
      - nvidia.com/gpu=all  # CDI device specification
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Step 3: Re-enable XTTS-v2

```python
# In backend/app.py, uncomment XTTS-v2:
models_to_try = [
    ("tts_models/multilingual/multi-dataset/xtts_v2", "XTTS-v2 (GPU)"),
    ("tts_models/multilingual/multi-dataset/your_tts", "YourTTS (fallback)"),
]
```

### Step 4: Increase Container Memory

```yaml
# In podman-compose.yml:
deploy:
  resources:
    limits:
      memory: 12G  # For XTTS-v2
```

### Step 5: Rebuild

```bash
./start-translation-pwa-podman.sh
```

## Verification

After setup, check if GPU works:
```bash
podman exec translation-pwa python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should show: CUDA: True
```

## Alternative: Use Docker Instead

If Podman GPU setup is too complex, Docker has better NVIDIA support:

```bash
# Install Docker + NVIDIA runtime
sudo apt-get install docker.io nvidia-docker2
sudo systemctl restart docker

# Convert to docker-compose.yml
# Add runtime: nvidia to services

# Run with Docker
docker-compose up -d
```

## Current Model Quality Comparison

| Model | Quality | Speed | GPU Needed | RAM Needed | Status |
|-------|---------|-------|------------|------------|--------|
| **YourTTS** (active) | ⭐⭐⭐⭐ | Fast | ❌ No | 2-3GB | ✅ Working |
| **Tacotron2** (active) | ⭐⭐⭐⭐ | Fast | ❌ No | 1-2GB | ✅ Working |
| **XTTS-v2** (disabled) | ⭐⭐⭐⭐⭐ | Medium | ✅ Yes | 8-10GB | ❌ No GPU |

## Recommendation

**For now: Use YourTTS (current setup)**
- ✅ Good quality multilingual TTS
- ✅ Works without GPU
- ✅ 1100+ languages supported
- ✅ Runs smoothly in 6GB container

**For best quality: Enable GPU for XTTS-v2**
- Requires NVIDIA Container Toolkit setup
- Worth it if you need the absolute best TTS quality
- Your hardware (RTX A2000) is perfect for it

## Summary

- **Current:** YourTTS on CPU (good quality, working ✅)
- **Optional:** XTTS-v2 on GPU (best quality, needs setup 🔧)
- **Your choice:** Current setup works well, GPU is optional upgrade

The app is working with good TTS quality now - GPU is only needed if you want the absolute best!
