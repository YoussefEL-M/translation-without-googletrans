# TTS Models - Open Source & Local Hosting

## Current Setup: Fully Open Source & Self-Hosted

All TTS models used in this application are:
- ✅ **100% Open Source** (Mozilla Public License 2.0 / Apache 2.0)
- ✅ **Run Locally** - No external API calls
- ✅ **Self-Hosted** - Models download once and run on your server
- ✅ **No Third-Party Dependencies** - Everything runs in your container

## Models Being Used (in order of preference)

### 1. YourTTS (Primary - Multilingual)
- **License:** Mozilla Public License 2.0 (Open Source)
- **Repository:** https://github.com/coqui-ai/TTS
- **Type:** Multilingual Text-to-Speech
- **Languages:** 1100+ languages supported
- **Local:** Yes, downloads model files once to `/app/.cache/tts/`
- **Size:** ~200MB
- **Quality:** Good quality, lightweight

### 2. Tacotron2-DDC (Fallback - English)
- **License:** Mozilla Public License 2.0 (Open Source)
- **Repository:** https://github.com/coqui-ai/TTS
- **Type:** English-only TTS
- **Local:** Yes, runs entirely locally
- **Size:** ~60MB
- **Quality:** High quality for English

### 3. FastPitch (Second Fallback - English)
- **License:** Mozilla Public License 2.0 (Open Source)
- **Repository:** https://github.com/coqui-ai/TTS
- **Type:** English-only TTS (faster inference)
- **Local:** Yes, runs entirely locally
- **Size:** ~50MB
- **Quality:** Good quality, very fast

## Why XTTS-v2 Was Removed

XTTS-v2 is also open source, but:
- ❌ **Too Resource Intensive** - Requires 8GB+ RAM to load
- ❌ **Crashes in Container** - OOM (Out of Memory) errors
- ❌ **Voice Cloning Design** - Requires speaker reference audio
- ⚠️ **Size:** ~2GB model files

## Coqui TTS Library

The entire TTS system uses **Coqui TTS**:
- **License:** Mozilla Public License 2.0
- **GitHub:** https://github.com/coqui-ai/TTS
- **Created by:** Mozilla Foundation / Coqui Team
- **Status:** Open Source, Community Maintained
- **Privacy:** 100% local, no data sent to external servers

## Danish TTS

For Danish language, we use:
- **DR TTS API** (Danmarks Radio public service)
- **Note:** This IS an external API call to DR's servers
- **Fallback:** If you want 100% local Danish, we can add a local Danish model

## How Models are Downloaded

1. On first use, Coqui TTS downloads model files from:
   - HuggingFace Model Hub (open source model repository)
   - Coqui's public CDN
2. Models are cached locally in `/app/.cache/tts/`
3. Subsequent uses are 100% offline/local
4. No telemetry or tracking

## Making Danish Fully Local (Optional)

If you want to remove the DR TTS API dependency for Danish:

```python
# Option 1: Use YourTTS for Danish (already supports it)
# No changes needed - YourTTS handles Danish

# Option 2: Add a dedicated Danish model
# Add to models_to_try:
("tts_models/da/cv/vits", "Danish VITS (local)")
```

## Verification Commands

Check that models are local:
```bash
# See cached models
ls -lh /app/.cache/tts/tts_models/

# Verify no external calls during TTS
podman exec translation-pwa tcpdump -i any -n port 443  # Should show no HTTPS traffic during TTS

# Check model source code
podman exec translation-pwa python3 -c "from TTS.api import TTS; print(TTS.__file__)"
```

## Summary

✅ **Everything is open source**  
✅ **Everything runs locally after initial download**  
✅ **No vendor lock-in**  
✅ **No external API dependencies** (except optional DR TTS for Danish)  
✅ **Full control over your infrastructure**

Your translation PWA is completely self-hosted and privacy-friendly!
