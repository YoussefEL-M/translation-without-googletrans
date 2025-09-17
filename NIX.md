# Nix Setup for Translation PWA

This document explains how to use Nix to set up the Translation PWA development environment.

## Quick Start

### Using nix-shell (Recommended)

```bash
# Enter the development environment
nix-shell

# Once inside the shell, set up Python environment
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Start the application
python backend/app.py
```

### Using the startup script

```bash
# Enter the development environment
nix-shell

# Use the provided startup script
./start.sh
```

## What's Included

The Nix environment provides:

- **Python 3.11** with pip, setuptools, and wheel
- **FFmpeg** for audio/video processing
- **eSpeak** and eSpeak-data for text-to-speech
- **PortAudio** for audio input/output
- **ALSA libraries** for Linux audio support
- **Development tools** (git, curl, wget)

## Environment Variables

The shell automatically sets up:

- `PYTHONPATH` - Includes the project directories
- `LD_LIBRARY_PATH` - Includes audio library paths
- `ESPEAK_DATA_PATH` - Points to eSpeak data files
- `FFMPEG_BINARY` - Points to FFmpeg executable

## Building as a Package

To build the application as a Nix package:

```bash
# Build the package
nix-build

# Run the built application
./result/bin/translation-pwa
```

## Customization

### Adding CUDA Support

If you have CUDA available, uncomment these lines in `shell.nix`:

```nix
# cudaPackages.cudatoolkit
# cudaPackages.cudnn
```

### Adding Additional Dependencies

To add more system dependencies, add them to the `buildInputs` list in `shell.nix`:

```nix
buildInputs = with pkgs; [
  # ... existing dependencies ...
  your-new-dependency
];
```

## Troubleshooting

### Audio Issues

If you encounter audio-related issues:

1. Make sure ALSA is properly configured
2. Check that PortAudio can access audio devices
3. Verify eSpeak is working: `espeak "Hello world"`

### Python Package Issues

If Python packages fail to install:

1. Make sure you're in the nix-shell environment
2. Try creating a fresh virtual environment
3. Check that all system dependencies are available

### FFmpeg Issues

If FFmpeg-related errors occur:

1. Verify FFmpeg is available: `ffmpeg -version`
2. Check that the `FFMPEG_BINARY` environment variable is set
3. Ensure audio codecs are available

## Development Workflow

1. **Enter the environment**: `nix-shell`
2. **Set up Python**: `python -m venv venv && source venv/bin/activate`
3. **Install dependencies**: `pip install -r backend/requirements.txt`
4. **Initialize database**: `python -c "from backend.app import init_database; init_database()"`
5. **Start development**: `python backend/app.py`

## Production Deployment

For production deployment with Nix:

1. Build the package: `nix-build`
2. Copy the result to your production server
3. Set up systemd service or similar
4. Configure nginx reverse proxy

## Benefits of Using Nix

- **Reproducible environments** - Same setup across different machines
- **Isolated dependencies** - No conflicts with system packages
- **Easy cleanup** - Remove the environment without affecting the system
- **Version pinning** - Exact versions of all dependencies
- **Cross-platform** - Works on Linux, macOS, and WSL

## Alternative: Docker

If you prefer Docker over Nix, use the provided `Dockerfile`:

```bash
docker-compose up -d
```

The Nix approach is recommended for development as it provides better integration with the host system and faster iteration cycles.