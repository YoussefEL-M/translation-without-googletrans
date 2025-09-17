{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python and pip
    python311
    python311Packages.pip
    python311Packages.setuptools
    python311Packages.wheel
    
    # System dependencies
    ffmpeg
    espeak
    portaudio
    alsa-lib
    alsa-utils
    
    # Audio processing libraries
    libsndfile
    libsamplerate
    
    # Development tools
    git
    curl
    wget
    
    # Optional: CUDA support (uncomment if you have CUDA)
    # cudaPackages.cudatoolkit
    # cudaPackages.cudnn
  ];

  shellHook = ''
    echo "🌍 Translation PWA Development Environment"
    echo "=========================================="
    echo ""
    echo "Available commands:"
    echo "  python -m venv venv          # Create virtual environment"
    echo "  source venv/bin/activate     # Activate virtual environment"
    echo "  pip install -r backend/requirements.txt  # Install Python dependencies"
    echo "  python backend/app.py        # Start the application"
    echo "  ./start.sh                   # Use the startup script"
    echo ""
    echo "System dependencies installed:"
    echo "  ✅ Python 3.11"
    echo "  ✅ FFmpeg"
    echo "  ✅ eSpeak (TTS)"
    echo "  ✅ PortAudio"
    echo "  ✅ ALSA libraries"
    echo ""
    echo "Current directory: $(pwd)"
    echo ""
    
    # Set environment variables for better compatibility
    export PYTHONPATH="$PWD:$PWD/backend:$PYTHONPATH"
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.portaudio pkgs.alsa-lib pkgs.libsndfile pkgs.espeak]}:$LD_LIBRARY_PATH"
    
    # Create necessary directories if they don't exist
    mkdir -p database backend/uploads static/icons
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
      echo "💡 Tip: Run 'python -m venv venv' to create a virtual environment"
    else
      echo "✅ Virtual environment found"
    fi
    
    echo "Ready to develop! 🚀"
  '';

  # Environment variables
  env = {
    # Python environment
    PYTHONPATH = "$PWD:$PWD/backend";
    
    # Audio system
    ALSA_CARD = "0";
    
    # FFmpeg
    FFMPEG_BINARY = "${pkgs.ffmpeg}/bin/ffmpeg";
    
    # eSpeak
    ESPEAK_DATA_PATH = "${pkgs.espeak}/share/espeak-data";
    
    # PortAudio
    PA_ALSA_PLUGHW = "1";
  };
}