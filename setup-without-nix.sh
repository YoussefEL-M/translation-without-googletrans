#!/bin/bash

# Setup script for Translation PWA without Nix
# This script installs dependencies using the system package manager

echo "🌍 Translation PWA Setup (Without Nix)"
echo "======================================"
echo ""

# Detect the operating system
if [ -f /etc/debian_version ]; then
    OS="debian"
elif [ -f /etc/redhat-release ]; then
    OS="redhat"
elif [ -f /etc/arch-release ]; then
    OS="arch"
else
    OS="unknown"
fi

echo "Detected OS: $OS"
echo ""

# Function to install packages on Debian/Ubuntu
install_debian() {
    echo "Installing packages for Debian/Ubuntu..."
    sudo apt-get update
    sudo apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        ffmpeg \
        espeak \
        espeak-data \
        libespeak1 \
        libespeak-dev \
        portaudio19-dev \
        libasound2-dev \
        libsndfile1-dev \
        libsamplerate0-dev \
        git \
        curl \
        wget
}

# Function to install packages on Red Hat/CentOS/Fedora
install_redhat() {
    echo "Installing packages for Red Hat/CentOS/Fedora..."
    sudo yum install -y \
        python3.11 \
        python3.11-pip \
        python3.11-devel \
        ffmpeg \
        espeak \
        espeak-devel \
        portaudio-devel \
        alsa-lib-devel \
        libsndfile-devel \
        libsamplerate-devel \
        git \
        curl \
        wget
}

# Function to install packages on Arch Linux
install_arch() {
    echo "Installing packages for Arch Linux..."
    sudo pacman -S --noconfirm \
        python \
        python-pip \
        ffmpeg \
        espeak \
        portaudio \
        alsa-lib \
        libsndfile \
        libsamplerate \
        git \
        curl \
        wget
}

# Install packages based on OS
case $OS in
    "debian")
        install_debian
        ;;
    "redhat")
        install_redhat
        ;;
    "arch")
        install_arch
        ;;
    *)
        echo "❌ Unsupported operating system. Please install dependencies manually:"
        echo "   - Python 3.11"
        echo "   - FFmpeg"
        echo "   - eSpeak"
        echo "   - PortAudio"
        echo "   - ALSA libraries"
        echo "   - libsndfile"
        echo "   - libsamplerate"
        exit 1
        ;;
esac

echo ""
echo "✅ System dependencies installed successfully!"
echo ""

# Set up Python environment
echo "Setting up Python environment..."

# Create virtual environment
python3.11 -m venv venv || python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

# Initialize database
echo "Initializing database..."
python -c "
import sys
sys.path.append('/opt/praktik/translation-pwa/backend')
from app import init_database
init_database()
print('✅ Database initialized successfully')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Start the app: python backend/app.py"
echo ""
echo "Or use the startup script: ./start.sh"
echo ""
echo "The application will be available at: http://localhost:5000"