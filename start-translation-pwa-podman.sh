#!/bin/bash

# Translation PWA Startup Script for Podman with GPU Support
# This script starts the Translation PWA using Podman with GPU acceleration

echo "🌐 Starting Translation PWA with Podman and GPU support..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "backend/app.py" ]; then
    echo "❌ Error: backend/app.py not found. Please run this script from the translation-pwa directory."
    exit 1
fi

# Check if Podman is available
if ! command -v podman &> /dev/null; then
    echo "❌ Error: Podman is not installed or not in PATH"
    exit 1
fi

echo "✅ Podman found"
echo "📁 Working directory: $(pwd)"
echo ""

# Function to stop existing container
stop_existing_container() {
    if podman ps -q --filter "name=translation-pwa" | grep -q .; then
        echo "🛑 Stopping existing translation-pwa container..."
        podman stop translation-pwa
        podman rm translation-pwa
    fi
}

# Function to build and start container
start_container() {
    echo "🔨 Building Translation PWA container..."
    podman build -t translation-pwa:latest .
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to build container"
        exit 1
    fi
    
    echo "🚀 Starting Translation PWA with GPU acceleration..."
    podman run -d \
        --name translation-pwa \
        --restart unless-stopped \
        -p 8000:8000 \
        -v ./database:/app/database \
        -v ./backend/uploads:/app/backend/uploads \
        -v ./static:/app/static \
        -v ./templates:/app/templates \
        -e SECRET_KEY=${SECRET_KEY:-your-secret-key-here} \
        -e SMTP_SERVER=${SMTP_SERVER:-localhost} \
        -e SMTP_PORT=${SMTP_PORT:-587} \
        -e SMTP_USERNAME=${SMTP_USERNAME:-} \
        -e SMTP_PASSWORD=${SMTP_PASSWORD:-} \
        -e FROM_EMAIL=${FROM_EMAIL:-noreply@ballerup.dk} \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e COQUI_TTS_LICENSE_ACCEPTED=true \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e PYTORCH_NVML_BASED_CUDA_CHECK=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
        -e CUDA_LAUNCH_BLOCKING=0 \
        -e BNB_CUDA_VERSION="" \
        --memory=8g \
        translation-pwa:latest
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to start container"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "✅ Translation PWA started successfully with GPU support!"
    echo ""
    echo "🌐 Local endpoints:"
    echo "   - Main app: http://localhost:8000/translation-pwa"
    echo "   - Health check: http://localhost:8000/translation-pwa/health"
    echo "   - Admin panel: http://localhost:8000/translation-pwa/admin"
    echo ""
    echo "🌐 Nginx endpoints:"
    echo "   - HTTPS: https://rosetta.semaphor.dk/translation-pwa"
    echo "   - Health: https://rosetta.semaphor.dk/translation-pwa/health"
    echo ""
    echo "📊 Container status:"
    podman ps --filter "name=translation-pwa" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "📋 Useful commands:"
    echo "   - View logs: podman logs -f translation-pwa"
    echo "   - Stop service: podman stop translation-pwa"
    echo "   - Restart service: podman restart translation-pwa"
    echo "   - Enter container: podman exec -it translation-pwa bash"
    echo "   - Check GPU: podman exec translation-pwa python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
    echo ""
}

# Main execution
stop_existing_container
start_container
show_status

# Follow logs if requested
if [ "$1" = "--logs" ]; then
    echo "📋 Following container logs (Ctrl+C to exit)..."
    podman logs -f translation-pwa
fi