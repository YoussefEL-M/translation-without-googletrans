#!/bin/bash

# Translation PWA Restart Script with No Cache
# This script stops, removes, rebuilds with no cache, and restarts the Translation PWA

echo "🔄 Restarting Translation PWA with no cache..."
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

# Function to stop and remove existing container
stop_and_remove_container() {
    if podman ps -q --filter "name=translation-pwa" | grep -q .; then
        echo "🛑 Stopping existing translation-pwa container..."
        podman stop translation-pwa
    fi
    
    if podman ps -aq --filter "name=translation-pwa" | grep -q .; then
        echo "🗑️ Removing existing translation-pwa container..."
        podman rm translation-pwa
    fi
}

# Function to remove existing image
remove_existing_image() {
    if podman images -q translation-pwa:latest | grep -q .; then
        echo "🗑️ Removing existing translation-pwa image..."
        podman rmi translation-pwa:latest
    fi
}

# Function to build with no cache and start container
rebuild_and_start() {
    echo "🔨 Building Translation PWA container with no cache..."
    podman build --no-cache -t translation-pwa:latest .
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to build container"
        exit 1
    fi
    
    echo "🚀 Starting Translation PWA with CPU-optimized TTS..."
    podman run -d \
        --name translation-pwa \
        --restart unless-stopped \
        -p 8002:8000 \
        -v ./database:/app/database \
        -v ./backend/uploads:/app/backend/uploads \
        -v ./static:/app/static \
        -v ./templates:/app/templates \
        -v ./fish_speech_models:/app/fish_speech_models \
        -e SECRET_KEY=${SECRET_KEY:-your-secret-key-here} \
        -e SMTP_SERVER=${SMTP_SERVER:-localhost} \
        -e SMTP_PORT=${SMTP_PORT:-587} \
        -e SMTP_USERNAME=${SMTP_USERNAME:-} \
        -e SMTP_PASSWORD=${SMTP_PASSWORD:-} \
        -e FROM_EMAIL=${FROM_EMAIL:-noreply@ballerup.dk} \
        -e CUDA_VISIBLE_DEVICES="" \
        -e TORCH_DEVICE=cpu \
        -e FORCE_DEVICE=cpu \
        -e COQUI_TTS_LICENSE_ACCEPTED=true \
        -e HF_HOME=/tmp/huggingface_cache \
        -e TRANSFORMERS_CACHE=/tmp/transformers_cache \
        -e HF_DATASETS_CACHE=/tmp/datasets_cache \
        --memory=4g \
        translation-pwa:latest
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to start container"
        exit 1
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "✅ Translation PWA restarted successfully with no cache!"
    echo ""
    echo "🌐 Local endpoints:"
    echo "   - Main app: http://localhost:8002/translation-pwa"
    echo "   - Health check: http://localhost:8002/translation-pwa/health"
    echo "   - Admin panel: http://localhost:8002/translation-pwa/admin"
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
    echo "   - Check TTS: podman exec translation-pwa python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
    echo ""
}

# Main execution
stop_and_remove_container
remove_existing_image
rebuild_and_start
show_status

# Follow logs if requested
if [ "$1" = "--logs" ]; then
    echo "📋 Following container logs (Ctrl+C to exit)..."
    podman logs -f translation-pwa
fi