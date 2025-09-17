#!/bin/bash

# Translation PWA Startup Script

echo "🌍 Starting Translation PWA..."

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "🐳 Running in Docker container"
    cd /app
else
    echo "🖥️  Running on host system"
    cd /opt/praktik/translation-pwa
fi

# Create necessary directories
mkdir -p database backend/uploads static/icons

# Set permissions
chmod 755 database backend/uploads static/icons

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment (if not in Docker)
if [ ! -f /.dockerenv ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r backend/requirements.txt

# Initialize database
echo "🗄️  Initializing database..."
python3 -c "
import sys
sys.path.append('/opt/praktik/translation-pwa/backend')
from app import init_database
init_database()
print('Database initialized successfully')
"

# Start the application
echo "🚀 Starting Translation PWA server..."
python3 backend/app.py