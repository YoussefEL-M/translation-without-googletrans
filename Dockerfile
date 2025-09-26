FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    portaudio19-dev \
    python3-pyaudio \
    curl \
    libsndfile1 \
    libsndfile1-dev \
    festival \
    festvox-kallpc16k \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt /app/requirements.txt

# Install Python dependencies with proper dependency management
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies first
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install Chatterbox TTS first (before other dependencies to avoid conflicts)
RUN pip install --no-cache-dir chatterbox-tts

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/database /app/backend/uploads /app/static/icons

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=backend/app.py
ENV FLASK_ENV=production
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=""

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/translation-pwa/health || exit 1

# Run the application
CMD ["python", "backend/app.py"]