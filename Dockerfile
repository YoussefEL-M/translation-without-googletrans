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
# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install all Python dependencies from requirements.txt
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
ENV TTS_HOME=/app/.cache/tts
# Enable GPU for XTTS-v2
ENV CUDA_VISIBLE_DEVICES="0"
# Coqui TTS License acceptance
ENV COQUI_TTS_LICENSE_ACCEPTED=true
ENV TTS_LICENSE_ACCEPTED=true
ENV TTS_LICENSE_ACCEPTED_CPML=true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/translation-pwa/health || exit 1

# Run the application
CMD ["python", "backend/app.py"]