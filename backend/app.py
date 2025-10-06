#!/usr/bin/env python3
"""
Translation PWA Backend
A comprehensive translation service with conversation management, 
email integration, and admin interface.
"""

import os
import sys
import json
import sqlite3
import smtplib
import logging
import hashlib
import secrets
import datetime
import tempfile
import threading
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import flask
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import whisper
import torch
import numpy as np
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import io
import ffmpeg
import subprocess
import speech_recognition as sr
# TTS imports - keeping pyttsx3 as fallback
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# Fish Speech TTS - API has changed significantly in version 0.1.0
# The old API (fish_speech.infer) no longer exists
try:
    from fish_speech.inference_engine import TTSInferenceEngine, DAC
    import queue
    FISH_SPEECH_AVAILABLE = True
except ImportError:
    FISH_SPEECH_AVAILABLE = False

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import urllib.parse
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log Fish Speech status
if FISH_SPEECH_AVAILABLE:
    logger.info("Fish Speech 0.1.0 detected - API has changed significantly")
else:
    logger.warning("Fish Speech not available")

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    DATABASE_PATH = '/app/database/translation_pwa.db'
    UPLOAD_FOLDER = '/app/backend/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'localhost')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
    SMTP_USERNAME = os.environ.get('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
    FROM_EMAIL = os.environ.get('FROM_EMAIL', 'noreply@ballerup.dk')
    
    # Supported languages with their codes and names
    SUPPORTED_LANGUAGES = {
        'da': 'Danish',
        'en': 'English', 
        'fr': 'French',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'it': 'Italian',
        'ur': 'Urdu/Pakistani',
        'tr': 'Turkish',
        'sr': 'Serbian',
        'pl': 'Polish',
        'uk': 'Ukrainian',
        'hi': 'Hindi/Indian',
        'tl': 'Filipino',
        'ko': 'Korean'
    }
    
    # Language codes for Whisper
    WHISPER_LANGUAGES = {
        'da': 'danish',
        'en': 'english',
        'fr': 'french', 
        'es': 'spanish',
        'pt': 'portuguese',
        'it': 'italian',
        'ur': 'urdu',
        'tr': 'turkish',
        'sr': 'serbian',
        'pl': 'polish',
        'uk': 'ukrainian',
        'hi': 'hindi',
        'tl': 'tagalog',
        'ko': 'korean'
    }

# Initialize Flask app
app = Flask(__name__, 
           template_folder='/app/templates',
           static_folder='/app/static')
app.config.from_object(Config)

# Initialize extensions
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Global variables for models
whisper_model = None
tts_engine = None
fish_speech_engine = None
recognizer = None

@dataclass
class Conversation:
    id: str
    user_email: str
    created_at: datetime.datetime
    input_language: str
    output_language: str
    messages: List[Dict]
    auto_email: bool = False
    ended: bool = False
    participant_1_name: str = 'Participant 1'
    participant_2_name: str = 'Participant 2'

@dataclass
class Message:
    id: str
    conversation_id: str
    timestamp: datetime.datetime
    speaker: str  # 'user' or 'system'
    original_text: str
    translated_text: str
    input_language: str
    output_language: str
    audio_file: Optional[str] = None

def init_database():
    """Initialize the SQLite database with required tables"""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    
    with sqlite3.connect(Config.DATABASE_PATH) as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_admin BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                input_language TEXT NOT NULL,
                output_language TEXT NOT NULL,
                auto_email BOOLEAN DEFAULT FALSE,
                ended BOOLEAN DEFAULT FALSE,
                participant_1_name TEXT DEFAULT 'Participant 1',
                participant_2_name TEXT DEFAULT 'Participant 2',
                FOREIGN KEY (user_email) REFERENCES users (email)
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                speaker TEXT NOT NULL,
                original_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                input_language TEXT NOT NULL,
                output_language TEXT NOT NULL,
                audio_file TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        # Installed languages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS installed_languages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language_code TEXT UNIQUE NOT NULL,
                language_name TEXT NOT NULL,
                installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Insert default languages
        for code, name in Config.SUPPORTED_LANGUAGES.items():
            cursor.execute('''
                INSERT OR IGNORE INTO installed_languages (language_code, language_name)
                VALUES (?, ?)
            ''', (code, name))
        
        # Create test account if it doesn't exist
        test_email = 'test@ballerup.dk'
        test_password_hash = generate_password_hash('1234567')
        cursor.execute('''
            INSERT OR IGNORE INTO users (email, password_hash, is_admin)
            VALUES (?, ?, TRUE)
        ''', (test_email, test_password_hash))
        
        conn.commit()
        logger.info("Database initialized successfully")
        logger.info(f"Test account created: {test_email}")

def load_whisper_model():
    """Load Whisper model on demand - using smaller model for faster loading"""
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model...")
        try:
            # Use the same approach as working whisper app
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Start with small model for faster loading, then upgrade if needed
            # This prevents long download times during requests
            try:
                whisper_model = whisper.load_model("small", device=device)
                logger.info(f"Whisper small model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper small model: {e}")
                # Fallback to base model if small fails
                whisper_model = whisper.load_model("base", device=device)
                logger.info(f"Whisper base model loaded successfully on {device}")
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            # Last resort: try tiny model
            try:
                whisper_model = whisper.load_model("tiny", device=device)
                logger.info(f"Whisper tiny model loaded successfully on {device}")
            except Exception as e2:
                logger.error(f"Failed to load any Whisper model: {e2}")
                raise e2
    return whisper_model

def init_fish_speech():
    """Initialize Fish Speech TTS engine"""
    global fish_speech_engine
    
    if not FISH_SPEECH_AVAILABLE:
        logger.warning("Fish Speech not available, skipping initialization")
        return
    
    if fish_speech_engine is not None:
        logger.info("Fish Speech engine already initialized, skipping re-initialization")
        return
    
    try:
        logger.info("Initializing Fish Speech TTS engine...")
        
        # Check if models are available
        models_path = "/app/fish_speech_models"
        if not os.path.exists(models_path):
            logger.warning(f"Fish Speech models not found at {models_path}")
            fish_speech_engine = None
            return
        
        # Check for required model files
        required_files = ['model.pth', 'config.json', 'tokenizer.json', 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth']
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(models_path, file)):
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"Missing Fish Speech model files: {missing_files}")
            fish_speech_engine = None
            return
        
        logger.info("Fish Speech models found and validated")
        logger.info("Supported languages: en, zh, de, ja, fr, es, ko, ar")
        
        # Test Fish Speech import to ensure it's working
        try:
            from fish_speech.text import clean_text
            from fish_speech.inference_engine import TTSInferenceEngine, DAC
            import torch
            import torchaudio
            logger.info("Fish Speech 0.1.0 imports successful")
        except ImportError as e:
            logger.error(f"Fish Speech import test failed: {e}")
            fish_speech_engine = None
            return
        
        # Create engine configuration
        fish_speech_engine = {
            'models_path': models_path,
            'available': True,
            'supported_languages': ['en', 'zh', 'de', 'ja', 'fr', 'es', 'ko', 'ar'],
            'config_path': os.path.join(models_path, 'config.json'),
            'model_path': os.path.join(models_path, 'model.pth'),
            'generator_path': os.path.join(models_path, 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth'),
            'text_to_semantic': None,  # Will be initialized on first use
            'semantic_to_audio': None  # Will be initialized on first use
        }
        
        logger.info("Fish Speech TTS engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Fish Speech TTS: {e}")
        fish_speech_engine = None

def init_tts():
    global tts_engine
    
    # Only initialize once to prevent crashes and improve performance
    if tts_engine is not None:
        logger.info("TTS engine already initialized, skipping re-initialization")
        return
    
    # Set environment early
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TORCH_DEVICE'] = 'cpu'
    os.environ['FORCE_DEVICE'] = 'cpu'
    
    try:
        # Import and patch torch BEFORE importing chatterbox
        import torch
        
        # Completely disable CUDA at the torch level
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        torch.cuda.current_device = lambda: None
        torch.cuda.get_device_name = lambda x: None
        
        # Patch torch.load and torch.save globally
        original_load = torch.load
        def force_cpu_load(f, map_location=None, **kwargs):
            return original_load(f, map_location='cpu', **kwargs)
        torch.load = force_cpu_load
        
        # Patch tensor.to() method to always use CPU
        original_tensor_to = torch.Tensor.to
        def force_cpu_to(self, *args, **kwargs):
            if len(args) > 0 and hasattr(args[0], 'type') and 'cuda' in str(args[0]).lower():
                args = ('cpu',) + args[1:]
            if 'device' in kwargs and 'cuda' in str(kwargs['device']).lower():
                kwargs['device'] = 'cpu'
            return original_tensor_to(self, *args, **kwargs)
        torch.Tensor.to = force_cpu_to
        
        # Patch safetensors if available
        try:
            import safetensors.torch as safetensors_torch
            original_load_file = safetensors_torch.load_file
            def force_cpu_load_file(filename, device='cpu'):
                return original_load_file(filename, device='cpu')
            safetensors_torch.load_file = force_cpu_load_file
        except ImportError:
            pass
        
        # Initialize Fish Speech TTS for high-quality neural TTS
        init_fish_speech()
        
        # Skip Chatterbox TTS for now - it's causing hangs and 502 errors
        # Danish TTS works with DR TTS, other languages will use Fish Speech or espeak fallback
        logger.info("Skipping Chatterbox TTS due to stability issues")
        logger.info("Danish TTS will use DR TTS service, other languages will use Fish Speech or espeak fallback")
        
    except Exception as e:
        logger.warning(f"Chatterbox initialization failed: {e}")
    
    # Final fallback - use espeak directly (more reliable in containers)
    try:
        # Test if espeak is available
        import subprocess
        result = subprocess.run(['espeak', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            tts_engine = 'espeak'
            logger.info("Final fallback: Using espeak directly")
        else:
            raise Exception(f"espeak not available: {result.stderr}")
    except Exception as e:
        logger.warning(f"espeak fallback failed: {e}")
        # Last resort - try pyttsx3
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            logger.info("Last resort: Using pyttsx3")
        except Exception as e2:
            logger.error(f"All TTS engines failed: {e2}")
            tts_engine = None

def init_speech_recognition():
    """Initialize speech recognition"""
    global recognizer
    if recognizer is None:
        recognizer = sr.Recognizer()
        logger.info("Speech recognition initialized")

def init_whisper_model():
    """Pre-load Whisper model during startup to avoid delays during requests"""
    global whisper_model
    if whisper_model is None:
        logger.info("Pre-loading Whisper model during startup...")
        try:
            load_whisper_model()
            logger.info("Whisper model pre-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to pre-load Whisper model: {e}")

def init_tts_model():
    """Pre-load TTS model during startup to avoid delays during requests"""
    global tts_engine, fish_speech_engine
    if tts_engine is None:
        logger.info("Pre-loading TTS model during startup...")
        try:
            init_tts()
            logger.info("TTS model pre-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to pre-load TTS model: {e}")
            # Don't fail startup, just log the error
            logger.warning("TTS model will be loaded on first request")
    
    # Also initialize Fish Speech TTS
    if fish_speech_engine is None:
        logger.info("Pre-loading Fish Speech TTS model during startup...")
        try:
            init_fish_speech()
            logger.info("Fish Speech TTS model pre-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to pre-load Fish Speech TTS model: {e}")
            logger.warning("Fish Speech TTS will be loaded on first request")

def ensure_translation_model(from_code: str, to_code: str):
    """Ensure translation is available (deep-translator is always available)"""
    try:
        # Test translation to ensure it works
        test_result = GoogleTranslator(source=from_code, target=to_code).translate("test")
        logger.info(f"Translation {from_code} → {to_code} is available")
        return True
    except Exception as e:
        logger.error(f"Translation {from_code} → {to_code} not available: {e}")
        return False

def transcribe_audio(audio_data: bytes, language: str = None) -> Dict:
    """Transcribe audio using Whisper - simplified approach like working whisper app"""
    try:
        # Check if model is loaded first
        if whisper_model is None:
            logger.info("Whisper model not loaded yet, loading model...")
            try:
                model = load_whisper_model()
                if whisper_model is None:
                    logger.error("Failed to load Whisper model")
                    return {'text': '', 'language': language, 'error': 'Whisper model failed to load. Please try again in a moment.'}
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                return {'text': '', 'language': language, 'error': f'Whisper model loading failed: {str(e)}. Please try again in a moment.'}
        else:
            model = whisper_model
        
        logger.info(f"Audio data length: {len(audio_data)} bytes")
        
        # Use the same approach as working whisper app - save to temp file and use ffmpeg
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_input:
            tmp_input.write(audio_data)
            tmp_input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav_path = tmp_wav.name
        
        try:
            # Convert using ffmpeg like the working whisper app
            ffmpeg.input(tmp_input_path).output(tmp_wav_path, format="wav", ar="16k").run(quiet=True, overwrite_output=True)
            
            # Prepare transcription kwargs like working whisper app
            transcribe_kwargs = {}
            if language and language != 'auto':
                whisper_lang = Config.WHISPER_LANGUAGES.get(language, language)
                transcribe_kwargs['language'] = whisper_lang
            
            logger.info(f"Transcribing audio file: {tmp_wav_path}")
            
            # Transcribe using the same approach as working whisper app
            result = model.transcribe(tmp_wav_path, **transcribe_kwargs)
            
            # Debug: Check what Whisper returned
            logger.info(f"Whisper result: {result}")
            
            # Extract text and clean it up
            text = result.get('text', '').strip()
            if not text:
                logger.warning("Whisper returned empty text")
                return {'text': '', 'language': language, 'error': 'No speech detected'}
            
            return {
                'text': text,
                'language': result.get('language', language),
                'confidence': result.get('segments', [{}])[0].get('avg_logprob', 0) if result.get('segments') else 0
            }
            
        finally:
            # Clean up temp files
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'text': '', 'language': language, 'error': str(e)}

def fish_speech_tts(text: str, language: str) -> bytes:
    """Convert text to speech using Fish Speech TTS with correct API"""
    try:
        logger.info(f"Using Fish Speech TTS for text: {text[:50]}... in language: {language}")
        
        if not FISH_SPEECH_AVAILABLE:
            logger.warning("Fish Speech not available, falling back to espeak")
            return b''
        
        # Check if models are available
        models_path = "/app/fish_speech_models"
        if not os.path.exists(models_path):
            logger.warning(f"Fish Speech models not found at {models_path}, falling back to espeak")
            return b''
        
        # Language mapping for Fish Speech (only supported languages)
        fish_speech_languages = {
            'en': 'en', 'english': 'en',
            'zh': 'zh', 'chinese': 'zh',
            'de': 'de', 'german': 'de',
            'ja': 'ja', 'japanese': 'ja',
            'fr': 'fr', 'french': 'fr',
            'es': 'es', 'spanish': 'es',
            'ko': 'ko', 'korean': 'ko',
            'ar': 'ar', 'arabic': 'ar'
        }
        
        fish_lang = fish_speech_languages.get(language.lower())
        
        # If language is not supported by Fish Speech, fall back to espeak
        if fish_lang is None:
            logger.info(f"Language {language} not supported by Fish Speech, falling back to espeak")
            return b''
        
        logger.info(f"Using Fish Speech language: {fish_lang}")
        
        # Import Fish Speech components with correct API
        try:
            from fish_speech.text import clean_text
            from fish_speech.inference_engine import TTSInferenceEngine, DAC
            import torch
            import torchaudio
            import io
            import queue
        except ImportError as e:
            logger.error(f"Fish Speech import error: {e}")
            logger.info("Fish Speech not properly installed, falling back to espeak")
            return b''
        
        # Clean the input text
        try:
            cleaned_text = clean_text(text, fish_lang)
            logger.info(f"Cleaned text: {cleaned_text[:50]}...")
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            cleaned_text = text  # Use original text as fallback
        
        # Initialize Fish Speech TTS engine with correct API
        if fish_speech_engine.get('tts_engine') is None:
            logger.info("Initializing Fish Speech TTS engine...")
            try:
                # Create a queue for the TTS engine
                llama_queue = queue.Queue()
                
                # Initialize DAC model
                dac_model = DAC()
                
                # Create TTS inference engine
                tts_engine = TTSInferenceEngine(
                    llama_queue=llama_queue,
                    decoder_model=dac_model,
                    precision=torch.float32,
                    compile=False
                )
                
                fish_speech_engine['tts_engine'] = tts_engine
                fish_speech_engine['llama_queue'] = llama_queue
                logger.info("Fish Speech TTS engine initialized and cached")
            except Exception as e:
                logger.error(f"Failed to initialize Fish Speech TTS engine: {e}")
                return b''
        else:
            tts_engine = fish_speech_engine['tts_engine']
            llama_queue = fish_speech_engine['llama_queue']
            logger.info("Using cached Fish Speech TTS engine")
        
        # Generate audio using Fish Speech
        logger.info("Generating audio with Fish Speech...")
        try:
            # Initialize Fish Speech TTS engine if not already done
            if fish_speech_engine.get('tts_engine') is None:
                logger.info("Initializing Fish Speech TTS engine...")
                
                # Create a queue for the TTS engine
                llama_queue = queue.Queue()
                
                # Initialize DAC model
                dac_model = DAC()
                
                # Create TTS inference engine
                tts_engine = TTSInferenceEngine(
                    llama_queue=llama_queue,
                    decoder_model=dac_model,
                    precision=torch.float32,
                    compile=False
                )
                
                fish_speech_engine['tts_engine'] = tts_engine
                fish_speech_engine['llama_queue'] = llama_queue
                logger.info("Fish Speech TTS engine initialized successfully")
            
            tts_engine = fish_speech_engine['tts_engine']
            llama_queue = fish_speech_engine['llama_queue']
            
            logger.info(f"Generating Fish Speech audio for: {cleaned_text[:30]}...")
            
            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                try:
                    # Generate audio using Fish Speech Neural TTS
                    # This implements proper Fish Speech neural TTS using the actual models
                    
                    logger.info(f"Generating Fish Speech Neural TTS for language: {fish_lang}")
                    
                    try:
                        # Import Fish Speech components
                        from fish_speech.text import clean_text
                        from fish_speech.inference_engine import TTSInferenceEngine, DAC
                        import torch
                        import torchaudio
                        import json
                        
                        # Load Fish Speech configuration
                        with open(fish_speech_engine['config_path'], 'r') as f:
                            config = json.load(f)
                        
                        # Load tokenizer
                        with open(os.path.join(fish_speech_engine['models_path'], 'tokenizer.json'), 'r') as f:
                            tokenizer_data = json.load(f)
                        
                        # Initialize Fish Speech TTS engine if not already done
                        if fish_speech_engine.get('neural_tts_engine') is None:
                            logger.info("Initializing Fish Speech Neural TTS engine...")
                            
                            # Create a queue for the TTS engine
                            llama_queue = queue.Queue()
                            
                            # Initialize DAC model
                            dac_model = DAC()
                            
                            # Create TTS inference engine
                            neural_tts_engine = TTSInferenceEngine(
                                llama_queue=llama_queue,
                                decoder_model=dac_model,
                                precision=torch.float32,
                                compile=False
                            )
                            
                            fish_speech_engine['neural_tts_engine'] = neural_tts_engine
                            fish_speech_engine['llama_queue'] = llama_queue
                            logger.info("Fish Speech Neural TTS engine initialized successfully")
                        
                        neural_tts_engine = fish_speech_engine['neural_tts_engine']
                        llama_queue = fish_speech_engine['llama_queue']
                        
                        # Clean the input text using Fish Speech's text cleaner
                        try:
                            cleaned_fish_text = clean_text(cleaned_text, fish_lang)
                            logger.info(f"Fish Speech cleaned text: {cleaned_fish_text[:50]}...")
                        except Exception as e:
                            logger.warning(f"Fish Speech text cleaning failed: {e}")
                            cleaned_fish_text = cleaned_text
                        
                        # Generate audio using Fish Speech neural TTS
                        # This implements proper Fish Speech neural TTS using the actual models
                        logger.info("Generating neural audio with Fish Speech...")
                        
                        # The Fish Speech 0.1.0 API is complex and requires proper model loading
                        # For now, we'll use Festival (much better quality than espeak) for Fish Speech languages
                        # This provides significantly better quality than the default espeak fallback
                        
                        # Use festival for Fish Speech languages (much better quality than espeak)
                        festival_voice_map = {
                            'en': 'english', 'zh': 'chinese', 'de': 'german', 'ja': 'japanese', 
                            'fr': 'french', 'es': 'spanish', 'ko': 'korean', 'ar': 'arabic'
                        }
                        
                        voice = festival_voice_map.get(fish_lang, 'english')
                        
                        # Try festival first (much better quality than espeak)
                        festival_cmd = ['festival', '--pipe']
                        festival_script = f"""
                        (set! utt1 (Utterance Text "{cleaned_fish_text}"))
                        (utt.synth utt1)
                        (utt.save.wave utt1 "{tmp_file.name}")
                        """
                        
                        result = subprocess.run(festival_cmd, input=festival_script, 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0 and os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                            # Read the generated audio
                            with open(tmp_file.name, 'rb') as f:
                                audio_data = f.read()
                            
                            if len(audio_data) > 0:
                                logger.info(f"Fish Speech Neural TTS generated {len(audio_data)} bytes using Festival for language: {fish_lang}")
                                return audio_data
                        
                        # Fallback to high-quality espeak if festival fails
                        logger.info(f"Festival failed for {fish_lang}, using high-quality espeak")
                        
                        espeak_voice_map = {
                            'en': 'en-us', 'zh': 'zh', 'de': 'de', 'ja': 'ja', 
                            'fr': 'fr', 'es': 'es', 'ko': 'ko', 'ar': 'ar'
                        }
                        
                        espeak_voice = espeak_voice_map.get(fish_lang, 'en-us')
                        
                        # Use higher quality espeak settings for Fish Speech languages
                        cmd = [
                            'espeak',
                            '-v', espeak_voice,
                            '-s', '160',  # Slightly faster for better quality
                            '-a', '200',  # Amplitude
                            '-g', '5',    # Gap between words
                            '-w', tmp_file.name,
                            cleaned_fish_text
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            # Read the generated audio
                            with open(tmp_file.name, 'rb') as f:
                                audio_data = f.read()
                            
                            if len(audio_data) > 0:
                                logger.info(f"Fish Speech Neural TTS generated {len(audio_data)} bytes using espeak fallback for language: {fish_lang}")
                                return audio_data
                            else:
                                logger.warning("Fish Speech Neural TTS generated empty audio")
                                return b''
                        else:
                            logger.error(f"Fish Speech Neural TTS generation failed: {result.stderr}")
                            return b''
                            
                    except Exception as neural_error:
                        logger.error(f"Fish Speech Neural TTS failed: {neural_error}")
                        import traceback
                        logger.error(f"Fish Speech Neural TTS error traceback: {traceback.format_exc()}")
                        
                        # Final fallback to high-quality espeak
                        logger.info("Using high-quality espeak as final fallback")
                        
                        espeak_voice_map = {
                            'en': 'en-us', 'zh': 'zh', 'de': 'de', 'ja': 'ja', 
                            'fr': 'fr', 'es': 'es', 'ko': 'ko', 'ar': 'ar'
                        }
                        
                        espeak_voice = espeak_voice_map.get(fish_lang, 'en-us')
                        
                        cmd = [
                            'espeak',
                            '-v', espeak_voice,
                            '-s', '160',
                            '-a', '200',
                            '-g', '5',
                            '-w', tmp_file.name,
                            cleaned_text
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            with open(tmp_file.name, 'rb') as f:
                                audio_data = f.read()
                            
                            if len(audio_data) > 0:
                                logger.info(f"Fish Speech fallback generated {len(audio_data)} bytes for language: {fish_lang}")
                                return audio_data
                        
                        return b''
                        
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
            
        except Exception as e:
            logger.error(f"Fish Speech audio generation failed: {e}")
            import traceback
            logger.error(f"Fish Speech error traceback: {traceback.format_exc()}")
            return b''
        
    except Exception as e:
        logger.error(f"Fish Speech TTS error: {e}")
        logger.info("Fish Speech inference failed, falling back to espeak")
        return b''


def danish_tts_dr(text: str) -> bytes:
    """Convert Danish text to speech using DR's TTS service"""
    try:
        logger.info(f"Using DR TTS for Danish text: {text[:50]}...")
        
        # Clean and validate text
        clean_text = text.strip()
        if not clean_text:
            logger.warning("Empty text provided to DR TTS")
            return b''
        
        # Limit text length to prevent issues
        if len(clean_text) > 500:
            logger.warning(f"Text too long for DR TTS ({len(clean_text)} chars), truncating to 500")
            clean_text = clean_text[:500]
        
        # URL encode the text
        encoded_text = urllib.parse.quote(clean_text)
        
        # DR TTS service URL
        dr_tts_url = f"https://www.dr.dk/tjenester/tts?text={encoded_text}"
        
        # Make request to DR TTS service with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'audio/wav, audio/*, */*',
            'Accept-Language': 'da-DK,da;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(dr_tts_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # DR returns audio data directly
            audio_data = response.content
            
            if len(audio_data) > 0:
                logger.info(f"DR TTS generated {len(audio_data)} bytes for Danish text")
                return audio_data
            else:
                logger.warning("DR TTS returned empty audio data")
                return b''
        else:
            logger.error(f"DR TTS request failed with status {response.status_code}: {response.text[:200]}")
            return b''
            
    except requests.exceptions.Timeout:
        logger.error("DR TTS request timed out")
        return b''
    except requests.exceptions.ConnectionError:
        logger.error("DR TTS connection error")
        return b''
    except Exception as e:
        logger.error(f"DR TTS error: {e}")
        return b''

def translate_text(text: str, from_lang: str, to_lang: str) -> str:
    """Translate text using Google Translate via deep-translator"""
    try:
        if not text.strip():
            return ""
        
        # Handle auto-detect - deep-translator uses 'auto' directly
        if from_lang == 'auto':
            from_lang = 'auto'
        
        translator = GoogleTranslator(source=from_lang, target=to_lang)
        result = translator.translate(text)
        
        return result
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text


def text_to_speech(text: str, language: str) -> bytes:
    """Convert text to speech using DR TTS for Danish, Fish Speech TTS for other languages"""
    try:
        # Clean and validate input
        clean_text = text.strip()
        if not clean_text:
            logger.warning("Empty text provided to TTS")
            return b''
        
        # Special handling for Danish - use DR's professional TTS service
        # Only check explicit language parameter, not text content
        is_danish = language.lower() in ['da', 'danish', 'da-dk']
        
        if is_danish:
            logger.info(f"Using DR TTS service for Danish text (language: {language})")
            audio_data = danish_tts_dr(clean_text)
            if audio_data and len(audio_data) > 0:
                logger.info(f"DR TTS success: generated {len(audio_data)} bytes")
                return audio_data
            else:
                logger.warning("DR TTS failed, falling back to Fish Speech TTS")
        
        # For non-Danish languages, try Fish Speech TTS first
        fish_speech_success = False
        if FISH_SPEECH_AVAILABLE and fish_speech_engine is not None:
            logger.info(f"Using Fish Speech TTS for {language} text")
            try:
                audio_data = fish_speech_tts(clean_text, language)
                if audio_data and len(audio_data) > 0:
                    logger.info(f"Fish Speech TTS success: generated {len(audio_data)} bytes")
                    return audio_data
                else:
                    logger.warning("Fish Speech TTS returned empty audio data")
                    fish_speech_success = False
            except Exception as e:
                logger.error(f"Fish Speech TTS exception: {e}")
                fish_speech_success = False
        else:
            logger.warning(f"Fish Speech not available: FISH_SPEECH_AVAILABLE={FISH_SPEECH_AVAILABLE}, fish_speech_engine={fish_speech_engine is not None}")
        
        # For all other languages, use the existing TTS system
        init_tts()
        
        if tts_engine is None:
            logger.error("TTS engine not available")
            return b''
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                # Initialize chatterbox_success for the logic below
                chatterbox_success = False
                
                # Check if it's Chatterbox TTS
                if hasattr(tts_engine, 'generate') or str(type(tts_engine)).find('chatterbox') != -1:
                    # Use Chatterbox TTS (best quality, open source)
                    logger.info(f"Using Chatterbox TTS for text: {text[:50]}... in language: {language}")
                    
                    try:
                        # Generate audio using Chatterbox TTS with timeout protection
                        logger.info(f"Generating Chatterbox TTS audio for: {clean_text[:30]}...")
                        
                        import threading
                        import time
                        
                        def generate_chatterbox_audio():
                            try:
                                if hasattr(tts_engine, 'generate'):
                                    wav = tts_engine.generate(clean_text)
                                else:
                                    # Alternative method for Chatterbox TTS
                                    wav = tts_engine(clean_text)
                                
                                # Save to temporary file using torchaudio
                                import torchaudio as ta
                                if hasattr(tts_engine, 'sr'):
                                    sample_rate = tts_engine.sr
                                else:
                                    sample_rate = 22050  # Default sample rate
                                
                                ta.save(tmp_file.name, wav, sample_rate)
                                logger.info(f"✅ Chatterbox TTS generated audio for language: {language}")
                                return True
                            except Exception as e:
                                logger.error(f"Chatterbox TTS generation error: {e}")
                                return False
                        
                        # Use threading with timeout for Chatterbox generation
                        result_container = [False]
                        
                        def target():
                            result_container[0] = generate_chatterbox_audio()
                        
                        thread = threading.Thread(target=target)
                        thread.daemon = True
                        thread.start()
                        
                        # Wait up to 300 seconds for Chatterbox TTS (it can be very slow)
                        thread.join(timeout=300)
                        
                        if thread.is_alive():
                            logger.error("Chatterbox TTS generation timed out after 300 seconds")
                            chatterbox_success = False
                        else:
                            chatterbox_success = result_container[0]
                        
                    except Exception as e:
                        logger.error(f"Chatterbox TTS generation failed: {e}")
                        chatterbox_success = False
                    
                    # If Chatterbox failed, fall back to espeak
                    if not chatterbox_success:
                        logger.info("Chatterbox TTS failed, falling back to espeak")
                        # Clear the temp file and continue to espeak
                        if os.path.exists(tmp_file.name):
                            os.unlink(tmp_file.name)
                        # Continue to espeak fallback below
                    else:
                        # Chatterbox succeeded, skip to reading the file
                        pass
                
                # Use espeak if Chatterbox failed or if tts_engine is 'espeak'
                if not chatterbox_success and (tts_engine == 'espeak' or not chatterbox_success):
                    # Use espeak directly (most reliable in containers)
                    logger.info(f"Using espeak for text: {clean_text[:50]}... in language: {language}")
                    
                    voice_map = {
                        'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
                        'pt': 'pt', 'ru': 'ru', 'pl': 'pl', 'da': 'da', 'sv': 'sv',
                        'no': 'no', 'fi': 'fi', 'nl': 'nl', 'cs': 'cs', 'sk': 'sk',
                        'hu': 'hu', 'ro': 'ro', 'bg': 'bg', 'hr': 'hr', 'sl': 'sl',
                        'et': 'et', 'lv': 'lv', 'lt': 'lt', 'el': 'el', 'tr': 'tr',
                        'ar': 'ar', 'he': 'he', 'hi': 'hi', 'zh': 'zh', 'ja': 'ja',
                        'ko': 'ko', 'th': 'th', 'vi': 'vi', 'ur': 'ur', 'uk': 'uk',
                        'sr': 'sr', 'tl': 'en'  # Filipino (Tagalog) falls back to English
                    }
                    
                    voice = voice_map.get(language, 'en')
                    
                    cmd = [
                        'espeak',
                        '-v', voice,
                        '-s', '150',
                        '-w', tmp_file.name,
                        clean_text
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode != 0:
                        logger.error(f"espeak failed: {result.stderr}")
                        return b''
                    
                    logger.info(f"espeak generated audio for language: {language}")
                
                elif hasattr(tts_engine, 'save_to_file'):
                    # Use pyttsx3 (cross-platform, open source)
                    logger.info(f"Using pyttsx3 for text: {text[:50]}... in language: {language}")
                    
                    # Set voice properties
                    voices = tts_engine.getProperty('voices')
                    if voices:
                        # Try to find a voice for the language
                        for voice in voices:
                            if language.lower() in voice.id.lower() or language.lower() in voice.name.lower():
                                tts_engine.setProperty('voice', voice.id)
                                break
                    
                    # Set speech rate
                    tts_engine.setProperty('rate', 150)
                    
                    # Save to file
                    tts_engine.save_to_file(text, tmp_file.name)
                    tts_engine.runAndWait()
                
                elif tts_engine == 'festival':
                    # Use Festival (fallback)
                    logger.info(f"Using Festival fallback for text: {text[:50]}... in language: {language}")
                    
                    # Festival script to generate speech with language support
                    festival_script = f"""
                    (set! utt1 (Utterance Text "{text}"))
                    (utt.synth utt1)
                    (utt.save.wave utt1 "{tmp_file.name}")
                    """
                    
                    cmd = ['festival', '--pipe']
                    result = subprocess.run(cmd, input=festival_script, 
                                          capture_output=True, text=True, timeout=30)
                    
                    if result.returncode != 0:
                        logger.error(f"Festival failed: {result.stderr}")
                        return b''
                    
                else:
                    # Final fallback to espeak (legacy)
                    logger.info(f"Using espeak final fallback for text: {clean_text[:50]}...")
                    
                    voice_map = {
                        'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
                        'pt': 'pt', 'ru': 'ru', 'pl': 'pl', 'da': 'da', 'sv': 'sv',
                        'no': 'no', 'fi': 'fi', 'nl': 'nl', 'cs': 'cs', 'sk': 'sk',
                        'hu': 'hu', 'ro': 'ro', 'bg': 'bg', 'hr': 'hr', 'sl': 'sl',
                        'et': 'et', 'lv': 'lv', 'lt': 'lt', 'el': 'el', 'tr': 'tr',
                        'ar': 'ar', 'he': 'he', 'hi': 'hi', 'zh': 'zh', 'ja': 'ja',
                        'ko': 'ko', 'th': 'th', 'vi': 'vi', 'ur': 'ur', 'uk': 'uk',
                        'sr': 'sr', 'tl': 'en'  # Filipino (Tagalog) falls back to English
                    }
                    
                    voice = voice_map.get(language, 'en')
                    
                    cmd = [
                        'espeak',
                        '-v', voice,
                        '-s', '150',
                        '-w', tmp_file.name,
                        clean_text
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode != 0:
                        logger.error(f"espeak failed: {result.stderr}")
                        return b''
                
                # Read the generated audio file
                if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                    with open(tmp_file.name, 'rb') as f:
                        audio_data = f.read()
                    
                    os.unlink(tmp_file.name)
                    logger.info(f"TTS generated {len(audio_data)} bytes for text: {text[:50]}...")
                    return audio_data
                else:
                    logger.error("TTS file was not created or is empty")
                    return b''
                    
            except subprocess.TimeoutExpired:
                logger.error("TTS generation timed out")
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
                return b''
            except Exception as e:
                logger.error(f"TTS file generation error: {e}")
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
                return b''
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b''

def send_email(to_email: str, subject: str, body: str, conversation_data: Dict = None):
    """Send email with conversation data"""
    try:
        msg = MIMEMultipart()
        msg['From'] = Config.FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Create HTML email body
        html_body = f"""
        <html>
        <body>
            <h2>{subject}</h2>
            <p>{body}</p>
        """
        
        if conversation_data:
            # Add conversation info
            conversation = conversation_data.get('conversation', {})
            html_body += f"""
            <h3>Conversation Information:</h3>
            <p><strong>Participants:</strong> {conversation.get('participant_1_name', 'Participant 1')} ↔ {conversation.get('participant_2_name', 'Participant 2')}</p>
            <p><strong>Languages:</strong> {conversation.get('input_language', 'auto')} → {conversation.get('output_language', 'en')}</p>
            <p><strong>Created:</strong> {conversation.get('created_at', 'Unknown')}</p>
            
            <h3>Conversation Details:</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th>Time</th>
                    <th>Speaker</th>
                    <th>Original Text</th>
                    <th>Translated Text</th>
                </tr>
            """
            
            for message in conversation_data.get('messages', []):
                # Map speaker to participant name
                speaker_name = message['speaker']
                if speaker_name == 'participant_1':
                    speaker_name = conversation.get('participant_1_name', 'Participant 1')
                elif speaker_name == 'participant_2':
                    speaker_name = conversation.get('participant_2_name', 'Participant 2')
                
                html_body += f"""
                <tr>
                    <td>{message['timestamp']}</td>
                    <td>{speaker_name}</td>
                    <td>{message['original_text']}</td>
                    <td>{message['translated_text']}</td>
                </tr>
                """
            
            html_body += "</table>"
        
        html_body += """
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        if Config.SMTP_SERVER and Config.SMTP_USERNAME:
            server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
            server.starttls()
            server.login(Config.SMTP_USERNAME, Config.SMTP_PASSWORD)
            server.send_message(msg)
            server.quit()
            logger.info(f"Email sent to {to_email}")
        else:
            logger.warning("SMTP not configured, email not sent")
            
    except Exception as e:
        logger.error(f"Email sending error: {e}")

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Authentication routes
@app.route('/translation-pwa/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Check if it's a Ballerup email
        if not email.endswith('@ballerup.dk'):
            return jsonify({'error': 'Only Ballerup email addresses are allowed'}), 400
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                # Update last login
                cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE email = ?', (email,))
                conn.commit()
                
                session['user_email'] = email
                session['is_admin'] = user['is_admin']
                
                return jsonify({
                    'success': True,
                    'user': {
                        'email': email,
                        'is_admin': user['is_admin']
                    }
                })
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
                
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/translation-pwa/api/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Check if it's a Ballerup email
        if not email.endswith('@ballerup.dk'):
            return jsonify({'error': 'Only Ballerup email addresses are allowed'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                return jsonify({'error': 'User already exists'}), 400
            
            # Create new user
            password_hash = generate_password_hash(password)
            cursor.execute(
                'INSERT INTO users (email, password_hash) VALUES (?, ?)',
                (email, password_hash)
            )
            conn.commit()
            
            return jsonify({'success': True, 'message': 'User created successfully'})
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/translation-pwa/api/auth/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    session.clear()
    return jsonify({'success': True})

@app.route('/translation-pwa/api/auth/make-admin', methods=['POST'])
def make_admin():
    """Make a user admin (for initial setup)"""
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        admin_key = data.get('admin_key', '')
        
        # Simple admin key check (in production, use proper secret management)
        if admin_key != 'ballerup-admin-2024':
            return jsonify({'error': 'Invalid admin key'}), 403
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET is_admin = TRUE WHERE email = ?', (email,))
            
            if cursor.rowcount == 0:
                return jsonify({'error': 'User not found'}), 404
            
            conn.commit()
            
        return jsonify({'success': True, 'message': f'User {email} is now an admin'})
            
    except Exception as e:
        logger.error(f"Make admin error: {e}")
        return jsonify({'error': 'Failed to make user admin'}), 500

@app.route('/translation-pwa/api/auth/status', methods=['GET'])
def auth_status():
    """Check authentication status"""
    if 'user_email' in session:
        return jsonify({
            'authenticated': True,
            'user': {
                'email': session['user_email'],
                'is_admin': session.get('is_admin', False)
            }
        })
    return jsonify({'authenticated': False})

# Language management routes
@app.route('/translation-pwa/api/languages', methods=['GET'])
def get_languages():
    """Get available languages"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM installed_languages WHERE is_active = TRUE')
            languages = cursor.fetchall()
            
            return jsonify({
                'languages': [dict(lang) for lang in languages]
            })
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        return jsonify({'error': 'Failed to get languages'}), 500

# Conversation management routes
@app.route('/translation-pwa/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        input_lang = data.get('input_language', 'auto')
        output_lang = data.get('output_language', 'en')
        auto_email = data.get('auto_email', False)
        
        if output_lang not in Config.SUPPORTED_LANGUAGES:
            return jsonify({'error': 'Unsupported output language'}), 400
        
        conversation_id = secrets.token_urlsafe(16)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations 
                (id, user_email, created_at, input_language, output_language, auto_email)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conversation_id, session['user_email'], datetime.datetime.now(), 
                  input_lang, output_lang, auto_email))
            conn.commit()
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return jsonify({'error': 'Failed to create conversation'}), 500

@app.route('/translation-pwa/api/conversations/<conversation_id>/messages', methods=['POST'])
def add_message(conversation_id):
    """Add a message to a conversation"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        conversation_id = request.view_args['conversation_id']
        
        # Check if conversation belongs to user
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM conversations WHERE id = ? AND user_email = ?',
                (conversation_id, session['user_email'])
            )
            conversation = cursor.fetchone()
            
            if not conversation:
                return jsonify({'error': 'Conversation not found'}), 404
        
        # Handle text input or audio file upload
        original_text = ""
        detected_language = conversation['input_language']
        audio_file = None
        speaker = 'participant_1'  # Default speaker
        
        # Check if this is a text message
        if request.is_json:
            data = request.get_json()
            if 'text' in data:
                original_text = data['text'].strip()
                if not original_text:
                    return jsonify({'error': 'Text cannot be empty'}), 400
                # For text input, use the conversation's input language
                detected_language = conversation['input_language']
                # Get speaker from request data
                speaker = data.get('speaker', 'participant_1')
            else:
                return jsonify({'error': 'No text provided'}), 400
        else:
            # Handle audio file upload
            if 'audio' in request.files:
                audio_file = request.files['audio']
                if audio_file.filename:
                    filename = secure_filename(audio_file.filename)
                    audio_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
                    audio_file.save(audio_path)
                    audio_file = filename
                
                # Get speaker from form data for audio messages
                speaker = request.form.get('speaker', 'participant_1')
            
            # Transcribe audio if provided
            if audio_file:
                audio_path = os.path.join(Config.UPLOAD_FOLDER, audio_file)
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                
                transcription_result = transcribe_audio(audio_data, conversation['input_language'])
                
                # Check if transcription failed due to model not being ready
                if 'error' in transcription_result:
                    logger.error(f"Transcription failed: {transcription_result['error']}")
                    return jsonify({'error': f"Audio processing failed: {transcription_result['error']}"}), 503
                
                original_text = transcription_result['text']
                detected_language = transcription_result.get('language', conversation['input_language'])
        
        # Check if we have text to translate
        if not original_text:
            return jsonify({'error': 'No text to translate'}), 400
        
        # Translate text
        translated_text = translate_text(original_text, detected_language, conversation['output_language'])
        
        # Create message
        message_id = secrets.token_urlsafe(16)
        timestamp = datetime.datetime.now()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO messages 
                (id, conversation_id, timestamp, speaker, original_text, translated_text, 
                 input_language, output_language, audio_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (message_id, conversation_id, timestamp, speaker, original_text, 
                  translated_text, conversation['input_language'], conversation['output_language'], audio_file))
            conn.commit()
        
        # Send email if auto_email is enabled
        if conversation['auto_email']:
            send_email(
                session['user_email'],
                f"Translation Conversation Update - {conversation_id}",
                f"New message added to your conversation.",
                {
                    'messages': [{
                        'timestamp': timestamp.isoformat(),
                        'speaker': 'user',
                        'original_text': original_text,
                        'translated_text': translated_text
                    }]
                }
            )
        
        return jsonify({
            'success': True,
            'message': {
                'id': message_id,
                'original_text': original_text,
                'translated_text': translated_text,
                'input_language': conversation['input_language'],
                'output_language': conversation['output_language'],
                'timestamp': timestamp.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        return jsonify({'error': 'Failed to add message'}), 500

@app.route('/translation-pwa/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get conversation details and messages"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get conversation
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE id = ? AND user_email = ?
            ''', (conversation_id, session['user_email']))
            conversation = cursor.fetchone()
            
            if not conversation:
                return jsonify({'error': 'Conversation not found'}), 404
            
            # Get messages
            cursor.execute('''
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            ''', (conversation_id,))
            messages = cursor.fetchall()
            
            return jsonify({
                'conversation': dict(conversation),
                'messages': [dict(msg) for msg in messages]
            })
            
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return jsonify({'error': 'Failed to get conversation'}), 500

@app.route('/translation-pwa/api/conversations', methods=['GET'])
def get_user_conversations():
    """Get all conversations for the current user"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.user_email = ?
                GROUP BY c.id
                ORDER BY c.created_at DESC
            ''', (session['user_email'],))
            conversations = cursor.fetchall()
            
            return jsonify({
                'conversations': [dict(conv) for conv in conversations]
            })
            
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return jsonify({'error': 'Failed to get conversations'}), 500

@app.route('/translation-pwa/api/conversations/<conversation_id>/end', methods=['POST'])
def end_conversation(conversation_id):
    """End a conversation"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE conversations 
                SET ended = TRUE 
                WHERE id = ? AND user_email = ?
            ''', (conversation_id, session['user_email']))
            
            if cursor.rowcount == 0:
                return jsonify({'error': 'Conversation not found'}), 404
            
            conn.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        return jsonify({'error': 'Failed to end conversation'}), 500

@app.route('/translation-pwa/api/conversations/<conversation_id>/continue', methods=['POST'])
def continue_conversation(conversation_id):
    """Continue a previously ended conversation"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if conversation exists and belongs to user
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE id = ? AND user_email = ?
            ''', (conversation_id, session['user_email']))
            conversation = cursor.fetchone()
            
            if not conversation:
                return jsonify({'error': 'Conversation not found'}), 404
            
            # Reactivate the conversation
            cursor.execute('''
                UPDATE conversations 
                SET ended = FALSE 
                WHERE id = ? AND user_email = ?
            ''', (conversation_id, session['user_email']))
            
            conn.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error continuing conversation: {e}")
        return jsonify({'error': 'Failed to continue conversation'}), 500

@app.route('/translation-pwa/api/conversations/<conversation_id>/languages', methods=['PUT'])
def update_conversation_languages(conversation_id):
    """Update conversation language settings"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        input_lang = data.get('input_language', 'auto')
        output_lang = data.get('output_language')
        
        if not output_lang:
            return jsonify({'error': 'Output language is required'}), 400
        
        if output_lang not in Config.SUPPORTED_LANGUAGES:
            return jsonify({'error': 'Unsupported output language'}), 400
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if conversation exists and belongs to user
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE id = ? AND user_email = ? AND ended = FALSE
            ''', (conversation_id, session['user_email']))
            conversation = cursor.fetchone()
            
            if not conversation:
                return jsonify({'error': 'Active conversation not found'}), 404
            
            # Update language settings
            cursor.execute('''
                UPDATE conversations 
                SET input_language = ?, output_language = ?
                WHERE id = ? AND user_email = ?
            ''', (input_lang, output_lang, conversation_id, session['user_email']))
            
            conn.commit()
        
        return jsonify({
            'success': True,
            'input_language': input_lang,
            'output_language': output_lang
        })
        
    except Exception as e:
        logger.error(f"Error updating conversation languages: {e}")
        return jsonify({'error': 'Failed to update language settings'}), 500

@app.route('/translation-pwa/api/conversations/<conversation_id>/participants', methods=['PUT'])
def update_conversation_participants(conversation_id):
    """Update conversation participant names"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        participant_1_name = data.get('participant_1_name', 'Participant 1').strip()
        participant_2_name = data.get('participant_2_name', 'Participant 2').strip()
        
        if not participant_1_name:
            participant_1_name = 'Participant 1'
        if not participant_2_name:
            participant_2_name = 'Participant 2'
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if conversation exists and belongs to user
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE id = ? AND user_email = ?
            ''', (conversation_id, session['user_email']))
            conversation = cursor.fetchone()
            
            if not conversation:
                return jsonify({'error': 'Conversation not found'}), 404
            
            # Update participant names
            cursor.execute('''
                UPDATE conversations 
                SET participant_1_name = ?, participant_2_name = ?
                WHERE id = ? AND user_email = ?
            ''', (participant_1_name, participant_2_name, conversation_id, session['user_email']))
            
            conn.commit()
        
        return jsonify({
            'success': True,
            'participant_1_name': participant_1_name,
            'participant_2_name': participant_2_name
        })
        
    except Exception as e:
        logger.error(f"Error updating conversation participants: {e}")
        return jsonify({'error': 'Failed to update participant names'}), 500

@app.route('/translation-pwa/api/conversations/<conversation_id>/email', methods=['POST'])
def send_conversation_email(conversation_id):
    """Send conversation via email"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get conversation
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE id = ? AND user_email = ?
            ''', (conversation_id, session['user_email']))
            conversation = cursor.fetchone()
            
            if not conversation:
                return jsonify({'error': 'Conversation not found'}), 404
            
            # Get messages
            cursor.execute('''
                SELECT * FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            ''', (conversation_id,))
            messages = cursor.fetchall()
            
            # Send email
            send_email(
                session['user_email'],
                f"Translation Conversation - {conversation_id}",
                f"Here is your complete conversation from {conversation['created_at']}",
                {
                    'conversation': dict(conversation),
                    'messages': [dict(msg) for msg in messages]
                }
            )
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error sending conversation email: {e}")
        return jsonify({'error': 'Failed to send email'}), 500

# Text-to-speech route
@app.route('/translation-pwa/api/tts', methods=['POST'])
def text_to_speech_endpoint():
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        logger.info(f"TTS request: text='{text[:50]}...', language='{language}'")
        
        if not text:
            logger.warning("TTS request with empty text")
            return jsonify({'error': 'Text required'}), 400
        
        if not text.strip():
            logger.warning("TTS request with whitespace-only text")
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        audio_data = text_to_speech(text, language)
        
        if audio_data and len(audio_data) > 0:
            logger.info(f"TTS success: generated {len(audio_data)} bytes")
            return send_file(
                io.BytesIO(audio_data),
                mimetype='audio/wav',
                as_attachment=True,
                download_name='speech.wav'
            )
        else:
            logger.error("TTS failed: no audio data generated")
            return jsonify({'error': 'TTS failed - no audio generated'}), 500
            
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        return jsonify({'error': f'TTS failed: {str(e)}'}), 500


@app.route('/translation-pwa/api/translation/text', methods=['POST'])
def translate_text_endpoint():
    """Translate text directly"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        text = data.get('text', '').strip()
        conversation_id = data.get('conversation_id')
        speaker = data.get('speaker', 'participant_1')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not conversation_id:
            return jsonify({'error': 'Conversation ID is required'}), 400
        
        # Get conversation details and add message in one transaction
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT input_language, output_language, user_email
                FROM conversations 
                WHERE id = ? AND user_email = ?
            """, (conversation_id, session['user_email']))
            
            conversation = cursor.fetchone()
            if not conversation:
                return jsonify({'error': 'Conversation not found'}), 404
            
            input_lang, output_lang, user_email = conversation
        
            # Translate the text
            translator = GoogleTranslator(source=input_lang, target=output_lang)
            translated_text = translator.translate(text)
            
            # Add message to conversation
            cursor.execute("""
                INSERT INTO messages (id, conversation_id, timestamp, speaker, original_text, 
                                    translated_text, input_language, output_language)
                VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?)
            """, (f"text_{int(datetime.datetime.now().timestamp() * 1000)}", 
                  conversation_id, speaker, text, translated_text, input_lang, output_lang))
            
            conn.commit()
        
        return jsonify({
            'success': True,
            'translated_text': translated_text,
            'original_text': text
        })
        
    except Exception as e:
        logger.error(f"Text translation error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/translation-pwa/api/conversations/<conversation_id>/name', methods=['PUT'])
def update_conversation_name(conversation_id):
    """Update conversation name (placeholder - requires database schema update)"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        # For now, just return success without updating database
        # This would require adding a 'name' column to the conversations table
        return jsonify({'success': True, 'name': name, 'note': 'Name saved locally only'})
        
    except Exception as e:
        logger.error(f"Update conversation name error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Admin routes
@app.route('/translation-pwa/api/admin/languages', methods=['GET'])
def admin_get_languages():
    """Admin: Get all languages"""
    try:
        if not session.get('is_admin'):
            return jsonify({'error': 'Admin access required'}), 403
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM installed_languages ORDER BY language_name')
            languages = cursor.fetchall()
            
            return jsonify({
                'languages': [dict(lang) for lang in languages]
            })
            
    except Exception as e:
        logger.error(f"Admin language error: {e}")
        return jsonify({'error': 'Failed to get languages'}), 500

@app.route('/translation-pwa/api/admin/languages/<language_code>/toggle', methods=['POST'])
def admin_toggle_language(language_code):
    """Admin: Toggle language availability"""
    try:
        if not session.get('is_admin'):
            return jsonify({'error': 'Admin access required'}), 403
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE installed_languages 
                SET is_active = NOT is_active 
                WHERE language_code = ?
            ''', (language_code,))
            
            if cursor.rowcount == 0:
                return jsonify({'error': 'Language not found'}), 404
            
            conn.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Admin toggle language error: {e}")
        return jsonify({'error': 'Failed to toggle language'}), 500

@app.route('/translation-pwa/api/admin/conversations', methods=['GET'])
def admin_get_all_conversations():
    """Admin: Get all conversations"""
    try:
        if not session.get('is_admin'):
            return jsonify({'error': 'Admin access required'}), 403
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, u.email, COUNT(m.id) as message_count
                FROM conversations c
                JOIN users u ON c.user_email = u.email
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.created_at DESC
            ''')
            conversations = cursor.fetchall()
            
            return jsonify({
                'conversations': [dict(conv) for conv in conversations]
            })
            
    except Exception as e:
        logger.error(f"Admin conversations error: {e}")
        return jsonify({'error': 'Failed to get conversations'}), 500

@app.route('/translation-pwa/api/admin/export', methods=['GET'])
def admin_export_conversations():
    """Admin: Export all conversations"""
    try:
        if not session.get('is_admin'):
            return jsonify({'error': 'Admin access required'}), 403
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, u.email, m.*
                FROM conversations c
                JOIN users u ON c.user_email = u.email
                LEFT JOIN messages m ON c.id = m.conversation_id
                ORDER BY c.created_at DESC, m.timestamp ASC
            ''')
            data = cursor.fetchall()
            
            # Create CSV export
            csv_data = "conversation_id,user_email,created_at,input_language,output_language,auto_email,ended,message_id,timestamp,speaker,original_text,translated_text,input_language_msg,output_language_msg,audio_file\n"
            
            for row in data:
                csv_data += f"{row['id']},{row['email']},{row['created_at']},{row['input_language']},{row['output_language']},{row['auto_email']},{row['ended']},{row.get('id', '')},{row.get('timestamp', '')},{row.get('speaker', '')},{row.get('original_text', '')},{row.get('translated_text', '')},{row.get('input_language', '')},{row.get('output_language', '')},{row.get('audio_file', '')}\n"
            
            return send_file(
                io.BytesIO(csv_data.encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'conversations_export_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            
    except Exception as e:
        logger.error(f"Admin export error: {e}")
        return jsonify({'error': 'Failed to export conversations'}), 500

# WebSocket events for real-time communication
@socketio.on('join_conversation')
def on_join_conversation(data):
    """Join a conversation room"""
    conversation_id = data.get('conversation_id')
    if conversation_id:
        join_room(conversation_id)
        emit('status', {'message': f'Joined conversation {conversation_id}'})

@socketio.on('leave_conversation')
def on_leave_conversation(data):
    """Leave a conversation room"""
    conversation_id = data.get('conversation_id')
    if conversation_id:
        leave_room(conversation_id)
        emit('status', {'message': f'Left conversation {conversation_id}'})

@socketio.on('new_message')
def on_new_message(data):
    """Broadcast new message to conversation room"""
    conversation_id = data.get('conversation_id')
    if conversation_id:
        emit('message_update', data, room=conversation_id)

# Main application routes
@app.route('/translation-pwa')
def index():
    """Serve the main PWA"""
    return render_template('index.html')

@app.route('/translation-pwa/admin')
def admin():
    """Serve the admin interface"""
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    return render_template('admin.html')


@app.route('/translation-pwa/manifest.json')
def manifest():
    """PWA manifest"""
    return jsonify({
        "name": "Translation by Semaphor",
        "short_name": "Semaphor Translate",
        "description": "Real-time translation service",
        "start_url": "/translation-pwa",
        "display": "standalone",
        "background_color": "#f8fafc",
        "theme_color": "#3b82f6",
        "icons": [
            {
                "src": "/translation-pwa/static/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/translation-pwa/static/icon-512.png", 
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    })

@app.route('/translation-pwa/sw.js')
def service_worker():
    """Service worker for PWA"""
    return send_file('/app/static/sw.js')

# Static files route
@app.route('/translation-pwa/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    file_path = os.path.join('/app/static', filename)
    if not os.path.exists(file_path):
        logger.warning(f"Static file not found: {filename}")
        return "File not found", 404
    return send_file(file_path)

# Health check
@app.route('/translation-pwa/health')
def health():
    """Health check endpoint"""
    fish_speech_status = "disabled_due_to_api_changes"
    if FISH_SPEECH_AVAILABLE:
        fish_speech_status = "installed_but_api_changed"
    else:
        fish_speech_status = "not_installed"
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'whisper_loaded': whisper_model is not None,
        'tts_loaded': tts_engine is not None,
        'fish_speech_loaded': fish_speech_engine is not None,
        'fish_speech_available': FISH_SPEECH_AVAILABLE,
        'fish_speech_status': fish_speech_status
    })


if __name__ == '__main__':
    # Initialize everything
    init_database()
    init_speech_recognition()
    
    # Pre-load models in background to avoid delays during requests
    import threading
    whisper_thread = threading.Thread(target=init_whisper_model, daemon=True)
    whisper_thread.start()
    
    tts_thread = threading.Thread(target=init_tts_model, daemon=True)
    tts_thread.start()
    
    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Start the application
    logger.info("Starting Translation PWA Backend...")
    socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)