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

# Coqui XTTS-v2 - High-quality multilingual TTS
try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import urllib.parse
# requests removed - no longer using external DR TTS API

# Configure logging with immediate flush
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    stream=sys.stderr,  # Use stderr for immediate output
    force=True
)
logger = logging.getLogger(__name__)

# Force flush after every log
import functools
original_info = logger.info
original_warning = logger.warning
original_error = logger.error

def flush_log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        sys.stderr.flush()
        return result
    return wrapper

logger.info = flush_log(original_info)
logger.warning = flush_log(original_warning)
logger.error = flush_log(original_error)

# Log Coqui XTTS-v2 status
if COQUI_TTS_AVAILABLE:
    logger.info("Coqui XTTS-v2 detected - high-quality multilingual TTS available")
else:
    logger.warning("Coqui XTTS-v2 not available")

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
coqui_tts_engine = None
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

def init_coqui_tts():
    """Initialize Coqui XTTS-v2 engine"""
    global coqui_tts_engine
    
    logger.info(f"init_coqui_tts() called - COQUI_TTS_AVAILABLE={COQUI_TTS_AVAILABLE}, coqui_tts_engine={coqui_tts_engine}")
    
    if not COQUI_TTS_AVAILABLE:
        logger.warning("Coqui XTTS-v2 not available, skipping initialization")
        return
    
    if coqui_tts_engine is not None:
        logger.info("Coqui XTTS-v2 engine already initialized, skipping re-initialization")
        return
    
    try:
        logger.info("Initializing Coqui TTS engine...")
        
        # Import torch for GPU detection
        import torch
        
        # Set environment variables for license acceptance BEFORE importing TTS
        os.environ['COQUI_TTS_LICENSE_ACCEPTED'] = 'true'
        os.environ['TTS_LICENSE_ACCEPTED'] = 'true'
        os.environ['TTS_LICENSE_ACCEPTED_CPML'] = 'true'
        
        # Also set the specific model cache directory
        os.environ['TTS_HOME'] = '/app/.cache/tts'
        
        # Create cache directory if it doesn't exist
        os.makedirs('/app/.cache/tts', exist_ok=True)
        
        # Create license acceptance files
        license_files = [
            '/app/.cache/tts/.tos_agreed',
            '/app/.cache/tts/tts_models/multilingual/multi-dataset/xtts_v2/.tos_agreed',
            '/app/.cache/tts/coqui/XTTS-v2/.tos_agreed'
        ]
        
        for license_file in license_files:
            os.makedirs(os.path.dirname(license_file), exist_ok=True)
            with open(license_file, 'w') as f:
                f.write('true')
        
        # Try different TTS models in order of preference
        # GPU is now available via --device nvidia.com/gpu=all
        models_to_try = [
            # XTTS-v2 - Best quality with GPU (12GB RAM allocated)
            ("tts_models/multilingual/multi-dataset/xtts_v2", "XTTS-v2 (best quality, 16 languages, GPU)"),
            # YourTTS - Fallback if XTTS-v2 fails
            ("tts_models/multilingual/multi-dataset/your_tts", "YourTTS (multilingual, lightweight)"),
            # Tacotron2 - Second fallback
            ("tts_models/en/ljspeech/tacotron2-DDC", "Tacotron2 DDC (English only)"),
        ]
        
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TTS will use device: {device}")
        
        model_loaded = False
        for model_name, model_desc in models_to_try:
            try:
                logger.info(f"Trying to load {model_desc}...")
                
                # Initialize TTS with GPU if available
                if device == "cuda":
                    coqui_tts_engine = TTS(model_name=model_name, progress_bar=False, gpu=True)
                    logger.info(f"✅ Successfully loaded {model_desc} on GPU")
                else:
                    coqui_tts_engine = TTS(model_name=model_name, progress_bar=False)
                    logger.info(f"✅ Successfully loaded {model_desc} on CPU")
                
                model_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_desc}: {str(e)[:200]}")
                continue
        
        if not model_loaded:
            raise Exception("Failed to load any TTS model")
        
        logger.info("TTS engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Coqui XTTS-v2: {e}")
        logger.error("Falling back to espeak for TTS")
        coqui_tts_engine = None

def init_tts():
    global tts_engine
    
    print(f">>> INIT_TTS CALLED - tts_engine={tts_engine}", file=sys.stderr, flush=True)
    logger.info(f"init_tts() called - tts_engine={tts_engine}")
    
    # Only initialize once to prevent crashes and improve performance
    if tts_engine is not None:
        logger.info("TTS engine already initialized, skipping re-initialization")
        return
    
    try:
        # Import torch for device detection
        import torch
        
        # Let torch detect GPU availability naturally
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"PyTorch will use device: {device}")
        
        # Initialize Coqui XTTS-v2 for high-quality multilingual TTS
        logger.info("About to call init_coqui_tts()...")
        init_coqui_tts()
        logger.info(f"After init_coqui_tts() - coqui_tts_engine is None: {coqui_tts_engine is None}")
        
        # Set primary TTS engine to Coqui XTTS-v2
        if COQUI_TTS_AVAILABLE and coqui_tts_engine is not None:
            tts_engine = 'coqui'
            logger.info("Primary TTS: Coqui XTTS-v2 initialized successfully")
        else:
            logger.warning("Coqui XTTS-v2 not available, falling back to espeak")
            tts_engine = 'espeak'
        
    except Exception as e:
        logger.warning(f"Coqui XTTS-v2 initialization failed: {e}")
        tts_engine = 'espeak'
    
    # Final fallback - use espeak directly (more reliable in containers)
    if tts_engine != 'coqui':
        try:
            # Test if espeak is available
            import subprocess
            result = subprocess.run(['espeak', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                tts_engine = 'espeak'
                logger.info("Fallback TTS: Using espeak directly")
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
    global tts_engine, coqui_tts_engine
    if tts_engine is None:
        logger.info("Pre-loading TTS model during startup...")
        try:
            init_tts()
            logger.info("TTS model pre-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to pre-load TTS model: {e}")
            # Don't fail startup, just log the error
            logger.warning("TTS model will be loaded on first request")
    
    # Skip Coqui XTTS-v2 initialization during startup to avoid download conflicts
    # It will be loaded on first request instead
    logger.info("Skipping Coqui XTTS-v2 pre-loading to avoid download conflicts")
    logger.info("Coqui XTTS-v2 will be loaded on first TTS request")

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

def coqui_tts(text: str, language: str) -> bytes:
    """Convert text to speech using Coqui XTTS-v2 - high-quality multilingual TTS"""
    try:
        logger.info(f"Using Coqui XTTS-v2 for text: {text[:50]}... in language: {language}")
        
        if not COQUI_TTS_AVAILABLE:
            logger.warning("Coqui XTTS-v2 not available, falling back to espeak")
            return b''
        
        # Language mapping for XTTS-v2 (supports 16 languages natively)
        # For unsupported languages, we map to the closest supported language
        coqui_languages = {
            # Natively supported by XTTS-v2
            'en': 'en', 'english': 'en',
            'es': 'es', 'spanish': 'es',
            'fr': 'fr', 'french': 'fr',
            'de': 'de', 'german': 'de',
            'it': 'it', 'italian': 'it',
            'pt': 'pt', 'portuguese': 'pt',
            'pl': 'pl', 'polish': 'pl',
            'tr': 'tr', 'turkish': 'tr',
            'ru': 'ru', 'russian': 'ru',
            'nl': 'nl', 'dutch': 'nl',
            'cs': 'cs', 'czech': 'cs',
            'ar': 'ar', 'arabic': 'ar',
            'zh': 'zh', 'chinese': 'zh',
            'ja': 'ja', 'japanese': 'ja',
            'hu': 'hu', 'hungarian': 'hu',
            'ko': 'ko', 'korean': 'ko',
            
            # Mapped to closest supported language (XTTS-v2 will pronounce correctly)
            'da': 'de', 'danish': 'de', 'da-dk': 'de',  # Danish → German (closer Germanic language)
            'uk': 'ru', 'ukrainian': 'ru',  # Ukrainian → Russian (similar Slavic)
            'sr': 'ru', 'serbian': 'ru',  # Serbian → Russian (similar Slavic)
            'hi': 'en', 'hindi': 'en', 'indian': 'en',  # Hindi → English
            'tl': 'en', 'tagalog': 'en', 'filipino': 'en',  # Filipino/Tagalog → English
            'ur': 'ar', 'urdu': 'ar', 'pakistani': 'ar',  # Urdu → Arabic (similar script/sounds)
        }
        
        coqui_lang = coqui_languages.get(language.lower())
        
        if coqui_lang is None:
            logger.warning(f"Language {language} not mapped for XTTS-v2, using English as fallback")
            coqui_lang = 'en'  # Default to English instead of failing
        
        logger.info(f"Using Coqui XTTS-v2 language: {coqui_lang}")
        
        # Initialize Coqui XTTS-v2 engine if not already done
        global coqui_tts_engine
        if coqui_tts_engine is None:
            logger.info("Initializing Coqui XTTS-v2 engine on first request...")
            try:
                init_coqui_tts()
                if coqui_tts_engine is None:
                    logger.warning("Coqui XTTS-v2 engine initialization failed, falling back to espeak")
                    return b''
            except Exception as e:
                logger.error(f"Failed to initialize Coqui XTTS-v2 engine: {e}")
                logger.info("Falling back to espeak for this request")
                return b''
        
        # Generate audio using Coqui XTTS-v2
        logger.info("Generating audio with Coqui XTTS-v2...")
        
        try:
            import tempfile
            import os
            
            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                try:
                    # Generate audio using Coqui TTS
                    logger.info(f"Generating speech with Coqui TTS for language: {coqui_lang}")
                    
                    # XTTS-v2 is multi-speaker and requires a speaker parameter
                    # Try different approaches based on model requirements
                    try:
                        # Method 1: Try with language and default speaker (for XTTS-v2)
                        # XTTS-v2 has built-in speakers we can use
                        coqui_tts_engine.tts_to_file(
                            text=text,
                            language=coqui_lang,
                            speaker="Claribel Dervla",  # Use a built-in XTTS-v2 speaker
                            file_path=tmp_file.name
                        )
                        logger.info(f"✅ Generated speech with speaker for language: {coqui_lang}")
                    except (TypeError, ValueError) as e:
                        if "speaker" in str(e).lower():
                            # Try alternate speaker names
                            try:
                                coqui_tts_engine.tts_to_file(
                                    text=text,
                                    language=coqui_lang,
                                    speaker="Ana Florence",  # Try different speaker
                                    file_path=tmp_file.name
                                )
                                logger.info(f"✅ Generated speech with alternate speaker")
                            except Exception as e2:
                                # If speaker selection fails, try with speaker_wav
                                logger.info(f"Speaker selection failed: {e2}, trying without speaker")
                                coqui_tts_engine.tts_to_file(
                                    text=text,
                                    language=coqui_lang,
                                    file_path=tmp_file.name
                                )
                        else:
                            # Method 2: Model doesn't support language parameter (English-only models)
                            logger.info(f"Language param not supported, using default: {e}")
                            coqui_tts_engine.tts_to_file(
                                text=text,
                                file_path=tmp_file.name
                            )
                            logger.info("✅ Generated speech using default model voice")
                    
                    if os.path.exists(tmp_file.name) and os.path.getsize(tmp_file.name) > 0:
                        with open(tmp_file.name, 'rb') as f:
                            audio_data = f.read()
                        
                        logger.info(f"Coqui XTTS-v2 generated {len(audio_data)} bytes for language: {coqui_lang}")
                        return audio_data
                    else:
                        logger.error("Coqui XTTS-v2 generated empty audio")
                        return b''
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Coqui XTTS-v2 error: {e}")
            import traceback
            logger.error(f"Coqui XTTS-v2 traceback: {traceback.format_exc()}")
            return b''
                    
    except Exception as e:
        logger.error(f"Coqui XTTS-v2 error: {e}")
        return b''


# DR TTS removed - using Coqui XTTS-v2 for all languages including Danish

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
    """Convert text to speech using Coqui XTTS-v2 for all languages (100% local, open source)"""
    try:
        # Clean and validate input
        clean_text = text.strip()
        if not clean_text:
            logger.warning("Empty text provided to TTS")
            return b''
        
        # Use Coqui XTTS-v2 for ALL languages (including Danish)
        if COQUI_TTS_AVAILABLE and coqui_tts_engine is not None:
            logger.info(f"Using Coqui XTTS-v2 for {language} text")
            try:
                audio_data = coqui_tts(clean_text, language)
                if audio_data and len(audio_data) > 0:
                    logger.info(f"Coqui XTTS-v2 success: generated {len(audio_data)} bytes")
                    return audio_data
                else:
                    logger.warning("Coqui XTTS-v2 returned empty audio data, falling back to espeak")
            except Exception as e:
                logger.error(f"Coqui XTTS-v2 exception: {e}")
                logger.info("Falling back to espeak")
        else:
            logger.warning(f"Coqui XTTS-v2 not available: COQUI_TTS_AVAILABLE={COQUI_TTS_AVAILABLE}, coqui_tts_engine={coqui_tts_engine is not None}")
        
        # Fallback to espeak if Coqui XTTS-v2 failed or is not available
        init_tts()
        
        if tts_engine is None:
            logger.error("TTS engine not available")
            return b''
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
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
        # Suppress warnings for source map files (used for browser debugging)
        if not filename.endswith('.map'):
            logger.warning(f"Static file not found: {filename}")
        return "File not found", 404
    return send_file(file_path)

# Health check
@app.route('/translation-pwa/health')
def health():
    """Health check endpoint"""
    coqui_tts_status = "not_ready"
    if COQUI_TTS_AVAILABLE and coqui_tts_engine is not None:
        coqui_tts_status = "ready"
    elif COQUI_TTS_AVAILABLE:
        coqui_tts_status = "available_but_not_loaded"
    else:
        coqui_tts_status = "not_installed"
    
    tts_engine_type = "unknown"
    if tts_engine == 'coqui':
        tts_engine_type = "coqui"
    elif tts_engine == 'espeak':
        tts_engine_type = "espeak"
    elif hasattr(tts_engine, 'save_to_file'):
        tts_engine_type = "pyttsx3"
    elif tts_engine is None:
        tts_engine_type = "none"
    
    # Override for current setup
    if tts_engine_type == "coqui":
        tts_engine_type = "coqui (XTTS-v2 high-quality multilingual TTS)"
    elif tts_engine_type == "espeak":
        tts_engine_type = "espeak (fallback)"
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'whisper_loaded': whisper_model is not None,
        'tts_loaded': tts_engine is not None,
        'tts_engine_type': tts_engine_type,
        'coqui_tts_loaded': coqui_tts_engine is not None,
        'coqui_tts_available': COQUI_TTS_AVAILABLE,
        'coqui_tts_status': coqui_tts_status
    })

@app.route('/translation-pwa/api/cleanup/end-old-conversations', methods=['POST'])
def cleanup_old_conversations():
    """Cleanup endpoint to end conversations that have been active for too long"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # End conversations that have been active for more than 2 hours
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=2)
            cursor.execute('''
                UPDATE conversations 
                SET ended = TRUE 
                WHERE ended = FALSE 
                AND created_at < ?
            ''', (cutoff_time.isoformat(),))
            
            affected_rows = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleaned up {affected_rows} old conversations")
            
            return jsonify({
                'success': True,
                'ended_conversations': affected_rows
            })
            
    except Exception as e:
        logger.error(f"Error cleaning up old conversations: {e}")
        return jsonify({'error': 'Failed to cleanup old conversations'}), 500


if __name__ == '__main__':
    # Initialize everything
    init_database()
    init_speech_recognition()
    
    # Skip background model loading to avoid download conflicts
    # Models will be loaded on first request instead
    logger.info("Skipping background model loading to avoid conflicts")
    logger.info("Models will be loaded on first request")
    
    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Start the application
    logger.info("Starting Translation PWA Backend...")
    socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)