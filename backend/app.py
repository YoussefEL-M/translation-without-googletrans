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
import argostranslate.package
import argostranslate.translate
from pydub import AudioSegment
import io
import ffmpeg
import pyttsx3
import speech_recognition as sr
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    DATABASE_PATH = '/opt/praktik/translation-pwa/database/conversations.db'
    UPLOAD_FOLDER = '/opt/praktik/translation-pwa/backend/uploads'
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
        'fil': 'Filipino'
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
        'fil': 'filipino'
    }

# Initialize Flask app
app = Flask(__name__, 
           template_folder='/opt/praktik/translation-pwa/templates',
           static_folder='/opt/praktik/translation-pwa/static')
app.config.from_object(Config)

# Initialize extensions
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Global variables for models
whisper_model = None
tts_engine = None
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
        
        conn.commit()
        logger.info("Database initialized successfully")

def load_whisper_model():
    """Load Whisper model on demand"""
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model...")
        try:
            whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            whisper_model = whisper.load_model("tiny", device="cpu")
            logger.info("Whisper model loaded on CPU with tiny model")
    return whisper_model

def init_tts():
    """Initialize text-to-speech engine"""
    global tts_engine
    if tts_engine is None:
        try:
            tts_engine = pyttsx3.init()
            # Configure TTS settings
            voices = tts_engine.getProperty('voices')
            tts_engine.setProperty('rate', 150)  # Speed of speech
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")

def init_speech_recognition():
    """Initialize speech recognition"""
    global recognizer
    if recognizer is None:
        recognizer = sr.Recognizer()
        logger.info("Speech recognition initialized")

def ensure_translation_model(from_code: str, to_code: str):
    """Ensure translation model is installed"""
    try:
        installed_langs = argostranslate.translate.get_installed_languages()
        from_lang = next((l for l in installed_langs if l.code == from_code), None)
        to_lang = next((l for l in installed_langs if l.code == to_code), None)
        
        if from_lang and to_lang:
            try:
                from_lang.get_translation(to_lang)
                logger.info(f"Translation {from_code} → {to_code} already installed")
                return True
            except Exception:
                pass
        
        logger.info(f"Installing translation model from {from_code} to {to_code}...")
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        
        package_to_install = next(
            (p for p in available_packages if p.from_code == from_code and p.to_code == to_code), 
            None
        )
        
        if package_to_install:
            path = package_to_install.download()
            argostranslate.package.install_from_path(path)
            logger.info(f"Installed {from_code} → {to_code} model")
            return True
        else:
            logger.warning(f"No model available for {from_code} → {to_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error ensuring translation model: {e}")
        return False

def transcribe_audio(audio_data: bytes, language: str = None) -> Dict:
    """Transcribe audio using Whisper"""
    try:
        model = load_whisper_model()
        
        # Convert audio bytes to numpy array
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
        audio_array = np.array(audio_segment.get_array_of_samples())
        
        # Convert to float32 and normalize
        if audio_segment.sample_width == 2:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        
        # Transcribe
        transcribe_kwargs = {}
        if language and language != 'auto':
            whisper_lang = Config.WHISPER_LANGUAGES.get(language, language)
            transcribe_kwargs['language'] = whisper_lang
        
        result = model.transcribe(audio_array, **transcribe_kwargs)
        
        return {
            'text': result['text'].strip(),
            'language': result.get('language', language),
            'confidence': result.get('segments', [{}])[0].get('avg_logprob', 0) if result.get('segments') else 0
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {'text': '', 'language': language, 'error': str(e)}

def translate_text(text: str, from_lang: str, to_lang: str) -> str:
    """Translate text using Argos Translate"""
    try:
        if not text.strip():
            return ""
        
        # Ensure translation model is installed
        if not ensure_translation_model(from_lang, to_lang):
            return text  # Return original text if translation fails
        
        installed_langs = argostranslate.translate.get_installed_languages()
        from_language = next((l for l in installed_langs if l.code == from_lang), None)
        to_language = next((l for l in installed_langs if l.code == to_lang), None)
        
        if not from_language or not to_language:
            logger.error(f"Language not found: {from_lang} or {to_lang}")
            return text
        
        translation = from_language.get_translation(to_language)
        translated_text = translation.translate(text)
        
        return translated_text
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

def text_to_speech(text: str, language: str) -> bytes:
    """Convert text to speech"""
    try:
        init_tts()
        
        # Set language-specific voice if available
        voices = tts_engine.getProperty('voices')
        for voice in voices:
            if language in voice.id.lower() or language in voice.name.lower():
                tts_engine.setProperty('voice', voice.id)
                break
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tts_engine.save_to_file(text, tmp_file.name)
            tts_engine.runAndWait()
            
            # Read the generated audio file
            with open(tmp_file.name, 'rb') as f:
                audio_data = f.read()
            
            os.unlink(tmp_file.name)
            return audio_data
            
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
            html_body += """
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
                html_body += f"""
                <tr>
                    <td>{message['timestamp']}</td>
                    <td>{message['speaker']}</td>
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
@app.route('/api/auth/login', methods=['POST'])
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

@app.route('/api/auth/register', methods=['POST'])
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

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    session.clear()
    return jsonify({'success': True})

@app.route('/api/auth/status', methods=['GET'])
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
@app.route('/api/languages', methods=['GET'])
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
@app.route('/api/conversations', methods=['POST'])
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

@app.route('/api/conversations/<conversation_id>/messages', methods=['POST'])
def add_message():
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
        
        # Handle audio file upload
        audio_file = None
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename:
                filename = secure_filename(audio_file.filename)
                audio_path = os.path.join(Config.UPLOAD_FOLDER, filename)
                os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
                audio_file.save(audio_path)
                audio_file = filename
        
        # Transcribe audio if provided
        original_text = ""
        detected_language = conversation['input_language']
        
        if audio_file:
            audio_path = os.path.join(Config.UPLOAD_FOLDER, audio_file)
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            transcription_result = transcribe_audio(audio_data, conversation['input_language'])
            original_text = transcription_result['text']
            detected_language = transcription_result.get('language', conversation['input_language'])
        
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
            ''', (message_id, conversation_id, timestamp, 'user', original_text, 
                  translated_text, detected_language, conversation['output_language'], audio_file))
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
                'detected_language': detected_language,
                'timestamp': timestamp.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        return jsonify({'error': 'Failed to add message'}), 500

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation():
    """Get conversation details and messages"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        conversation_id = request.view_args['conversation_id']
        
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

@app.route('/api/conversations', methods=['GET'])
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

@app.route('/api/conversations/<conversation_id>/end', methods=['POST'])
def end_conversation():
    """End a conversation"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        conversation_id = request.view_args['conversation_id']
        
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

@app.route('/api/conversations/<conversation_id>/email', methods=['POST'])
def send_conversation_email():
    """Send conversation via email"""
    try:
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        conversation_id = request.view_args['conversation_id']
        
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
@app.route('/api/tts', methods=['POST'])
def text_to_speech_endpoint():
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'Text required'}), 400
        
        audio_data = text_to_speech(text, language)
        
        if audio_data:
            return send_file(
                io.BytesIO(audio_data),
                mimetype='audio/wav',
                as_attachment=True,
                download_name='speech.wav'
            )
        else:
            return jsonify({'error': 'TTS failed'}), 500
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({'error': 'TTS failed'}), 500

# Admin routes
@app.route('/api/admin/languages', methods=['GET'])
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

@app.route('/api/admin/languages/<language_code>/toggle', methods=['POST'])
def admin_toggle_language():
    """Admin: Toggle language availability"""
    try:
        if not session.get('is_admin'):
            return jsonify({'error': 'Admin access required'}), 403
        
        language_code = request.view_args['language_code']
        
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

@app.route('/api/admin/conversations', methods=['GET'])
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

@app.route('/api/admin/export', methods=['GET'])
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
@app.route('/')
def index():
    """Serve the main PWA"""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Serve the admin interface"""
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    return render_template('admin.html')

@app.route('/manifest.json')
def manifest():
    """PWA manifest"""
    return jsonify({
        "name": "Translation PWA",
        "short_name": "TranslatePWA",
        "description": "Real-time translation service",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1f2937",
        "theme_color": "#3b82f6",
        "icons": [
            {
                "src": "/static/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/static/icon-512.png", 
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    })

@app.route('/sw.js')
def service_worker():
    """Service worker for PWA"""
    return send_file('/opt/praktik/translation-pwa/static/sw.js')

# Health check
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'whisper_loaded': whisper_model is not None,
        'tts_loaded': tts_engine is not None
    })

if __name__ == '__main__':
    # Initialize everything
    init_database()
    init_tts()
    init_speech_recognition()
    
    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Start the application
    logger.info("Starting Translation PWA Backend...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)