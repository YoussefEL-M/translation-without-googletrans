# Translation PWA

A comprehensive Progressive Web Application (PWA) for real-time translation services, designed specifically for Ballerup emails. This application provides voice-to-voice translation capabilities with conversation management, email integration, and administrative features.

## 🌟 Features

### Core Functionality
- **Real-time Translation**: Convert speech to text, translate, and convert back to speech
- **Multi-language Support**: Danish, English, French, Spanish, Portuguese, Italian, Urdu/Pakistani, Turkish, Serbian, Polish, Ukrainian, Hindi/Indian, Filipino, and Korean
- **Auto-detection**: Automatically detect input language or manually select
- **Conversation Management**: Save and manage translation conversations with participant names
- **Email Integration**: Send conversations via email automatically or on-demand
- **Advanced TTS System**: Multiple TTS engines with intelligent fallback

### User Features
- **Ballerup Email Authentication**: Secure login with @ballerup.dk email addresses
- **Mobile-responsive Design**: Optimized for Android and iOS devices
- **PWA Capabilities**: Install as a native app on mobile devices
- **Voice Recording**: Record audio directly in the browser using MediaRecorder API
- **Text-to-Speech**: Play translations in the target language with multiple TTS engines
- **Conversation History**: View and manage past conversations with search and filtering
- **Language Swapping**: Easily swap input and output languages
- **Participant Management**: Name participants in conversations for better organization
- **Real-time Communication**: WebSocket integration for live updates

### Admin Features
- **Language Management**: Enable/disable available languages
- **Conversation Monitoring**: View all user conversations with filtering and search
- **Data Export**: Export conversation data for analysis (CSV format)
- **System Status**: Monitor application health and performance
- **User Management**: View user statistics and activity
- **TTS Engine Status**: Monitor TTS engine availability and performance

## 🏗️ Architecture

### Backend
- **Flask**: Web framework with Socket.IO for real-time communication
- **Whisper**: OpenAI's speech-to-text model for audio transcription (small/base/tiny models)
- **Deep Translator**: Google Translate integration for text translation
- **SQLite**: Database for conversation and user management
- **Advanced TTS System**: Multiple TTS engines with intelligent fallback
- **Email Integration**: SMTP support for conversation delivery
- **Audio Processing**: FFmpeg integration for audio format conversion

### Frontend
- **Progressive Web App**: Service worker for offline capabilities
- **Responsive Design**: Mobile-first approach with modern CSS and glassmorphism effects
- **Real-time Updates**: WebSocket integration for live communication
- **Audio Recording**: Browser MediaRecorder API for voice capture
- **Side Navigation**: Collapsible conversation history panel
- **Modern UI**: Gradient backgrounds, smooth animations, and intuitive controls

## 🚀 Installation

### Prerequisites
- Python 3.11+
- FFmpeg (for audio processing)
- eSpeak (for text-to-speech fallback)
- PortAudio (for audio I/O)
- Docker (optional)
- Nix (recommended for development)
- CUDA (optional, for GPU acceleration)

### Quick Start with Nix (Recommended)

1. **Enter the development environment**:
   ```bash
   cd /opt/praktik/translation-pwa
   nix-shell
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

3. **Start the application**:
   ```bash
   python backend/app.py
   ```

   Or use the startup script:
   ```bash
   ./start.sh
   ```

   The application will be available at http://localhost:8000

### Quick Start without Nix

If Nix is not available on your system, use the automated setup script:

```bash
cd /opt/praktik/translation-pwa
./setup-without-nix.sh
```

This script will:
- Detect your operating system (Debian/Ubuntu, Red Hat/CentOS, Arch)
- Install all required system dependencies
- Set up Python virtual environment
- Install Python packages
- Initialize the database

After setup, start the application:
```bash
source venv/bin/activate
python backend/app.py
```

### Quick Start with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd /opt/praktik/translation-pwa
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - Main app: http://localhost:8000
   - Admin panel: http://localhost:8000/admin

### Manual Installation

1. **Install system dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y ffmpeg espeak espeak-data libespeak1 libespeak-dev portaudio19-dev python3-pyaudio
   ```

2. **Set up Python environment**:
   ```bash
   cd /opt/praktik/translation-pwa
   python3 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

3. **Initialize the database**:
   ```bash
   python3 -c "
   import sys
   sys.path.append('/opt/praktik/translation-pwa/backend')
   from app import init_database
   init_database()
   "
   ```

4. **Start the application**:
   ```bash
   ./start.sh
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Flask Secret Key (generate a secure random key)
SECRET_KEY=your-secret-key-here-change-this

# SMTP Configuration for email functionality
SMTP_SERVER=your-smtp-server.com
SMTP_PORT=587
SMTP_USERNAME=your-email@ballerup.dk
SMTP_PASSWORD=your-password
FROM_EMAIL=noreply@ballerup.dk

# Database Configuration
DATABASE_PATH=/app/database/conversations.db

# Upload Configuration
UPLOAD_FOLDER=/app/backend/uploads
MAX_CONTENT_LENGTH=16777216
```

### Language Configuration

The application supports the following languages by default:
- Danish (da)
- English (en)
- French (fr)
- Spanish (es)
- Portuguese (pt)
- Italian (it)
- Urdu/Pakistani (ur)
- Turkish (tr)
- Serbian (sr)
- Polish (pl)
- Ukrainian (uk)
- Hindi/Indian (hi)
- Filipino (fil)

Languages can be enabled/disabled through the admin interface.

## 🔊 Text-to-Speech (TTS) System

The application features a sophisticated multi-engine TTS system with intelligent fallback mechanisms:

### TTS Engines

#### 1. **DR TTS Service** (Danish)
- **Primary for Danish**: Uses DR (Danish Broadcasting Corporation) professional TTS service
- **High Quality**: Professional-grade Danish speech synthesis
- **Online Service**: Requires internet connection
- **URL**: `https://www.dr.dk/tjenester/tts`
- **Fallback**: Falls back to Fish Speech TTS if DR service fails

#### 2. **Fish Speech TTS** (Multi-language)
- **Neural TTS**: Advanced neural text-to-speech model
- **Supported Languages**: English, Chinese, German, Japanese, French, Spanish, Korean, Arabic
- **High Quality**: Trained on 700k hours of audio data
- **Offline Capable**: Works without internet connection
- **Model Size**: ~2GB model files in `/fish_speech_models/`
- **API Version**: Fish Speech 0.1.0+ (API has changed significantly)

#### 3. **eSpeak** (Fallback)
- **Universal Fallback**: Available for all supported languages
- **Offline**: Works without internet connection
- **Lightweight**: Minimal resource usage
- **Quality**: Basic but reliable speech synthesis
- **Languages**: Supports 80+ languages including all app languages

#### 4. **pyttsx3** (Last Resort)
- **Cross-platform**: Works on Windows, macOS, Linux
- **System Voices**: Uses system-installed TTS voices
- **Fallback Only**: Used when all other engines fail

### TTS Selection Logic

1. **Danish Text**: DR TTS Service → Fish Speech TTS → eSpeak → pyttsx3
2. **Fish Speech Supported Languages**: Fish Speech TTS → eSpeak → pyttsx3
3. **Other Languages**: eSpeak → pyttsx3

### TTS Configuration

#### Environment Variables
```env
# CUDA settings (optional)
CUDA_VISIBLE_DEVICES=""  # Force CPU usage
TORCH_DEVICE=cpu
FORCE_DEVICE=cpu

# Fish Speech settings
FISH_SPEECH_MODELS_PATH=/app/fish_speech_models
```

#### Model Files
The Fish Speech models are located in `/fish_speech_models/`:
- `model.pth` - Main TTS model
- `config.json` - Model configuration
- `tokenizer.json` - Text tokenizer
- `firefly-gan-vq-fsq-8x1024-21hz-generator.pth` - Audio generator
- `special_tokens_map.json` - Special tokens mapping

### TTS Performance

- **Danish**: ~1-2 seconds (DR TTS service)
- **Fish Speech**: ~2-5 seconds (depending on text length)
- **eSpeak**: ~0.5-1 second (fastest)
- **pyttsx3**: ~1-3 seconds (system dependent)

### Troubleshooting TTS

#### Common Issues
1. **Fish Speech not working**: Check model files in `/fish_speech_models/`
2. **DR TTS failing**: Check internet connection and service availability
3. **eSpeak errors**: Verify eSpeak installation and voice data
4. **Audio not playing**: Check browser audio permissions and codec support

#### Health Check
Monitor TTS engine status via `/translation-pwa/health` endpoint:
```json
{
  "status": "healthy",
  "tts_loaded": true,
  "fish_speech_loaded": true,
  "fish_speech_available": true,
  "fish_speech_status": "installed_but_api_changed"
}
```

## 📱 Progressive Web App (PWA) Installation

The Translation PWA is designed to work seamlessly across all devices and can be installed as a native app on mobile phones, tablets, and desktop computers.

### 🚀 How to Install the App

#### **Automatic Installation (Recommended)**
1. **Visit the app** in your mobile browser: `http://localhost:8000/translation-pwa`
2. **Look for the install button** "📱 Install App" in the top bar
3. **Tap the button** to trigger the browser's native install prompt
4. **Follow the prompts** to add the app to your home screen

#### **Manual Installation Methods**

##### **iPhone/iPad (Safari)**
1. Open the app in Safari
2. Tap the **Share button** (📤) at the bottom of the screen
3. Scroll down and tap **"Add to Home Screen"**
4. Tap **"Add"** to confirm
5. The app will appear on your home screen with a custom icon

##### **Android (Chrome/Edge/Samsung Internet)**
1. **Chrome/Edge**: Open the app in Chrome or Edge
   - Look for **install icon** (⬇️) in address bar, OR
   - Tap **menu** (⋮) → **"Add to Home screen"** or **"Install app"**
2. **Samsung Internet**: Open the app in Samsung Internet
   - Tap **menu** (⋮) → **"Add page to"** → **"Home screen"**
3. Tap **"Add"** or **"Install"** to confirm
4. The app will be added to your home screen

**Note**: Samsung Galaxy users may need to use Chrome for full PWA features. Samsung Internet has limited PWA support.

##### **Desktop (Chrome/Edge/Firefox)**
1. Open the app in a supported browser
2. Look for the **install icon** (⬇️) in the address bar
3. Click the install icon and select **"Install"**
4. The app will be added to your desktop and app menu
5. You can launch it like any other desktop application

### 🎯 PWA Features

#### **App-like Experience**
- **Standalone Mode**: Runs without browser UI (no address bar, tabs, etc.)
- **Custom Icon**: Professional app icon on home screen/desktop
- **App Name**: "Translation by Semaphor" in app lists
- **Splash Screen**: Custom loading screen when opening

#### **Offline Capabilities**
- **Service Worker**: Caches essential files for offline access
- **Cached Resources**: App shell loads instantly even without internet
- **Background Sync**: Queues actions when offline, syncs when online
- **Smart Caching**: Automatically updates cached content

#### **Mobile Optimizations**
- **Touch-Friendly**: Large buttons and touch targets
- **Responsive Design**: Adapts to all screen sizes
- **Fast Loading**: Optimized for mobile networks
- **Battery Efficient**: Minimal background processing

#### **Desktop Features**
- **Window Management**: Resizable, movable windows
- **Keyboard Shortcuts**: Full keyboard navigation support
- **Multi-window**: Can open multiple instances
- **System Integration**: Appears in taskbar/dock

### 🔧 PWA Technical Details

#### **Manifest Configuration**
```json
{
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
}
```

#### **Service Worker Features**
- **Cache Strategy**: Cache-first for static assets, network-first for API calls
- **Background Sync**: Handles offline translation requests
- **Push Notifications**: Ready for future notification features
- **Update Management**: Automatic app updates

#### **Installation Detection**
- **beforeinstallprompt**: Captures browser install prompts
- **appinstalled**: Detects successful installations
- **display-mode**: Detects if running in standalone mode
- **Smart UI**: Shows/hides install button based on installation status

### 📊 PWA Benefits

#### **For Users**
- **Fast Access**: One tap to open from home screen
- **Offline Use**: Works without internet connection
- **Native Feel**: Behaves like a native mobile app
- **No App Store**: Install directly from browser
- **Always Updated**: Automatic updates without user intervention

#### **For Administrators**
- **Easy Deployment**: No app store approval process
- **Instant Updates**: Deploy updates immediately
- **Cross-Platform**: Single codebase for all devices
- **Analytics**: Full web analytics capabilities
- **Cost Effective**: No app store fees or restrictions

### 🛠️ PWA Development

#### **Testing PWA Features**
```bash
# Test PWA installation
curl -I http://localhost:8000/translation-pwa/manifest.json

# Test service worker
curl -I http://localhost:8000/translation-pwa/sw.js

# Check PWA score
# Use Chrome DevTools > Lighthouse > Progressive Web App
```

#### **Debugging PWA Issues**
1. **Chrome DevTools**: Application tab > Service Workers
2. **Manifest**: Check manifest.json is valid
3. **HTTPS**: PWA requires HTTPS in production
4. **Service Worker**: Verify registration and caching
5. **Install Prompt**: Check beforeinstallprompt event

#### **Samsung Galaxy Troubleshooting**
**If you can't find install options on Samsung Galaxy:**

1. **Use Chrome Browser**: Samsung Internet has limited PWA support
2. **Check Address Bar**: Look for install icon (⬇️) or "+" symbol
3. **Try Menu Method**: Chrome menu (⋮) → "Add to Home screen"
4. **Refresh Page**: Sometimes the install prompt appears after refresh
5. **Check Android Version**: PWA support varies by Android version
6. **Disable Ad Blockers**: Some ad blockers prevent PWA installation

**Common Samsung Issues:**
- **Samsung Internet**: Limited PWA features, use Chrome instead
- **Android 7-8**: May need Chrome 68+ for full PWA support
- **Custom ROMs**: Some custom Android versions have PWA limitations

## 📱 Usage

### For Users

1. **Access the application** at http://localhost:8000
2. **Sign up/Login** with your @ballerup.dk email address
3. **Configure translation settings**:
   - Select input language (or auto-detect)
   - Select output language
   - Enable auto-email if desired
4. **Start a conversation** and begin recording
5. **View translations** in real-time
6. **Manage conversations** through the history section

### For Administrators

1. **Access the admin panel** at http://localhost:8000/admin
2. **Monitor system status** and user activity
3. **Manage languages** - enable/disable as needed
4. **Export conversation data** for analysis
5. **View user statistics** and conversation metrics

## 🔧 API Endpoints

All API endpoints are prefixed with `/translation-pwa/api/`

### Authentication
- `POST /translation-pwa/api/auth/login` - User login
- `POST /translation-pwa/api/auth/register` - User registration
- `POST /translation-pwa/api/auth/logout` - User logout
- `GET /translation-pwa/api/auth/status` - Check authentication status
- `POST /translation-pwa/api/auth/make-admin` - Make user admin (with admin key)

### Conversations
- `POST /translation-pwa/api/conversations` - Create new conversation
- `GET /translation-pwa/api/conversations` - Get user conversations
- `GET /translation-pwa/api/conversations/{id}` - Get specific conversation
- `POST /translation-pwa/api/conversations/{id}/messages` - Add message to conversation
- `POST /translation-pwa/api/conversations/{id}/end` - End conversation
- `POST /translation-pwa/api/conversations/{id}/continue` - Continue ended conversation
- `POST /translation-pwa/api/conversations/{id}/email` - Send conversation via email
- `PUT /translation-pwa/api/conversations/{id}/languages` - Update conversation language settings
- `PUT /translation-pwa/api/conversations/{id}/participants` - Update participant names
- `PUT /translation-pwa/api/conversations/{id}/name` - Update conversation name

### Translation
- `POST /translation-pwa/api/translation/text` - Translate text directly

### Languages
- `GET /translation-pwa/api/languages` - Get available languages

### Text-to-Speech
- `POST /translation-pwa/api/tts` - Convert text to speech

### Admin
- `GET /translation-pwa/api/admin/languages` - Get all languages (admin)
- `POST /translation-pwa/api/admin/languages/{code}/toggle` - Toggle language status
- `GET /translation-pwa/api/admin/conversations` - Get all conversations (admin)
- `GET /translation-pwa/api/admin/export` - Export conversation data

### System
- `GET /translation-pwa/health` - Health check and system status
- `GET /translation-pwa/manifest.json` - PWA manifest
- `GET /translation-pwa/sw.js` - Service worker

## 🗄️ Database Schema

### Users Table
- `id` - Primary key
- `email` - User email (unique)
- `password_hash` - Hashed password
- `created_at` - Account creation timestamp
- `last_login` - Last login timestamp
- `is_admin` - Admin flag

### Conversations Table
- `id` - Conversation ID (unique)
- `user_email` - User email (foreign key)
- `created_at` - Conversation start timestamp
- `input_language` - Source language
- `output_language` - Target language
- `auto_email` - Auto-email setting
- `ended` - Conversation status
- `participant_1_name` - Name of first participant
- `participant_2_name` - Name of second participant

### Messages Table
- `id` - Message ID (unique)
- `conversation_id` - Conversation ID (foreign key)
- `timestamp` - Message timestamp
- `speaker` - Message sender (user/system)
- `original_text` - Original transcribed text
- `translated_text` - Translated text
- `input_language` - Detected input language
- `output_language` - Target language
- `audio_file` - Audio file path (optional)

### Installed Languages Table
- `id` - Primary key
- `language_code` - Language code (unique)
- `language_name` - Language display name
- `installed_at` - Installation timestamp
- `is_active` - Active status

## 🔒 Security Features

- **Email Domain Restriction**: Only @ballerup.dk emails allowed
- **Password Hashing**: Secure password storage with Werkzeug
- **Session Management**: Flask sessions for user authentication
- **Admin Access Control**: Role-based access to admin features
- **Input Validation**: Comprehensive input sanitization
- **File Upload Security**: Secure file handling and validation

## 📊 Monitoring and Logging

- **Health Checks**: Built-in health monitoring endpoints
- **System Status**: Real-time system status in admin panel
- **Error Logging**: Comprehensive error logging and handling
- **Performance Metrics**: Conversation and user statistics

## 🛠️ Technologies Used

### Backend Technologies
- **Python 3.11** - Core programming language
- **Flask 2.3.3** - Web framework
- **Flask-SocketIO 5.3.6** - Real-time communication
- **Flask-CORS 4.0.0** - Cross-origin resource sharing
- **SQLite** - Database for conversations and users
- **Werkzeug 2.3.7** - WSGI utilities and security

### AI/ML Technologies
- **OpenAI Whisper** - Speech-to-text transcription
- **PyTorch 2.6.0** - Deep learning framework
- **TorchAudio 2.6.0** - Audio processing
- **Deep Translator** - Google Translate integration
- **Fish Speech** - Neural text-to-speech
- **Transformers 4.21.0** - Hugging Face transformers

### Audio Processing
- **FFmpeg** - Audio/video processing
- **PyDub 0.25.1** - Audio manipulation
- **SpeechRecognition 3.10.0** - Speech recognition
- **eSpeak** - Text-to-speech synthesis
- **PortAudio** - Audio I/O
- **librosa** - Audio analysis

### Frontend Technologies
- **Progressive Web App (PWA)** - Offline capabilities and native app installation
- **Service Worker** - Background processing and caching
- **WebSocket** - Real-time communication
- **MediaRecorder API** - Audio recording
- **Modern CSS** - Responsive design with glassmorphism
- **JavaScript ES6+** - Client-side functionality
- **PWA Installation** - One-click app installation for mobile and desktop

### Development Tools
- **Nix** - Reproducible development environment
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **Podman** - Alternative container runtime

## 🚀 Deployment

### Production Deployment

1. **Set up a production server** with the required dependencies
2. **Configure environment variables** for production
3. **Set up reverse proxy** (nginx recommended)
4. **Configure SSL certificates** for HTTPS
5. **Set up email server** for SMTP functionality
6. **Configure backup strategy** for database and uploads
7. **Set up Fish Speech models** in `/fish_speech_models/` directory
8. **Configure TTS engines** and test audio output

### Nginx Configuration Example

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /static {
        alias /opt/praktik/translation-pwa/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is developed for Ballerup municipality and is intended for internal use.

## 🆘 Support

For technical support or questions:
- Check the admin panel for system status
- Review logs for error details
- Contact the development team

## 📜 Startup Scripts

The application includes several startup scripts for different deployment scenarios:

### Development Scripts
- `start.sh` - Main startup script with environment setup
- `setup-without-nix.sh` - Automated setup for systems without Nix
- `restart-translation-pwa-no-cache.sh` - Restart without Docker cache

### Container Scripts
- `start-translation-pwa-podman.sh` - Podman-based deployment
- `docker-compose.yml` - Docker Compose configuration
- `podman-compose.yml` - Podman Compose configuration

### Nix Scripts
- `shell.nix` - Nix development environment
- `default.nix` - Nix package definition

## 🔄 Updates

The application is designed to be self-contained and can be updated by:
1. Pulling the latest code
2. Updating dependencies (`pip install -r backend/requirements.txt`)
3. Restarting the service
4. Database migrations are handled automatically
5. TTS models are updated independently

## 🧪 Testing

### Health Check
Monitor application health via:
```bash
curl http://localhost:8000/translation-pwa/health
```

### TTS Testing
Test TTS functionality:
```bash
# Test Danish TTS (DR service)
curl -X POST http://localhost:8000/translation-pwa/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hej verden", "language": "da"}'

# Test other languages
curl -X POST http://localhost:8000/translation-pwa/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en"}'
```

### Audio Testing
Test audio recording and transcription:
1. Access the web interface
2. Start a conversation
3. Record audio using the browser
4. Verify transcription and translation

## 📊 Performance

### System Requirements
- **Minimum RAM**: 4GB (8GB recommended)
- **CPU**: Multi-core processor (GPU optional)
- **Storage**: 10GB free space (for models and audio files)
- **Network**: Stable internet for DR TTS and Google Translate

### Performance Metrics
- **Audio Transcription**: ~2-5 seconds (Whisper)
- **Text Translation**: ~0.5-1 second (Google Translate)
- **TTS Generation**: ~1-5 seconds (depending on engine)
- **Database Operations**: <100ms (SQLite)

---

**Translation PWA** - Bringing people together through technology 🌍
