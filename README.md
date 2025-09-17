# Translation PWA

A comprehensive Progressive Web Application (PWA) for real-time translation services, designed specifically for Ballerup municipality. This application provides voice-to-voice translation capabilities with conversation management, email integration, and administrative features.

## 🌟 Features

### Core Functionality
- **Real-time Translation**: Convert speech to text, translate, and convert back to speech
- **Multi-language Support**: Danish, English, French, Spanish, Portuguese, Italian, Urdu/Pakistani, Turkish, Serbian, Polish, Ukrainian, Hindi/Indian, and Filipino
- **Auto-detection**: Automatically detect input language or manually select
- **Conversation Management**: Save and manage translation conversations
- **Email Integration**: Send conversations via email automatically or on-demand

### User Features
- **Ballerup Email Authentication**: Secure login with @ballerup.dk email addresses
- **Mobile-responsive Design**: Optimized for Android and iOS devices
- **PWA Capabilities**: Install as a native app on mobile devices
- **Voice Recording**: Record audio directly in the browser
- **Text-to-Speech**: Play translations in the target language
- **Conversation History**: View and manage past conversations
- **Language Swapping**: Easily swap input and output languages

### Admin Features
- **Language Management**: Enable/disable available languages
- **Conversation Monitoring**: View all user conversations
- **Data Export**: Export conversation data for analysis
- **System Status**: Monitor application health and performance
- **User Management**: View user statistics and activity

## 🏗️ Architecture

### Backend
- **Flask**: Web framework with Socket.IO for real-time communication
- **Whisper**: OpenAI's speech-to-text model for audio transcription
- **Argos Translate**: Offline translation engine
- **SQLite**: Database for conversation and user management
- **pyttsx3**: Text-to-speech synthesis
- **Email Integration**: SMTP support for conversation delivery

### Frontend
- **Progressive Web App**: Service worker for offline capabilities
- **Responsive Design**: Mobile-first approach with modern CSS
- **Real-time Updates**: WebSocket integration for live communication
- **Audio Recording**: Browser MediaRecorder API for voice capture

## 🚀 Installation

### Prerequisites
- Python 3.11+
- FFmpeg
- espeak (for text-to-speech)
- Docker (optional)
- Nix (recommended for development)

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

   The application will be available at http://localhost:5000

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
   - Main app: http://localhost:5000
   - Admin panel: http://localhost:5000/admin

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

## 📱 Usage

### For Users

1. **Access the application** at http://localhost:5000
2. **Sign up/Login** with your @ballerup.dk email address
3. **Configure translation settings**:
   - Select input language (or auto-detect)
   - Select output language
   - Enable auto-email if desired
4. **Start a conversation** and begin recording
5. **View translations** in real-time
6. **Manage conversations** through the history section

### For Administrators

1. **Access the admin panel** at http://localhost:5000/admin
2. **Monitor system status** and user activity
3. **Manage languages** - enable/disable as needed
4. **Export conversation data** for analysis
5. **View user statistics** and conversation metrics

## 🔧 API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `POST /api/auth/logout` - User logout
- `GET /api/auth/status` - Check authentication status

### Conversations
- `POST /api/conversations` - Create new conversation
- `GET /api/conversations` - Get user conversations
- `GET /api/conversations/{id}` - Get specific conversation
- `POST /api/conversations/{id}/messages` - Add message to conversation
- `POST /api/conversations/{id}/end` - End conversation
- `POST /api/conversations/{id}/email` - Send conversation via email

### Languages
- `GET /api/languages` - Get available languages

### Text-to-Speech
- `POST /api/tts` - Convert text to speech

### Admin
- `GET /api/admin/languages` - Get all languages (admin)
- `POST /api/admin/languages/{code}/toggle` - Toggle language status
- `GET /api/admin/conversations` - Get all conversations (admin)
- `GET /api/admin/export` - Export conversation data

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

## 🚀 Deployment

### Production Deployment

1. **Set up a production server** with the required dependencies
2. **Configure environment variables** for production
3. **Set up reverse proxy** (nginx recommended)
4. **Configure SSL certificates** for HTTPS
5. **Set up email server** for SMTP functionality
6. **Configure backup strategy** for database and uploads

### Nginx Configuration Example

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
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

## 🔄 Updates

The application is designed to be self-contained and can be updated by:
1. Pulling the latest code
2. Updating dependencies
3. Restarting the service
4. Database migrations are handled automatically

---

**Translation PWA** - Bringing people together through technology 🌍