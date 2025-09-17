{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python311;
  pythonPackages = python.pkgs;
  
  # Custom Python package overrides for better compatibility
  customPythonPackages = pythonPackages.override {
    overrides = self: super: {
      # Override packages that might have issues
      pyttsx3 = super.pyttsx3.overridePythonAttrs (old: {
        buildInputs = (old.buildInputs or []) ++ [ pkgs.espeak ];
      });
      
      SpeechRecognition = super.SpeechRecognition.overridePythonAttrs (old: {
        buildInputs = (old.buildInputs or []) ++ [ pkgs.portaudio ];
      });
      
      pydub = super.pydub.overridePythonAttrs (old: {
        buildInputs = (old.buildInputs or []) ++ [ pkgs.ffmpeg ];
      });
    };
  };
  
  # Application dependencies
  appDependencies = with customPythonPackages; [
    flask
    flask-socketio
    flask-cors
    openai-whisper
    argostranslate
    pydub
    torch
    torchaudio
    numpy
    pyttsx3
    SpeechRecognition
    ffmpeg-python
    werkzeug
    python-socketio
    eventlet
  ];
  
in pkgs.stdenv.mkDerivation {
  pname = "translation-pwa";
  version = "1.0.0";
  
  src = ./.;
  
  buildInputs = with pkgs; [
    python
    ffmpeg
    espeak
    espeak-data
    libespeak
    portaudio
    alsa-lib
    libsndfile
    libsamplerate
    git
    curl
  ];
  
  propagatedBuildInputs = appDependencies;
  
  installPhase = ''
    mkdir -p $out
    cp -r . $out/
    
    # Create a wrapper script
    cat > $out/bin/translation-pwa << EOF
#!/bin/bash
export PYTHONPATH="$out:$out/backend:\$PYTHONPATH"
export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.portaudio pkgs.alsa-lib pkgs.libsndfile pkgs.espeak]}:$LD_LIBRARY_PATH"
    export ESPEAK_DATA_PATH="${pkgs.espeak}/share/espeak-data"
export FFMPEG_BINARY="${pkgs.ffmpeg}/bin/ffmpeg"
cd $out
exec ${python}/bin/python backend/app.py
EOF
    chmod +x $out/bin/translation-pwa
  '';
  
  meta = with pkgs.lib; {
    description = "Translation PWA - Real-time translation service for Ballerup";
    homepage = "https://github.com/ballerup/translation-pwa";
    license = licenses.mit;
    maintainers = [ ];
    platforms = platforms.linux;
  };
}