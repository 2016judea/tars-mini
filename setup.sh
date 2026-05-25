#!/usr/bin/env bash
# setup.sh — Run once on a fresh Pi OS to install TARS-Mini dependencies.
# Tested on Raspberry Pi OS Bookworm (64-bit).
set -e

echo "=== TARS-Mini Setup ==="

# ── System packages ────────────────────────────────────────────────────────
echo "[1/5] Installing system packages…"
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-venv \
    portaudio19-dev \    # PyAudio
    alsa-utils \         # aplay for speaker output
    ffmpeg \             # faster-whisper may need it
    wget

# ── Python venv ────────────────────────────────────────────────────────────
echo "[2/5] Creating Python virtual environment…"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

# ── Python packages ────────────────────────────────────────────────────────
echo "[3/5] Installing Python packages (this takes a few minutes on Pi)…"
pip install -r requirements.txt -q

# ── Piper TTS binary ───────────────────────────────────────────────────────
echo "[4/5] Installing Piper TTS…"
PIPER_VERSION="1.2.0"
PIPER_ARCH="aarch64"   # Pi 3B/4/5 64-bit OS; change to armv7l for 32-bit
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${PIPER_ARCH}.tar.gz"

wget -q "$PIPER_URL" -O piper.tar.gz
tar -xzf piper.tar.gz
rm piper.tar.gz

# Add piper to PATH for this session
export PATH="$PATH:$(pwd)/piper"

# Download the voice model
VOICE_URL="https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz"
wget -q "$VOICE_URL" -O voice.tar.gz
tar -xzf voice.tar.gz
rm voice.tar.gz
# Produces en_US-lessac-medium.onnx and en_US-lessac-medium.onnx.json

echo "Piper installed. Add $(pwd)/piper to your PATH in ~/.bashrc"

# ── Wake word model ────────────────────────────────────────────────────────
echo "[5/5] Wake word model…"
echo ""
echo "  IMPORTANT: You need a wake word model for 'Hey TARS'."
echo "  Options:"
echo ""
echo "  A) Train a custom 'Hey TARS' model (recommended, takes ~30 min on a laptop):"
echo "     https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb"
echo "     Then copy the .tflite file here and update WAKE_WORD_MODEL in tars.py."
echo ""
echo "  B) Use a pre-built model as a placeholder (not 'Hey TARS' but functional):"
echo "     python -c \"import openwakeword; openwakeword.utils.download_models()\""
echo "     This downloads 'hey_jarvis', 'alexa', etc. Set WAKE_WORD_MODEL accordingly."
echo ""

# ── API key reminder ───────────────────────────────────────────────────────
echo "=== Setup complete ==="
echo ""
echo "Before running tars.py, set your API key:"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "Or add it to ~/.bashrc:"
echo "  echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc"
echo ""
echo "Then run:"
echo "  source .venv/bin/activate"
echo "  python tars.py"
