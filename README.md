# TARS-Mini

A desk-sized voice assistant modeled after TARS from *Interstellar*. Runs on a Raspberry Pi 3B. Listens for a wake word, transcribes speech, calls Claude, speaks the reply through a Bluetooth speaker.

**Persona:** 90% honesty, 75% humor. Short responses. No pleasantries.

---

## Architecture

Two modes — pick one:

### Standalone (recommended for Pi 4/5)
Everything runs on the Pi itself.

```
Mic → openWakeWord → Silero VAD → faster-whisper → Claude → Piper TTS → BT speaker
```

**File:** `tars_v2.py`

### Client/Server (recommended for Pi 3B)
The Pi is a thin audio bridge. All AI runs on a more powerful computer on the same LAN.

```
Pi mic → TCP → Server (Whisper + Claude + Piper) → TCP → Pi speaker
```

**Files:** `tars_pi.py` (Pi) + `tars_server.py` (server)

---

## Hardware

- Raspberry Pi 3B or newer
- I2S MEMS mic or USB mic
- Bluetooth speaker (tested: Sony SRS-XB23 via BlueALSA SCO)

---

## Quick Start

### 1. Set up the Pi

```bash
# Clone and run setup
git clone https://github.com/2016judea/tars-mini
cd tars-mini
chmod +x setup.sh && ./setup.sh
```

`setup.sh` installs system packages, creates a venv, installs Python deps, and downloads Piper.

### 2. Get a wake word model

**Option A — Train "Hey TARS" (recommended):**
Use the [openWakeWord Colab notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb). Copy the `.tflite` here and set `WAKE_WORD_MODEL` in `tars_v2.py`.

**Option B — Use a pre-built placeholder:**
```bash
python -c "import openwakeword; openwakeword.utils.download_models()"
```
Then set `WAKE_WORD_MODEL = Path("hey_jarvis.tflite")` temporarily.

### 3. Download the Piper voice model

```bash
# Already done by setup.sh, but if needed manually:
wget https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz
tar -xzf voice-en-us-lessac-medium.tar.gz
```

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run

```bash
source .venv/bin/activate
python tars_v2.py
```

---

## Client/Server Setup

**On the server (Windows/Mac/Linux):**

```bash
pip install -r requirements_server.txt
# Place piper binary next to tars_server.py
export ANTHROPIC_API_KEY=sk-ant-...
python tars_server.py
```

**On the Pi:**

Edit `SERVER_HOST` in `tars_pi.py` to your server's LAN IP, then:

```bash
pip install -r requirements_pi.txt
python tars_pi.py
```

**Deploy from your computer to the Pi:**

```bash
# Configure TARS_HOST if needed
export TARS_HOST=pi@192.168.0.20
./deploy.sh
```

---

## Configuration

All constants are at the top of each script. Key ones:

| Variable | Default | Description |
|---|---|---|
| `WAKE_WORD_MODEL` | `hey_jarvis.tflite` | Path to .tflite wake word model |
| `WAKE_WORD_THRESHOLD` | `0.5` | Raise to reduce false triggers |
| `WHISPER_MODEL_SIZE` | `tiny` | `base` is better if Pi has headroom |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model |
| `MAX_TOKENS` | `256` | Short replies = lower TTS latency |
| `ALSA_OUTPUT_DEVICE` | BlueALSA SCO | Change to `default` for 3.5mm jack |
| `MIC_DEVICE_INDEX` | `None` (system default) | Set to int for a specific device |

Persona dials (`_honesty`, `_humor`) can be changed at runtime by saying *"set honesty to 70"* — TARS parses `SETTING:X:N` sentinels in its own output.

---

## Test Without Hardware

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python test_prompt.py
```

Interactive REPL — no mic, no speaker, no wake word. Good for tuning the persona.

---

## Files

| File | Purpose |
|---|---|
| `tars_v2.py` | Standalone Pi build — full pipeline |
| `tars_server.py` | Server-side AI pipeline |
| `tars_pi.py` | Pi audio bridge (client/server mode) |
| `test_prompt.py` | Local persona REPL, no hardware |
| `setup.sh` | One-time Pi setup |
| `deploy.sh` | Sync to Pi + restart |
| `requirements.txt` | Standalone Pi deps |
| `requirements_pi.txt` | Thin client Pi deps |
| `requirements_server.txt` | Server deps |

---

## Dependencies

- [openWakeWord](https://github.com/dscripka/openWakeWord) — wake word detection
- [Silero VAD](https://github.com/snakers4/silero-vad) — end-of-utterance detection
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — ASR
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) — LLM
- [Piper](https://github.com/rhasspy/piper) — TTS (persistent process, not per-sentence respawn)
- BlueALSA — Bluetooth audio on Pi (SCO profile)
