# TARS-Mini — Claude Code Brief

## What this is

TARS-Mini is a desk-scale replica of the TARS robot from *Interstellar*. It runs on a
Raspberry Pi 3B and uses the Anthropic API as its brain. You are working on the
software side: a Python voice pipeline that listens for a wake word, transcribes
speech, calls Claude, and speaks the response through a speaker.

The hardware (Pi, mic, speaker) lives at a remote address reachable over SSH. You
will edit code locally and deploy to the Pi via SSH. The human will test by talking
to the physical robot and reading the Pi's log output streamed back to your terminal.

---

## Repository layout

```
tars-mini/
├── CLAUDE.md          ← this file
├── tars_v2.py         ← main voice pipeline (the file you will edit most)
├── requirements.txt   ← Python dependencies
├── deploy.sh          ← rsync + restart helper (run this after every change)
└── setup.sh           ← one-time Pi provisioning (run once, not repeatedly)
```

---

## SSH / deployment workflow

**Pi is at `pi@192.168.0.20`.** SSH key auth is already configured.

**All deployment is handled by `deploy.sh`.** After editing any file, run:

```bash
./deploy.sh
```

This script will:
1. `scp` the project files to `~/tars-mini/` on the Pi
2. SSH in and `pkill` the running tars process (if any)
3. Restart `tars_v2.py` inside a `tmux` session named `tars`
4. Tail the log so you can watch it live

To watch the live log at any time without redeploying:
```bash
ssh pi@192.168.0.20 "tail -f ~/tars-mini/tars.log"
```

To stop TARS:
```bash
ssh pi@192.168.0.20 "pkill -f tars_v2.py"
```

### Windows SSH notes

- **`rsync` is not available** in Git Bash on this machine — `deploy.sh` uses `scp` instead.
- Git Bash `ssh` requires legacy compat flags for the Pi's older OpenSSH (7.4). These are
  already set in `~/.ssh/config` for `192.168.0.20` — do not remove that entry.
- If you need to SSH manually, just `ssh pi@192.168.0.20` works fine from Git Bash or PowerShell.
- PowerShell's `ssh` (Windows OpenSSH) also works as a fallback if Git Bash misbehaves.

---

## Architecture overview

```
Mic (I2S/USB)
    │
    ▼
_audio_reader_thread         — single PyAudio stream; pushes chunks to _audio_queue
    │
    ▼
wait_for_wake_word()         — consumes queue; openWakeWord scores each 80ms chunk
    │  (score ≥ threshold)
    ▼
record_utterance()           — consumes queue; Silero VAD detects end-of-speech
    │
    ▼
transcribe()                 — faster-whisper tiny model; returns text string
    │
    ▼
stream_response()            — Anthropic claude-sonnet-4-6, streaming, 8-turn history
    │                          parses SETTING:HONESTY/HUMOR sentinel lines
    ▼
speak_stream()               — splits on sentence boundaries (≥12 chars)
    │
    ▼
PiperTTS.speak()             — persistent Piper process; plays via aplay
    │
    ▼
Speaker (USB / ALSA)
```

**Key design decisions to be aware of:**
- One shared `_audio_queue`. The wake word listener and utterance recorder both pull
  from it — never open a second PyAudio stream on the same device.
- Piper runs as a persistent subprocess (not respawned per sentence) to avoid
  ~300ms model-load overhead on each sentence.
- Persona dials (honesty/humor) are changed via LLM sentinel lines, not regex on
  user speech. The system prompt instructs Claude to emit `SETTING:HUMOR:N` lines
  which `stream_response()` intercepts and strips before TTS.
- History is trimmed to 8 turns and always starts with a `user` message (Anthropic
  API requirement).

---

## Configuration reference

All tunable constants are at the top of `tars_v2.py`. The ones you'll touch most:

| Constant | Default | What it does |
|---|---|---|
| `WAKE_WORD_MODEL` | `hey_tars.tflite` | Path to openWakeWord .tflite model |
| `WAKE_WORD_THRESHOLD` | `0.5` | Raise if false triggers; lower if misses |
| `SILERO_SPEECH_THRESHOLD` | `0.5` | VAD sensitivity for end-of-utterance |
| `SILENCE_DURATION_MS` | `1200` | ms of silence before recording stops |
| `WHISPER_MODEL_SIZE` | `"tiny"` | `"base"` is more accurate but slower |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model alias |
| `MAX_TOKENS` | `256` | Keep short for TTS latency |
| `PIPER_MODEL` | `en_US-lessac-medium.onnx` | Path to Piper voice model |
| `REARM_DELAY_S` | `1.5` | Dead time after speaking before re-listening |
| `TTS_MIN_CHARS` | `12` | Min chars before flushing a TTS sentence |

---

## Environment variables (set on the Pi)

```bash
ANTHROPIC_API_KEY=sk-ant-...    # required
```

`deploy.sh` will check this is set in `/etc/environment` or `~/.bashrc` on the Pi
before starting. Add it once with:
```bash
ssh $TARS_HOST "echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc"
```

---

## Wake word model

The default config expects `hey_tars.tflite` in the project directory. You have two
options:

**Option A — use a placeholder while developing (recommended first step):**
```bash
ssh $TARS_HOST "cd ~/tars-mini && \
  python -c 'import openwakeword; openwakeword.utils.download_models()'"
```
Then in `tars_v2.py` change `WAKE_WORD_MODEL = Path("hey_jarvis.tflite")` temporarily.

**Option B — train a custom "Hey TARS" model:**
Use the openWakeWord training Colab notebook:
https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb
Copy the resulting `.tflite` to the Pi project directory and set `WAKE_WORD_MODEL`.

---

## Piper TTS voice model

Download the voice model once:
```bash
ssh $TARS_HOST "cd ~/tars-mini && \
  wget https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-lessac-medium.tar.gz && \
  tar -xzf voice-en-us-lessac-medium.tar.gz && \
  rm voice-en-us-lessac-medium.tar.gz"
```

---

## Implementation plan

Work through these phases in order. Each phase ends with a deployable, testable state.

### Phase 0 — SSH + deploy pipeline (do this first)
- [ ] Confirm SSH key auth works: `ssh pi@tars.local`
- [ ] Edit `TARS_HOST` in `deploy.sh`
- [ ] Run `./setup.sh` once on the Pi (installs system packages, venv, Piper binary)
- [ ] Run `./deploy.sh` — confirm files land at `~/tars-mini/` on the Pi
- [ ] Confirm `tmux` session `tars` starts and you can tail the log

### Phase 1 — Audio hardware validation (no LLM yet)
Goal: confirm mic and speaker work before touching the AI stack.

- [ ] Run `arecord -l` on the Pi to list capture devices; set `MIC_DEVICE_INDEX` if needed
- [ ] Run `aplay -l` to list playback devices; confirm default device plays audio
- [ ] Test mic capture: `arecord -d 3 -f S16_LE -r 16000 test.wav && aplay test.wav`
- [ ] Confirm Whisper transcribes the test file:
  ```bash
  python -c "
  from faster_whisper import WhisperModel
  m = WhisperModel('tiny', device='cpu', compute_type='int8')
  segs, _ = m.transcribe('test.wav', language='en')
  print([s.text for s in segs])
  "
  ```

### Phase 2 — Silero VAD + recording smoke test
Goal: verify `record_utterance()` captures speech and stops correctly.

- [ ] Add a temporary `__main__` test block to `tars_v2.py`:
  ```python
  # Temporary test — remove when done
  if __name__ == "__test__":
      reader = threading.Thread(target=_audio_reader_thread, daemon=True)
      reader.start()
      print("Speak something...")
      audio = record_utterance()
      if audio:
          with open("utterance.wav", "wb") as f:
              import wave
              wf = wave.open(f, "wb")
              wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
              wf.writeframes(audio)
          print("Saved utterance.wav — play it back to verify")
      else:
          print("No speech detected")
  ```
- [ ] Deploy and run, speak, verify the saved WAV sounds right
- [ ] Tune `SILERO_SPEECH_THRESHOLD` and `SILENCE_DURATION_MS` if cutoff timing is off

### Phase 3 — Wake word integration
Goal: the wake word reliably triggers a recording.

- [ ] Confirm wake word model file is present on the Pi
- [ ] Run full `tars_v2.py` — watch log for wake word score on each chunk
- [ ] Verify it triggers on "Hey TARS" (or placeholder word) and not on background noise
- [ ] Tune `WAKE_WORD_THRESHOLD` if needed

### Phase 4 — LLM + TTS end-to-end
Goal: a complete spoken conversation turn with no hardware errors.

- [ ] Confirm `ANTHROPIC_API_KEY` is set on the Pi
- [ ] Confirm Piper model `.onnx` file is present
- [ ] Run full pipeline; speak a question after wake word
- [ ] Verify TARS responds in character and audio plays through speaker
- [ ] Check log for any API errors or TTS pipe errors
- [ ] Test persona change: say "set humor to 40", verify log shows persona update

### Phase 5 — Tuning and polish
- [ ] Tune `REARM_DELAY_S` for your room acoustics (increase if TTS re-triggers wake word)
- [ ] Tune `MAX_TOKENS` vs response quality tradeoff
- [ ] Tune `TTS_MIN_CHARS` for natural sentence pacing
- [ ] Consider `WHISPER_MODEL_SIZE = "base"` if transcription accuracy is poor
- [ ] Add `systemd` service so TARS starts on Pi boot (see note below)

**Optional: systemd auto-start**
```ini
# /etc/systemd/system/tars.service
[Unit]
Description=TARS-Mini voice assistant
After=network.target sound.target

[Service]
User=pi
WorkingDirectory=/home/pi/tars-mini
EnvironmentFile=/home/pi/.bashrc
ExecStart=/home/pi/tars-mini/.venv/bin/python tars_v2.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: [Errno -9996]` on startup | No mic found at default device | Set `MIC_DEVICE_INDEX` to correct int from `arecord -l` |
| Wake word never triggers | Threshold too high or wrong model | Lower `WAKE_WORD_THRESHOLD`; verify model path |
| Wake word triggers constantly | Threshold too low or speaker feedback | Raise threshold; check speaker isn't feeding mic |
| Recording never stops | Silero VAD too sensitive | Lower `SILERO_SPEECH_THRESHOLD`; check mic noise floor |
| Piper pipe error on first sentence | Piper binary not on PATH | Run `which piper`; update `PIPER_BINARY` with full path |
| `anthropic.APIError` in log | Network issue or bad API key | Check `ANTHROPIC_API_KEY`; test with `curl` |
| TTS re-triggers wake word | Room reverb / speaker too loud | Increase `REARM_DELAY_S`; reduce speaker volume |
| History API error (role error) | `_trim()` left orphaned assistant message | Should be fixed in v2; if recurs, clear history with restart |

---

## Notes for Claude Code

- **Always deploy via `./deploy.sh`** after edits — don't try to edit files directly
  on the Pi over SSH. Keep the source of truth on your local machine.
- **The log is your primary debugging tool.** Every meaningful event is logged at
  INFO level. `tail -f ~/tars-mini/tars.log` during testing.
- **Don't change the audio queue architecture** (single stream → queue → consumers).
  Opening two PyAudio streams on the same ALSA device will fail on Pi.
- **`PiperTTS.speak()` is synchronous and blocking.** This is intentional — it
  prevents audio from overlapping. Don't make it async without carefully thinking
  through the interrupt logic.
- **The Anthropic API key must be an environment variable on the Pi**, not hardcoded.
  If it isn't in the environment when TARS starts, the script exits with a clear error.
- If you need to add tools/capabilities to TARS (motor control, etc.), add them as
  Anthropic tool definitions in `stream_response()` and handle `tool_use` blocks in
  the stream. The architecture is set up for this; it just isn't wired yet.
