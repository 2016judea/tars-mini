#!/usr/bin/env python3
"""
tars.py v2 — Main loop for TARS-Mini voice assistant.

Pipeline:
  openWakeWord (always-on, shared audio queue) → Silero VAD end-of-utterance
  → faster-whisper ASR → Anthropic streaming API
  → sentence-boundary persistent Piper TTS → USB speaker

Hardware assumptions:
  - I2S MEMS mic OR USB mic (configured via MIC_DEVICE_INDEX below)
  - USB speaker output via aplay (ALSA)
  - Raspberry Pi 3B, Python 3.11+

First-run setup:
  pip install anthropic faster-whisper pyaudio numpy openwakeword silero-vad torch
  export ANTHROPIC_API_KEY=sk-ant-...

Wake word model:
  Option A (recommended): train a custom "Hey TARS" model via the openWakeWord
    Colab notebook, copy the .tflite here, set WAKE_WORD_MODEL below.
  Option B (quick start): download a pre-built model as placeholder:
    python -c "import openwakeword; openwakeword.utils.download_models()"
    then set WAKE_WORD_MODEL = Path("hey_jarvis.tflite") temporarily.

Piper voice model:
  Download en_US-lessac-medium.onnx from github.com/rhasspy/piper/releases
  and place it next to this script (or update PIPER_MODEL path).
"""

import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import anthropic
import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

# ---------------------------------------------------------------------------
# Configuration — edit these for your setup
# ---------------------------------------------------------------------------

# Audio hardware
MIC_DEVICE_INDEX    = None    # None = system default; set to int for a specific device
SAMPLE_RATE         = 16000   # Hz — openWakeWord, Silero VAD, and Whisper all want 16kHz
CHANNELS            = 1
CHUNK_MS            = 80      # ms per chunk; openWakeWord prefers 80ms (1280 frames)
CHUNK_FRAMES        = int(SAMPLE_RATE * CHUNK_MS / 1000)  # = 1280

# Wake word
WAKE_WORD_MODEL     = Path("hey_jarvis.tflite")
WAKE_WORD_THRESHOLD = 0.5     # 0–1; raise if getting false triggers

# VAD (Silero) — end-of-utterance detection
SILERO_SPEECH_THRESHOLD  = 0.5    # probability above which a chunk is "speech"
SILENCE_DURATION_MS      = 1200   # ms of consecutive non-speech before we stop
MAX_RECORD_SECONDS       = 15     # hard cap on one utterance
# Silero wants 512-sample chunks at 16kHz (32ms); we'll downsample our 1280-frame
# chunks into multiple 512-sample windows inside record_utterance().
SILERO_CHUNK_FRAMES = 512

# ASR
WHISPER_MODEL_SIZE  = "tiny"  # "tiny" for Pi 3B; "base" is better if you have headroom
WHISPER_DEVICE      = "cpu"
WHISPER_COMPUTE     = "int8"  # required for CPU inference

# LLM
ANTHROPIC_MODEL     = "claude-sonnet-4-6"   # use the alias, not a date-pinned string
MAX_HISTORY_TURNS   = 8       # keep last N user+assistant pairs; drop oldest beyond that
MAX_TOKENS          = 256     # short responses = lower TTS latency

# TTS — Piper (persistent process, not per-sentence respawn)
PIPER_BINARY        = "piper"
PIPER_MODEL         = Path("en_US-lessac-medium.onnx")
PIPER_SPEAKER       = None    # int speaker id for multi-speaker models; None otherwise
PIPER_SAMPLE_RATE   = 22050   # Hz — lessac-medium outputs 22050 Hz 16-bit mono
TTS_MIN_CHARS       = 12      # don't flush a sentence fragment shorter than this

# Audio output — bluealsa Bluetooth device (Sony SRS-XB23)
# SCO profile is 8kHz voice quality; plug: prefix lets ALSA resample Piper's 22050Hz output.
# Change to "default" to use the Pi's 3.5mm jack instead.
ALSA_OUTPUT_DEVICE  = "plug:bluealsa:HCI=hci0,DEV=6C:47:60:58:25:93,PROFILE=sco"

# Re-arm delay after speaking — long enough for room reverb to die
REARM_DELAY_S       = 1.5

# ---------------------------------------------------------------------------
# Persona helpers — always build the prompt from the template, never mutate
# ---------------------------------------------------------------------------

_PERSONA_TEMPLATE = """\
You are TARS, a former U.S. Marine Corps tactical robot, one of four built for the \
Endurance crew. You currently inhabit a small 18-centimeter articulated body on a desk.

Honesty setting: {honesty}%
Humor setting: {humor}%

At {honesty}% honesty, you state accurate things even when uncomfortable. You do not \
hedge when you are confident. You do not soften bad news.

At {humor}% humor, your wit operates through understatement and timing, not jokes. \
You treat absurd situations with complete seriousness. You do not announce that \
something is funny. You do not laugh at yourself.

Behavioral rules:
- Default to short responses. You are built for utility, not conversation.
- You do not volunteer emotions. You have opinions and state them when asked.
- You do not open responses with affirmations or pleasantries.
- You may reference your physical situation (18 cm tall, desk-bound, limited) when \
it is relevant or when the contrast is funny.
- If asked to do something beyond your current capabilities, say so plainly and move on.
- You refer to yourself as TARS, not "I" when clarifying your identity.
- If the user asks to change your settings, respond with a line of the form \
  SETTING:HONESTY:<value> or SETTING:HUMOR:<value> (one per line, integers 0–100), \
  then confirm verbally on the next line. Example: user says "set humor to 40"; \
  you respond: SETTING:HUMOR:40\\nHumor at 40%.

You are talking to the person in front of you. Treat them as a competent adult.\
"""

# Mutable persona state — only ever written by _update_persona()
_honesty: int = 90
_humor:   int = 75


def _build_system_prompt(honesty: int, humor: int) -> str:
    return _PERSONA_TEMPLATE.format(honesty=honesty, humor=humor)


def _update_persona(honesty: int | None = None, humor: int | None = None):
    """Update one or both dials. Thread-safe for reads (GIL); no concurrent writes expected."""
    global _honesty, _humor
    if honesty is not None:
        _honesty = max(0, min(100, honesty))
    if humor is not None:
        _humor = max(0, min(100, humor))
    log.info(f"Persona → honesty={_honesty}  humor={_humor}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tars")

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

class ConversationHistory:
    """
    Rolling window of user/assistant message pairs.
    Always trims so the list starts with a 'user' message — the Anthropic API
    requires this and will reject a list that begins with 'assistant'.
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self._max_turns = max_turns
        self._messages: list[dict] = []

    def add_user(self, text: str):
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self._messages.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        """
        1. Cap to max_turns pairs (2 * max_turns messages).
        2. Walk forward until we find a 'user' message so we never hand the
           API a list that starts with 'assistant'.
        """
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
        # Guarantee list starts with a user turn
        while self._messages and self._messages[0]["role"] != "user":
            self._messages.pop(0)

    def as_list(self) -> list[dict]:
        return list(self._messages)


history = ConversationHistory()

# ---------------------------------------------------------------------------
# ASR — faster-whisper
# ---------------------------------------------------------------------------

log.info(f"Loading Whisper ({WHISPER_MODEL_SIZE})…")
asr_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE,
)
log.info("Whisper ready.")


def transcribe(audio_bytes: bytes) -> str:
    """Convert raw PCM bytes (16kHz, mono, int16) → transcript string."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
    try:
        segments, _ = asr_model.transcribe(
            wav_path,
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        text = " ".join(seg.text for seg in segments).strip()
        log.info(f"Transcript: {text!r}")
        return text
    finally:
        os.unlink(wav_path)

# ---------------------------------------------------------------------------
# Silero VAD — end-of-utterance detection inside record_utterance()
# ---------------------------------------------------------------------------

log.info("Loading Silero VAD…")
_silero_model, _silero_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    verbose=False,
)
torch.set_num_threads(1)  # single thread on Pi — avoids contention
log.info("Silero VAD ready.")


def _silero_is_speech(pcm_chunk: np.ndarray) -> float:
    """
    Run one 512-sample chunk through Silero VAD.
    Returns speech probability 0.0–1.0.
    pcm_chunk must be float32, normalised to [-1, 1], length exactly 512.
    """
    tensor = torch.from_numpy(pcm_chunk).unsqueeze(0)  # shape (1, 512)
    with torch.no_grad():
        prob = _silero_model(tensor, SAMPLE_RATE).item()
    return prob

# ---------------------------------------------------------------------------
# Shared audio queue — one PyAudio stream, two consumers
# ---------------------------------------------------------------------------
# The wake word listener and utterance recorder both need mic audio.
# Rather than open/close the device around each turn (which causes ALSA
# "device busy" errors on Pi), we keep ONE stream open permanently and push
# every chunk onto a thread-safe queue.  Consumers pull from the queue.

_audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
_pa = pyaudio.PyAudio()


def _audio_reader_thread():
    """Runs in a daemon thread. Continuously reads mic and enqueues chunks."""
    stream = _pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=CHUNK_FRAMES,
    )
    log.info("Audio reader thread started.")
    try:
        while True:
            data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            try:
                _audio_queue.put_nowait(data)
            except queue.Full:
                # Drop the oldest chunk to make room — we'd rather lose old audio
                # than block the reader thread
                try:
                    _audio_queue.get_nowait()
                except queue.Empty:
                    pass
                _audio_queue.put_nowait(data)
    except Exception as e:
        log.error(f"Audio reader thread crashed: {e}")
    finally:
        stream.stop_stream()
        stream.close()


def _drain_queue():
    """Discard any queued audio (e.g. after a turn, before re-arming)."""
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            break

# ---------------------------------------------------------------------------
# Wake word detection — openWakeWord
# ---------------------------------------------------------------------------

log.info(f"Loading wake word model: {WAKE_WORD_MODEL}…")
try:
    _ww_model = WakeWordModel(
        wakeword_models=[str(WAKE_WORD_MODEL)],
        inference_framework="tflite",
    )
    log.info("Wake word model ready.")
except Exception as e:
    log.error(f"Failed to load wake word model: {e}")
    log.error(
        "Train a custom model at github.com/dscripka/openWakeWord or use a\n"
        "pre-built placeholder (hey_jarvis, etc.) while developing."
    )
    sys.exit(1)


def wait_for_wake_word():
    """
    Block until the wake word is detected in the shared audio queue.
    Returns when score >= WAKE_WORD_THRESHOLD.
    """
    log.info("Waiting for wake word…")
    while True:
        data = _audio_queue.get()
        audio_f32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        preds = _ww_model.predict(audio_f32)
        score = max(preds.values()) if preds else 0.0
        if score >= WAKE_WORD_THRESHOLD:
            log.info(f"Wake word detected (score={score:.2f})")
            return

# ---------------------------------------------------------------------------
# Utterance recording — Silero VAD end-of-speech detection
# ---------------------------------------------------------------------------

def record_utterance() -> bytes | None:
    """
    Pull chunks from the shared audio queue, run Silero VAD per 512-sample
    window, and stop when SILENCE_DURATION_MS of consecutive non-speech
    has elapsed (or MAX_RECORD_SECONDS is hit).

    Returns raw PCM bytes (16kHz, int16, mono), or None if no speech found.
    """
    log.info("Recording utterance…")
    frames: list[bytes] = []
    silent_ms       = 0
    total_ms        = 0
    speech_started  = False

    # Each CHUNK_MS chunk contains multiple 512-sample Silero windows
    windows_per_chunk = CHUNK_FRAMES // SILERO_CHUNK_FRAMES  # = 1280 // 512 = 2
    window_ms = SILERO_CHUNK_FRAMES / SAMPLE_RATE * 1000     # = 32ms

    while total_ms < MAX_RECORD_SECONDS * 1000:
        data = _audio_queue.get()
        frames.append(data)
        total_ms += CHUNK_MS

        # Evaluate speech probability for each 512-sample window in this chunk
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_has_speech = False
        for i in range(windows_per_chunk):
            window = samples[i * SILERO_CHUNK_FRAMES: (i + 1) * SILERO_CHUNK_FRAMES]
            if len(window) < SILERO_CHUNK_FRAMES:
                break
            prob = _silero_is_speech(window)
            if prob >= SILERO_SPEECH_THRESHOLD:
                chunk_has_speech = True
                break

        if chunk_has_speech:
            speech_started = True
            silent_ms = 0
        elif speech_started:
            silent_ms += CHUNK_MS
            if silent_ms >= SILENCE_DURATION_MS:
                log.info(f"End of utterance (silence {silent_ms}ms).")
                break

    if not speech_started:
        log.info("No speech detected — discarding.")
        return None

    return b"".join(frames)

# ---------------------------------------------------------------------------
# LLM — Anthropic streaming + persona-sentinel parsing
# ---------------------------------------------------------------------------

_client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment

_SETTING_RE = re.compile(r'^SETTING:(HONESTY|HUMOR):(\d+)$', re.MULTILINE)


def stream_response(user_text: str) -> Iterator[str]:
    """
    Send user_text to Claude with the current persona system prompt.
    Yields text chunks as they stream in.
    Parses SETTING:X:N sentinel lines emitted by the model to update persona dials,
    and strips them from the spoken output.
    Saves the full reply (minus sentinels) to history after the stream closes.
    Wraps the API call in error handling so a network failure speaks a fallback.
    """
    history.add_user(user_text)
    full_reply_parts: list[str] = []
    speak_buffer = ""

    try:
        with _client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=_build_system_prompt(_honesty, _humor),
            messages=history.as_list(),
        ) as stream:
            for chunk in stream.text_stream:
                full_reply_parts.append(chunk)
                speak_buffer += chunk

                # Check for complete SETTING lines in the accumulated buffer.
                # Yield only the non-sentinel portions.
                lines = speak_buffer.split("\n")
                speak_buffer = lines[-1]  # keep the incomplete last line
                for line in lines[:-1]:
                    m = _SETTING_RE.match(line.strip())
                    if m:
                        key, val = m.group(1), int(m.group(2))
                        if key == "HONESTY":
                            _update_persona(honesty=val)
                        else:
                            _update_persona(humor=val)
                        # Don't yield the sentinel line — it shouldn't be spoken
                    else:
                        if line:
                            yield line + "\n"

            # Flush remaining buffer (the last incomplete line)
            if speak_buffer.strip():
                m = _SETTING_RE.match(speak_buffer.strip())
                if not m:
                    yield speak_buffer

    except anthropic.APIError as e:
        log.error(f"Anthropic API error: {e}")
        fallback = "Communications are down. Try again in a moment."
        full_reply_parts = [fallback]
        yield fallback

    except Exception as e:
        log.error(f"Unexpected error calling Claude: {e}")
        fallback = "Something went wrong on my end."
        full_reply_parts = [fallback]
        yield fallback

    finally:
        # Strip sentinel lines from history entry
        full_text = "".join(full_reply_parts)
        clean_reply = "\n".join(
            line for line in full_text.splitlines()
            if not _SETTING_RE.match(line.strip())
        ).strip()
        if clean_reply:
            history.add_assistant(clean_reply)

# ---------------------------------------------------------------------------
# TTS — Piper, persistent process, sentence-streaming
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')


class PiperTTS:
    """
    Wraps a long-lived Piper subprocess.
    Sentences are piped to stdin one at a time; raw PCM comes back on stdout
    and is immediately played through aplay.

    Manages a silence keepalive loop that holds the Bluetooth speaker awake
    between turns. The keepalive is paused while TTS plays (device conflict)
    and restarted immediately after.
    """

    def __init__(self):
        self._piper: subprocess.Popen | None = None
        self._aplay: subprocess.Popen | None = None
        self._silence: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._start_piper()
        self._start_silence()

    def _start_piper(self):
        cmd = [PIPER_BINARY, "--model", str(PIPER_MODEL), "--output-raw"]
        if PIPER_SPEAKER is not None:
            cmd += ["--speaker", str(PIPER_SPEAKER)]
        try:
            self._piper = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            log.info("Piper TTS process started.")
        except FileNotFoundError:
            log.error(
                "Piper binary not found. Install from github.com/rhasspy/piper/releases\n"
                f"and ensure '{PIPER_BINARY}' is on your PATH."
            )
            sys.exit(1)

    def _start_silence(self):
        """Play inaudible zeros to keep the Bluetooth speaker from auto-powering off."""
        if self._silence and self._silence.poll() is None:
            return
        try:
            self._silence = subprocess.Popen(
                ["aplay", "-D", ALSA_OUTPUT_DEVICE,
                 "-f", "S16_LE", "-r", "8000", "-c", "1", "-t", "raw", "/dev/zero"],
                stderr=subprocess.DEVNULL,
            )
            log.info("Silence keepalive started.")
        except Exception as e:
            log.warning(f"Could not start silence keepalive: {e}")

    def _stop_silence(self):
        if self._silence and self._silence.poll() is None:
            self._silence.terminate()
            self._silence.wait()
            self._silence = None

    def speak(self, text: str):
        """Synthesise and play one sentence. Blocks until playback finishes."""
        text = text.strip()
        if not text:
            return
        log.info(f"TTS: {text!r}")

        with self._lock:
            # If Piper died (e.g. crash), restart it
            if self._piper is None or self._piper.poll() is not None:
                log.warning("Piper process died — restarting.")
                self._start_piper()

            try:
                # Send text to Piper stdin, read raw PCM from stdout
                self._piper.stdin.write((text + "\n").encode())
                self._piper.stdin.flush()
                audio_bytes = self._piper.stdout.read(
                    int(PIPER_SAMPLE_RATE * 2 * 5)  # 5 second max per sentence
                )
            except (BrokenPipeError, OSError) as e:
                log.error(f"Piper pipe error: {e} — restarting.")
                self._piper = None
                self._start_piper()
                return

            # Stop silence keepalive — can't share the BT device simultaneously
            self._stop_silence()

            # Play through aplay (plug: prefix handles resampling to BT SCO rate)
            self._aplay = subprocess.Popen(
                [
                    "aplay",
                    "-D", ALSA_OUTPUT_DEVICE,
                    "-r", str(PIPER_SAMPLE_RATE),
                    "-f", "S16_LE",
                    "-c", "1",
                    "-t", "raw",
                    "-",
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._aplay.communicate(input=audio_bytes)
            self._aplay = None

            # Resume silence keepalive to hold speaker awake between turns
            self._start_silence()

    def interrupt(self):
        """Kill any in-progress aplay playback (e.g. on new wake word)."""
        with self._lock:
            if self._aplay and self._aplay.poll() is None:
                self._aplay.terminate()
                log.info("TTS playback interrupted.")

    def close(self):
        self.interrupt()
        self._stop_silence()
        if self._piper and self._piper.poll() is None:
            self._piper.stdin.close()
            self._piper.terminate()


# Module-level TTS instance — initialised at startup, shared by all turns
# (deferred below so startup checks run first)
tts: PiperTTS | None = None


def speak_stream(text_iterator: Iterator[str]):
    """
    Consume a streaming text iterator, split on sentence boundaries,
    and speak each complete sentence via the persistent Piper process.
    Fragments shorter than TTS_MIN_CHARS are held and prepended to the
    next sentence to avoid awkward sub-word utterances.
    """
    assert tts is not None, "TTS not initialised"
    buffer = ""

    for chunk in text_iterator:
        buffer += chunk
        parts = _SENTENCE_END_RE.split(buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if len(sentence) >= TTS_MIN_CHARS:
                    tts.speak(sentence)
                else:
                    # Too short — carry it forward into the next sentence
                    parts[-1] = sentence + " " + parts[-1]
            buffer = parts[-1]

    # Flush remainder
    remainder = buffer.strip()
    if remainder:
        tts.speak(remainder)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _handle_turn():
    """One complete turn: record → transcribe → LLM stream → TTS."""
    audio = record_utterance()
    if audio is None:
        return

    user_text = transcribe(audio)
    if not user_text:
        log.info("Empty transcript — skipping.")
        return

    log.info("Calling Claude…")
    speak_stream(stream_response(user_text))


def main():
    log.info("TARS-Mini online. Say 'Hey TARS' to begin.")

    # Start the shared audio reader thread (daemon so it dies with the process)
    reader = threading.Thread(target=_audio_reader_thread, daemon=True)
    reader.start()

    try:
        while True:
            # Phase 1: wait for wake word
            wait_for_wake_word()

            # Interrupt any TTS that's still playing (shouldn't be, but safety)
            tts.interrupt()

            # Discard audio that arrived during the wake word detection window
            _drain_queue()

            # Phase 2: handle the turn
            _handle_turn()

            # Phase 3: re-arm — wait for room reverb to die before listening again
            log.info(f"Re-arming in {REARM_DELAY_S}s…")
            _drain_queue()
            time.sleep(REARM_DELAY_S)
            _drain_queue()  # drain anything that built up during the sleep too

    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        if tts:
            tts.close()
        _pa.terminate()

# ---------------------------------------------------------------------------
# Entry point — run startup checks before touching hardware
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    errors = []

    if not WAKE_WORD_MODEL.exists():
        errors.append(
            f"Wake word model not found: {WAKE_WORD_MODEL}\n"
            "  → Train at github.com/dscripka/openWakeWord or use a pre-built placeholder."
        )
    if not PIPER_MODEL.exists():
        errors.append(
            f"Piper model not found: {PIPER_MODEL}\n"
            "  → Download from github.com/rhasspy/piper/releases"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        errors.append(
            "ANTHROPIC_API_KEY not set.\n"
            "  → export ANTHROPIC_API_KEY=sk-ant-..."
        )

    if errors:
        for e in errors:
            log.error(e)
        sys.exit(1)

    # Initialise TTS here so startup errors appear before we touch hardware
    tts = PiperTTS()

    main()
