#!/usr/bin/env python3
"""
tars_server.py — TARS-Mini AI server. Runs on your main computer (Windows/Mac/Linux).

Receives mic audio from tars_pi.py over TCP, runs the full pipeline:
  openWakeWord → Silero VAD → faster-whisper → Anthropic Claude → Piper TTS
Then sends synthesised audio back to the Pi for playback.

Setup:
  pip install -r requirements_server.txt
  Download Piper binary from github.com/rhasspy/piper/releases and place next to this file.
  Set ANTHROPIC_API_KEY in your environment.
"""

import logging
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Iterator, Optional

import anthropic
import numpy as np
import torch
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 57001

# Audio (matches tars_pi.py)
SAMPLE_RATE      = 16000
CHANNELS         = 1
CHUNK_MS         = 80
CHUNK_FRAMES     = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 1280

# Wake word
WAKE_WORD_MODEL     = Path("hey_jarvis_v0.1.tflite")
WAKE_WORD_THRESHOLD = 0.5

# VAD
SILERO_SPEECH_THRESHOLD = 0.5
SILENCE_DURATION_MS     = 1200
MAX_RECORD_SECONDS      = 15
SILERO_CHUNK_FRAMES     = 512

# ASR
WHISPER_MODEL_SIZE = "tiny"
WHISPER_DEVICE     = "cpu"
WHISPER_COMPUTE    = "int8"

# LLM
ANTHROPIC_MODEL   = "claude-sonnet-4-6"
MAX_HISTORY_TURNS = 8
MAX_TOKENS        = 256

# TTS — Piper
_HERE = Path(__file__).parent
PIPER_BINARY    = str(_HERE / "piper.exe") if sys.platform == "win32" else "piper"
PIPER_MODEL     = Path("en_US-lessac-medium.onnx")
PIPER_SPEAKER   = None
PIPER_SAMPLE_RATE = 22050
TTS_MIN_CHARS   = 12

REARM_DELAY_S = 1.5

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tars_server")

# ── Persona ───────────────────────────────────────────────────────────────────

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

_honesty: int = 90
_humor:   int = 75


def _build_system_prompt() -> str:
    return _PERSONA_TEMPLATE.format(honesty=_honesty, humor=_humor)


def _update_persona(honesty=None, humor=None):
    global _honesty, _humor
    if honesty is not None:
        _honesty = max(0, min(100, honesty))
    if humor is not None:
        _humor = max(0, min(100, humor))
    log.info(f"Persona → honesty={_honesty}  humor={_humor}")


# ── Conversation history ──────────────────────────────────────────────────────

class ConversationHistory:
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
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
        while self._messages and self._messages[0]["role"] != "user":
            self._messages.pop(0)

    def as_list(self) -> list[dict]:
        return list(self._messages)


history = ConversationHistory()

# ── Model loading ─────────────────────────────────────────────────────────────

log.info(f"Loading Whisper ({WHISPER_MODEL_SIZE})…")
asr_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
log.info("Whisper ready.")

log.info("Loading Silero VAD…")
_silero_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    verbose=False,
)
torch.set_num_threads(1)
log.info("Silero VAD ready.")

log.info(f"Loading wake word model: {WAKE_WORD_MODEL}…")
try:
    import openwakeword as _oww
    _model_path = Path(_oww.__file__).parent / "resources" / "models" / WAKE_WORD_MODEL
    _ww_model = WakeWordModel(
        wakeword_models=[str(_model_path)],
        inference_framework="tflite",
    )
    log.info("Wake word model ready.")
except Exception as e:
    log.error(f"Failed to load wake word model: {e}")
    log.error("Run: python -c \"import openwakeword; openwakeword.utils.download_models()\"")
    sys.exit(1)

_client = anthropic.Anthropic()
_SETTING_RE = re.compile(r'^SETTING:(HONESTY|HUMOR):(\d+)$', re.MULTILINE)
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

# ── ASR ───────────────────────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
    try:
        segments, _ = asr_model.transcribe(wav_path, language="en", beam_size=1, vad_filter=True)
        text = " ".join(seg.text for seg in segments).strip()
        log.info(f"Transcript: {text!r}")
        return text
    finally:
        os.unlink(wav_path)


# ── VAD ───────────────────────────────────────────────────────────────────────

def _silero_is_speech(pcm_chunk: np.ndarray) -> float:
    tensor = torch.from_numpy(pcm_chunk).unsqueeze(0)
    with torch.no_grad():
        return _silero_model(tensor, SAMPLE_RATE).item()


# ── LLM ───────────────────────────────────────────────────────────────────────

def stream_response(user_text: str) -> Iterator[str]:
    history.add_user(user_text)
    full_reply_parts: list[str] = []
    speak_buffer = ""

    try:
        with _client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=_build_system_prompt(),
            messages=history.as_list(),
        ) as stream:
            for chunk in stream.text_stream:
                full_reply_parts.append(chunk)
                speak_buffer += chunk
                lines = speak_buffer.split("\n")
                speak_buffer = lines[-1]
                for line in lines[:-1]:
                    m = _SETTING_RE.match(line.strip())
                    if m:
                        key, val = m.group(1), int(m.group(2))
                        _update_persona(**{key.lower(): val})
                    else:
                        if line:
                            yield line + "\n"
            if speak_buffer.strip():
                if not _SETTING_RE.match(speak_buffer.strip()):
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
        full_text = "".join(full_reply_parts)
        clean_reply = "\n".join(
            l for l in full_text.splitlines() if not _SETTING_RE.match(l.strip())
        ).strip()
        if clean_reply:
            history.add_assistant(clean_reply)


# ── TTS — Piper → socket ──────────────────────────────────────────────────────

class PiperTTS:
    """
    Wraps a persistent Piper subprocess. Synthesises sentences and sends
    raw PCM audio back to the Pi over the TCP connection.
    """

    def __init__(self):
        self._piper = None  # type: Optional[subprocess.Popen]
        self._conn = None   # type: Optional[socket.socket]
        self._lock = threading.Lock()
        self._start_piper()

    def set_connection(self, conn):
        self._conn = conn

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
                f"Piper binary '{PIPER_BINARY}' not found.\n"
                "Download from github.com/rhasspy/piper/releases and place next to this script."
            )
            sys.exit(1)

    def speak(self, text: str):
        text = text.strip()
        if not text or self._conn is None:
            return
        log.info(f"TTS: {text!r}")

        with self._lock:
            if self._piper is None or self._piper.poll() is not None:
                log.warning("Piper process died — restarting.")
                self._start_piper()
            try:
                self._piper.stdin.write((text + "\n").encode())
                self._piper.stdin.flush()
                audio_bytes = self._piper.stdout.read(int(PIPER_SAMPLE_RATE * 2 * 5))
            except (BrokenPipeError, OSError) as e:
                log.error(f"Piper pipe error: {e} — restarting.")
                self._piper = None
                self._start_piper()
                return

            header = struct.pack(">IH", len(audio_bytes), PIPER_SAMPLE_RATE)
            try:
                self._conn.sendall(header + audio_bytes)
            except OSError:
                log.warning("Pi disconnected during TTS send.")

    def close(self):
        if self._piper and self._piper.poll() is None:
            self._piper.stdin.close()
            self._piper.terminate()


tts = PiperTTS()


def speak_stream(text_iterator: Iterator[str]):
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
                    parts[-1] = sentence + " " + parts[-1]
            buffer = parts[-1]
    remainder = buffer.strip()
    if remainder:
        tts.speak(remainder)


# ── Per-connection pipeline ───────────────────────────────────────────────────

def handle_client(conn: socket.socket, addr):
    log.info(f"Pi connected from {addr}")
    tts.set_connection(conn)

    _audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
    stop_event = threading.Event()

    def socket_reader():
        """Fill audio queue from the Pi's mic stream."""
        try:
            while not stop_event.is_set():
                data = conn.recv(CHUNK_FRAMES * 2)
                if not data:
                    break
                try:
                    _audio_queue.put_nowait(data)
                except queue.Full:
                    try:
                        _audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    _audio_queue.put_nowait(data)
        except OSError:
            pass
        finally:
            stop_event.set()

    def drain():
        while not _audio_queue.empty():
            try:
                _audio_queue.get_nowait()
            except queue.Empty:
                break

    def wait_for_wake_word():
        log.info("Waiting for wake word…")
        while not stop_event.is_set():
            try:
                data = _audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            audio_f32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            preds = _ww_model.predict(audio_f32)
            score = max(preds.values()) if preds else 0.0
            if score >= WAKE_WORD_THRESHOLD:
                log.info(f"Wake word detected (score={score:.2f})")
                return True
        return False

    def record_utterance():
        log.info("Recording utterance…")
        frames: list[bytes] = []
        silent_ms = 0
        total_ms = 0
        speech_started = False
        windows_per_chunk = CHUNK_FRAMES // SILERO_CHUNK_FRAMES
        while total_ms < MAX_RECORD_SECONDS * 1000 and not stop_event.is_set():
            try:
                data = _audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            frames.append(data)
            total_ms += CHUNK_MS
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            chunk_has_speech = False
            for i in range(windows_per_chunk):
                window = samples[i * SILERO_CHUNK_FRAMES: (i + 1) * SILERO_CHUNK_FRAMES]
                if len(window) < SILERO_CHUNK_FRAMES:
                    break
                if _silero_is_speech(window) >= SILERO_SPEECH_THRESHOLD:
                    chunk_has_speech = True
                    break
            if chunk_has_speech:
                speech_started = True
                silent_ms = 0
            elif speech_started:
                silent_ms += CHUNK_MS
                if silent_ms >= SILENCE_DURATION_MS:
                    log.info(f"End of utterance ({silent_ms}ms silence).")
                    break
        if not speech_started:
            log.info("No speech detected.")
            return None
        return b"".join(frames)

    reader_thread = threading.Thread(target=socket_reader, daemon=True)
    reader_thread.start()

    try:
        while not stop_event.is_set():
            if not wait_for_wake_word():
                break
            drain()

            audio = record_utterance()
            if audio is None:
                continue

            user_text = transcribe(audio)
            if not user_text:
                log.info("Empty transcript — skipping.")
                continue

            log.info("Calling Claude…")
            speak_stream(stream_response(user_text))

            log.info(f"Re-arming in {REARM_DELAY_S}s…")
            drain()
            time.sleep(REARM_DELAY_S)
            drain()

    finally:
        stop_event.set()
        tts.set_connection(None)
        conn.close()
        log.info(f"Pi {addr} disconnected.")


# ── Server entry point ────────────────────────────────────────────────────────

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set. Export it before running.")
        sys.exit(1)
    if not PIPER_MODEL.exists():
        log.error(
            f"Piper model not found: {PIPER_MODEL}\n"
            "Download from github.com/rhasspy/piper/releases"
        )
        sys.exit(1)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(1)
    log.info(f"TARS server listening on port {SERVER_PORT}. Waiting for Pi…")

    try:
        while True:
            conn, addr = server.accept()
            handle_client(conn, addr)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        tts.close()
        server.close()


if __name__ == "__main__":
    main()
