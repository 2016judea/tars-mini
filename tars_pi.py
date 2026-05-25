#!/usr/bin/env python3
"""
tars_pi.py — Thin audio bridge for TARS-Mini. Runs on the Raspberry Pi.

Streams mic audio to tars_server.py over TCP and plays back TTS audio.
Only requires: pyaudio  (no torch, whisper, or anthropic needed on Pi)

Protocol (server -> Pi): 4-byte uint32 audio length + 2-byte uint16 sample rate + raw PCM
Protocol (Pi -> server): continuous raw PCM stream (16kHz, S16_LE, mono)

Compatible with Python 3.5+
"""

import logging
import socket
import struct
import subprocess
import threading
import time

import pyaudio

# -- Config -------------------------------------------------------------------
SERVER_HOST      = "192.168.0.7"   # your main computer's LAN IP
SERVER_PORT      = 57001
SAMPLE_RATE      = 16000
CHANNELS         = 1
CHUNK_MS         = 80
CHUNK_FRAMES     = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 1280 frames
MIC_DEVICE_INDEX = None  # None = system default; set to 1 if wrong device

ALSA_OUTPUT_DEVICE = "plug:bluealsa:HCI=hci0,DEV=6C:47:60:58:25:93,PROFILE=sco"

# -- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tars_pi")


# -- Bluetooth speaker keepalive ----------------------------------------------

class SilenceKeepAlive:
    """Plays inaudible zeros to prevent the BT speaker from auto-powering off."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return
            try:
                self._proc = subprocess.Popen(
                    ["aplay", "-D", ALSA_OUTPUT_DEVICE,
                     "-f", "S16_LE", "-r", "8000", "-c", "1", "-t", "raw", "/dev/zero"],
                    stderr=subprocess.DEVNULL,
                )
                log.info("Silence keepalive started.")
            except Exception as e:
                log.warning("Could not start silence keepalive: " + str(e))

    def stop(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait()
                self._proc = None


# -- Audio helpers ------------------------------------------------------------

def play_audio(raw_pcm, sample_rate):
    proc = subprocess.Popen(
        ["aplay", "-D", ALSA_OUTPUT_DEVICE,
         "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-t", "raw", "-"],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    proc.communicate(input=raw_pcm)


def recv_exactly(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server disconnected")
        buf += chunk
    return buf


# -- Main loop ----------------------------------------------------------------

def main():
    pa = pyaudio.PyAudio()
    silence = SilenceKeepAlive()

    while True:
        log.info("Connecting to server " + SERVER_HOST + ":" + str(SERVER_PORT) + "...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SERVER_HOST, SERVER_PORT))
            log.info("Connected.")
        except Exception as e:
            log.error("Connection failed: " + str(e) + ". Retrying in 5s...")
            time.sleep(5)
            continue

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=MIC_DEVICE_INDEX,
            frames_per_buffer=CHUNK_FRAMES,
        )

        silence.start()
        stop_event = threading.Event()

        def send_audio():
            try:
                while not stop_event.is_set():
                    data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                    sock.sendall(data)
            except Exception as e:
                log.error("Mic send error: " + str(e))
            finally:
                stop_event.set()

        def receive_audio():
            try:
                while not stop_event.is_set():
                    header = recv_exactly(sock, 6)
                    audio_len, sample_rate = struct.unpack(">IH", header)
                    audio = recv_exactly(sock, audio_len)
                    silence.stop()
                    play_audio(audio, sample_rate)
                    silence.start()
            except ConnectionError:
                log.warning("Server disconnected.")
            except Exception as e:
                log.error("Audio receive error: " + str(e))
            finally:
                stop_event.set()

        t_send = threading.Thread(target=send_audio)
        t_send.daemon = True
        t_recv = threading.Thread(target=receive_audio)
        t_recv.daemon = True
        t_send.start()
        t_recv.start()
        stop_event.wait()

        stream.stop_stream()
        stream.close()
        silence.stop()
        sock.close()
        log.info("Disconnected. Reconnecting in 5s...")
        time.sleep(5)


if __name__ == "__main__":
    main()
