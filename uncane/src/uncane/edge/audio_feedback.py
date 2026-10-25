"""User interface outputs: spatial beeps, prioritised speech, optional haptics.

* SpatialBeeper - continuous low-latency stereo tone stream. Pan = direction,
  pitch = height (ground/body/head), beep rate = distance. Uses `sounddevice`.
* Speaker       - text-to-speech worker (pyttsx3, offline). Safety messages
  jump the queue and flush stale lower-priority ones.
* Haptics       - optional serial haptic belt/vest ("L,C,R intensities").

Everything degrades to logging if the audio stack is missing, so the rest of
the pipeline keeps running (e.g. on a headless CI box).
"""
from __future__ import annotations

import itertools
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

from .arbiter import ToneCue

log = logging.getLogger(__name__)

PRIORITY_SAFETY = 0
PRIORITY_CLOUD = 1
PRIORITY_INFO = 2

_SAPI_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Rate = [int]$env:UNCANE_TTS_RATE; "
    "$s.Speak($env:UNCANE_TTS_TEXT)"
)


def _speak_windows_sapi(text: str, rate_wpm: int, timeout_s: float = 30.0) -> None:
    """Speaks through Windows SAPI. The text travels in an environment variable,
    so quotes or symbols in the answer can't break the PowerShell command.
    Maps pyttsx3's words-per-minute (about 180 is normal) onto SAPI's -10..10 rate.
    """
    sapi_rate = max(-10, min(10, round((rate_wpm - 180) / 20)))
    env = {**os.environ, "UNCANE_TTS_TEXT": text, "UNCANE_TTS_RATE": str(sapi_rate)}
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: no console flash
    subprocess.run(["powershell", "-NoProfile", "-Command", _SAPI_SCRIPT], env=env,
                   timeout=timeout_s, creationflags=flags, check=False)


class SpatialBeeper:
    SAMPLE_RATE = 44100
    BEEP_S = 0.06

    def __init__(self, volume: float = 0.35, enabled: bool = True):
        self.volume = volume
        self._cue: Optional[ToneCue] = None
        self._phase = 0.0
        self._since_beep = 1e9
        self._stream = None
        if not enabled:
            return
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(samplerate=self.SAMPLE_RATE, channels=2, dtype="float32",
                                           blocksize=256, latency="low", callback=self._callback)
            self._stream.start()
            log.info("spatial beeper started")
        except Exception as exc:  # no device, no PortAudio, ...
            log.warning("spatial audio unavailable (%s); beeps will be logged only", exc)

    def set_cue(self, cue: Optional[ToneCue]) -> None:
        self._cue = cue

    def _callback(self, outdata, frames, _time, _status) -> None:
        cue = self._cue
        out = np.zeros((frames, 2), dtype=np.float32)
        if cue is not None:
            t = np.arange(frames) / self.SAMPLE_RATE
            pos = self._since_beep + t
            period = max(cue.interval_s, self.BEEP_S * 1.5)
            on = (pos % period) < self.BEEP_S
            env = np.where(on, 1.0, 0.0).astype(np.float32)
            wave = np.sin(2 * math.pi * cue.freq_hz * t + self._phase).astype(np.float32) * env * self.volume
            self._phase = (self._phase + 2 * math.pi * cue.freq_hz * frames / self.SAMPLE_RATE) % (2 * math.pi)
            angle = (cue.pan + 1.0) * math.pi / 4.0          # equal-power panning
            out[:, 0] = wave * math.cos(angle)
            out[:, 1] = wave * math.sin(angle)
            self._since_beep = (self._since_beep + frames / self.SAMPLE_RATE) % period
        outdata[:] = out

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


class Speaker:
    def __init__(self, rate: int = 190, voice: Optional[str] = None, enabled: bool = True):
        self._q: "queue.PriorityQueue" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._rate, self._voice, self._enabled = rate, voice, enabled
        self._stop = threading.Event()
        self.last_spoken = ""
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()

    def say(self, text: str, priority: int = PRIORITY_CLOUD) -> None:
        if not text:
            return
        if priority == PRIORITY_SAFETY:
            self._flush_below(PRIORITY_SAFETY)
        elif priority == PRIORITY_CLOUD:
            # A new answer replaces queued status lines ("Looking.") but never safety speech.
            self._flush_below(PRIORITY_CLOUD)
        self._q.put((priority, next(self._seq), time.monotonic(), text))

    def _flush_below(self, priority: int) -> None:
        keep = []
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item[0] <= priority:
                keep.append(item)
        for item in keep:
            self._q.put(item)

    def _make_speak(self):
        """Returns speak(text), or None when speech is off or unavailable."""
        if not self._enabled:
            return None
        if sys.platform == "win32":
            # pyttsx3 stays silent after its first utterance in a process on Windows,
            # so speak through SAPI in a fresh PowerShell process per utterance instead.
            return lambda text: _speak_windows_sapi(text, self._rate)
        try:
            import pyttsx3

            pyttsx3.init()  # fail early if no engine is installed
        except Exception as exc:
            log.warning("TTS unavailable (%s); speech will be logged only", exc)
            return None

        def speak(text: str) -> None:
            engine = pyttsx3.init()  # fresh engine per utterance; reuse goes silent
            engine.setProperty("rate", self._rate)
            if self._voice:
                for v in engine.getProperty("voices"):
                    if self._voice.lower() in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
            engine.say(text)
            engine.runAndWait()

        return speak

    def _run(self) -> None:
        speak = self._make_speak()
        while not self._stop.is_set():
            try:
                priority, _, queued_at, text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if priority == PRIORITY_INFO and time.monotonic() - queued_at > 8.0:
                log.info("skipping stale status message: %s", text)
                continue  # the world has moved on; answers are aged in the app instead
            self.last_spoken = text
            log.info("SPEAK[%d]: %s", priority, text)
            if speak is not None:
                try:
                    speak(text)
                except Exception as exc:
                    log.warning("TTS error: %s", exc)

    def close(self) -> None:
        self._stop.set()


class Haptics:
    """Serial haptic driver. Protocol: one line per update, 'L,C,R\\n' with 0-255 each."""

    def __init__(self, port: Optional[str]):
        self._ser = None
        if not port:
            return
        try:
            import serial  # pyserial

            self._ser = serial.Serial(port, 115200, timeout=0)
            log.info("haptics on %s", port)
        except Exception as exc:
            log.warning("haptics unavailable on %s (%s)", port, exc)

    def update(self, cue: Optional[ToneCue]) -> None:
        if self._ser is None:
            return
        vals = [0, 0, 0]
        if cue is not None:
            idx = 0 if cue.pan < -0.3 else 2 if cue.pan > 0.3 else 1
            vals[idx] = int(min(255, 60 + 195 * (0.8 - min(cue.interval_s, 0.8)) / 0.72))
        try:
            self._ser.write(f"{vals[0]},{vals[1]},{vals[2]}\n".encode())
        except Exception as exc:
            log.warning("haptics write failed: %s", exc)
