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

import base64
import itertools
import logging
import math
import queue
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

from .arbiter import ToneCue

log = logging.getLogger(__name__)

PRIORITY_SAFETY = 0
PRIORITY_CLOUD = 1
PRIORITY_INFO = 2

# Line protocol: "SPEAK<TAB>rate<TAB>volume<TAB>base64(utf-8 text)" or "STOP"; replies "DONE" per SPEAK.
# SpVoice.Speak flags: 1 = async, 2 = SVSFPurgeBeforeSpeak. Speak('', 3) purges what is playing.
_SAPI_SERVER = r"""
$ErrorActionPreference = 'Stop'
$voice = New-Object -ComObject SAPI.SpVoice
$reader = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), [Text.Encoding]::UTF8)
$pending = $reader.ReadLineAsync()
$speaking = $false
$began = [DateTime]::UtcNow
while ($true) {
    if ($pending.IsCompleted) {
        $line = $pending.Result
        if ($null -eq $line) { break }
        $parts = $line.Split("`t")
        if ($parts[0] -eq 'STOP') {
            [void]$voice.Speak('', 3)
        } elseif ($parts[0] -eq 'SPEAK') {
            $voice.Rate = [int]$parts[1]
            $voice.Volume = [int]$parts[2]
            $text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($parts[3]))
            [void]$voice.Speak($text, 3)
            $speaking = $true
            $began = [DateTime]::UtcNow
        }
        $pending = $reader.ReadLineAsync()
    }
    if ($speaking) {
        if ($voice.WaitUntilDone(10) -or ([DateTime]::UtcNow - $began).TotalSeconds -gt 90) {
            $speaking = $false
            [Console]::Out.WriteLine('DONE')
            [Console]::Out.Flush()
        }
    } else {
        Start-Sleep -Milliseconds 10
    }
}
"""


class _SapiWorker:
    """One long-lived SAPI voice, so STOP/SPEAK can purge speech that is already playing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        encoded = base64.b64encode(_SAPI_SERVER.encode("utf-16-le")).decode("ascii")
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: no console flash
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", creationflags=flags)

    def _send(self, line: str) -> None:
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def speak(self, text: str, rate_wpm: int, volume: float, aborted: Callable[[], bool]) -> None:
        # SAPI rate is -10..10; pyttsx3's words-per-minute (about 180 is normal) maps onto it.
        rate = max(-10, min(10, round((rate_wpm - 180) / 20)))
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        with self._lock:
            if aborted():   # checked under the lock so a STOP can't land before this SPEAK
                return
            self._send(f"SPEAK\t{rate}\t{round(max(0.0, min(1.0, volume)) * 100)}\t{payload}")
        if not self._proc.stdout.readline():
            raise RuntimeError("SAPI worker exited")

    def stop(self) -> None:
        with self._lock:
            try:
                self._send("STOP")
            except OSError:
                pass

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        self._proc.terminate()


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

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))

    def _callback(self, outdata, frames, _time, _status) -> None:
        cue = self._cue
        out = np.zeros((frames, 2), dtype=np.float32)
        if cue is not None:
            t = np.arange(frames) / self.SAMPLE_RATE
            pos = self._since_beep + t
            period = max(cue.interval_s, self.BEEP_S * 1.5)
            phase = pos % period
            on = ((phase < 0.055) | ((phase >= 0.11) & (phase < 0.165))) if cue.kind == "dropoff" \
                else phase < self.BEEP_S
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
    """Priority speech: safety > cloud answers > status chatter.

    A cloud answer replaces queued status lines and status lines never cut it off. A safety
    warning interrupts whatever is playing; an answer it cut off is replayed afterwards.
    stop() is the user's command: it interrupts and clears everything, with no replay.
    """

    def __init__(self, rate: int = 190, voice: Optional[str] = None, enabled: bool = True,
                 volume: float = 1.0):
        self._q: "queue.PriorityQueue" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._cv = threading.Condition()   # guards the queue and the interrupt state below
        self._rate, self._voice, self._enabled = rate, voice, enabled
        self._volume = max(0.0, min(1.0, float(volume)))
        self._stop = threading.Event()
        self._current_priority: Optional[int] = None
        self._cancel_hook: Optional[Callable[[], None]] = None
        self._gen = 0     # bumped on every interrupt; speech started under an older value is aborted
        self._epoch = 0   # bumped only by stop(): interrupted speech is then discarded, not replayed
        self._sapi: Optional[_SapiWorker] = None
        self.last_spoken = ""
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()

    def say(self, text: str, priority: int = PRIORITY_CLOUD) -> None:
        if not text:
            return
        with self._cv:
            if priority < PRIORITY_INFO:
                self._drop_queued_info()
            self._q.put((priority, next(self._seq), time.monotonic(), text))
            if self._current_priority is not None and self._current_priority > priority:
                self._interrupt_locked()
            self._cv.notify()

    def set_controls(self, rate: Optional[int] = None, volume: Optional[float] = None) -> None:
        if rate is not None:
            self._rate = max(80, min(300, int(rate)))
        if volume is not None:
            self._volume = max(0.0, min(1.0, float(volume)))

    def stop(self) -> None:
        with self._cv:
            self._epoch += 1
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._interrupt_locked()

    def _interrupt_locked(self) -> None:
        self._gen += 1
        hook = self._cancel_hook
        if hook is not None:
            try:
                hook()
            except Exception as exc:
                log.warning("could not interrupt speech: %s", exc)

    def _drop_queued_info(self) -> None:
        keep = []
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item[0] < PRIORITY_INFO:
                keep.append(item)
        for item in keep:
            self._q.put(item)

    def _make_speak(self):
        """Returns speak(text, aborted), or None when speech is off or unavailable."""
        if not self._enabled:
            return None
        if sys.platform == "win32":
            # pyttsx3 stays silent after its first utterance in a process on Windows, so use SAPI
            # directly. One persistent voice lets the purge flag interrupt speech already playing.
            try:
                self._sapi = _SapiWorker()
            except OSError as exc:
                log.warning("TTS unavailable (%s); speech will be logged only", exc)
                return None
            self._cancel_hook = self._sapi_cancel

            def speak_sapi(text: str, aborted: Callable[[], bool]) -> None:
                if self._sapi is None:
                    self._sapi = _SapiWorker()
                try:
                    self._sapi.speak(text, self._rate, self._volume, aborted)
                except Exception:
                    self._sapi.close()
                    self._sapi = None   # respawned on the next utterance
                    raise

            return speak_sapi
        try:
            import pyttsx3

            pyttsx3.init()  # fail early if no engine is installed
        except Exception as exc:
            log.warning("TTS unavailable (%s); speech will be logged only", exc)
            return None

        def speak(text: str, aborted: Callable[[], bool]) -> None:
            engine = pyttsx3.init()  # fresh engine per utterance; reuse goes silent
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            if self._voice:
                for v in engine.getProperty("voices"):
                    if self._voice.lower() in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break
            engine.say(text)
            with self._cv:
                if aborted():
                    return
                self._cancel_hook = engine.stop
            try:
                engine.runAndWait()
            finally:
                with self._cv:
                    self._cancel_hook = None

        return speak

    def _sapi_cancel(self) -> None:
        if self._sapi is not None:
            self._sapi.stop()

    def _run(self) -> None:
        speak = self._make_speak()
        while True:
            with self._cv:
                while self._q.empty() and not self._stop.is_set():
                    self._cv.wait(0.2)
                if self._stop.is_set():
                    return
                item = self._q.get_nowait()
                gen, epoch = self._gen, self._epoch
                self._current_priority = item[0]
            priority, _, queued_at, text = item

            def aborted(started_gen: int = gen) -> bool:
                return self._gen != started_gen

            if priority == PRIORITY_INFO and time.monotonic() - queued_at > 8.0:
                log.info("skipping stale status message: %s", text)  # answers are aged in the app instead
            else:
                self.last_spoken = text
                log.info("SPEAK[%d]: %s", priority, text)
                if speak is not None and not aborted():
                    try:
                        speak(text, aborted)
                    except Exception as exc:
                        log.warning("TTS error: %s", exc)
            with self._cv:
                self._current_priority = None
                if aborted() and self._epoch == epoch and priority == PRIORITY_CLOUD:
                    self._q.put(item)   # a safety warning cut the answer off: replay it afterwards

    def close(self) -> None:
        self.stop()
        with self._cv:
            self._stop.set()
            self._cv.notify_all()
        if self._sapi is not None:
            self._sapi.close()


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
