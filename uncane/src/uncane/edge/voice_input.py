"""Voice-activated querying.

Backends:
  * keyboard - type a question in the terminal (development / judging fallback)
  * vosk     - offline speech-to-text; push-to-talk records `record_seconds`
               of microphone audio when triggered (key 'v' or a hardware button)
Preset questions (keys 1-9) make live demos deterministic.
"""
from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from typing import Callable, Optional

log = logging.getLogger(__name__)

QueryCallback = Callable[[str], None]


class KeyboardVoice:
    """Reads questions from stdin on a background thread."""

    def __init__(self, on_query: QueryCallback):
        self.on_query = on_query
        threading.Thread(target=self._run, name="stdin-voice", daemon=True).start()

    def _run(self) -> None:
        print("Type a question and press Enter (e.g. 'where is the door?'):", flush=True)
        for line in sys.stdin:
            text = line.strip()
            if text:
                self.on_query(text)

    def push_to_talk(self) -> None:
        log.info("keyboard backend: type your question in the terminal")


class VoskVoice:
    """Offline push-to-talk speech recognition with Vosk + sounddevice."""

    SAMPLE_RATE = 16000

    def __init__(self, on_query: QueryCallback, model_path: str, record_seconds: float = 4.0):
        import sounddevice as sd  # noqa: F401  (fail fast if missing)
        from vosk import KaldiRecognizer, Model

        self.on_query = on_query
        self.record_seconds = record_seconds
        self.model = Model(model_path)
        self._recognizer_cls = KaldiRecognizer
        self._busy = threading.Lock()
        KeyboardVoice(on_query)  # typing still works as a fallback

    def push_to_talk(self) -> None:
        if not self._busy.acquire(blocking=False):
            return
        threading.Thread(target=self._record, name="vosk", daemon=True).start()

    def _record(self) -> None:
        import sounddevice as sd

        try:
            rec = self._recognizer_cls(self.model, self.SAMPLE_RATE)
            q: "queue.Queue[bytes]" = queue.Queue()
            with sd.RawInputStream(samplerate=self.SAMPLE_RATE, blocksize=4000, dtype="int16", channels=1,
                                   callback=lambda data, *_: q.put(bytes(data))):
                log.info("listening for %.1fs...", self.record_seconds)
                for _ in range(int(self.record_seconds * self.SAMPLE_RATE / 4000)):
                    rec.AcceptWaveform(q.get())
            text = json.loads(rec.FinalResult()).get("text", "").strip()
            log.info("heard: %r", text)
            if text:
                self.on_query(text)
        except Exception as exc:
            log.warning("voice capture failed: %s", exc)
        finally:
            self._busy.release()


def make_voice(backend: str, on_query: QueryCallback, model_path: Optional[str] = None,
               record_seconds: float = 4.0):
    if backend == "vosk":
        try:
            return VoskVoice(on_query, model_path or "", record_seconds)
        except Exception as exc:
            log.warning("vosk unavailable (%s); falling back to keyboard input", exc)
    return KeyboardVoice(on_query)
