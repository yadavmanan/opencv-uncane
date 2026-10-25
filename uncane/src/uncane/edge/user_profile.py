"""Local preferences and personal notes, persisted with atomic replacement."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import copy
from pathlib import Path
from typing import Any, Dict


DEFAULT_PROFILE: Dict[str, Any] = {
    "environment": "outdoor",
    "quiet_mode": False,
    "speech_rate": 190,
    "speech_volume": 1.0,
    "notes": [],
    "feedback_count": 0,
}


class UserProfile:
    def __init__(self, path: str, defaults: Dict[str, Any] | None = None):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.defaults = copy.deepcopy(DEFAULT_PROFILE)
        self.defaults.update(defaults or {})
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("profile must be a JSON object")
            merged = {**self.defaults, **value}
            notes = merged.get("notes", [])
            merged["notes"] = [str(note) for note in notes if str(note).strip()] if isinstance(notes, list) else []
            merged["speech_rate"] = max(80, min(300, int(merged["speech_rate"])))
            merged["speech_volume"] = max(0.0, min(1.0, float(merged["speech_volume"])))
            merged["feedback_count"] = max(0, int(merged["feedback_count"]))
            merged["quiet_mode"] = bool(merged["quiet_mode"])
            merged["environment"] = "indoor" if merged["environment"] == "indoor" else "outdoor"
            return merged
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return copy.deepcopy(self.defaults)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(self.data, stream, indent=2, ensure_ascii=True)
                    stream.write("\n")
                os.replace(temp_path, self.path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def update(self, **values: Any) -> None:
        with self._lock:
            self.data.update(values)
            self.save()

    def add_note(self, note: str) -> None:
        text = note.strip()
        if text:
            with self._lock:
                self.data["notes"].append(text)
                self.data["notes"] = self.data["notes"][-100:]
                self.save()

    def clear_notes(self) -> None:
        self.update(notes=[])
