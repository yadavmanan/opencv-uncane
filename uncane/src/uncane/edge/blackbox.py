"""Short in-memory event history with explicit local JSONL export."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import cv2
import numpy as np


class BlackBox:
    def __init__(self, path: str, capacity: int = 300, save_snapshots: bool = False):
        self.path = Path(path)
        self.events: Deque[Dict[str, Any]] = deque(maxlen=max(1, capacity))
        self.save_snapshots = save_snapshots
        self._lock = threading.Lock()

    def record(self, event: str, details: Optional[Dict[str, Any]] = None,
               frame: Optional[np.ndarray] = None, export: bool = False) -> Dict[str, Any]:
        item: Dict[str, Any] = {"timestamp": time.time(), "event": event, "details": details or {}}
        with self._lock:
            self.events.append(item)
            if export:
                self._export_locked(item, frame)
        return item

    def export(self, event: str = "manual", frame: Optional[np.ndarray] = None) -> Path:
        with self._lock:
            item = {"timestamp": time.time(), "event": event,
                    "history": list(self.events)}
            return self._export_locked(item, frame)

    def _export_locked(self, item: Dict[str, Any], frame: Optional[np.ndarray]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")
        if self.save_snapshots and frame is not None:
            snapshot = self.path.with_name(f"{self.path.stem}-{int(item['timestamp'] * 1000)}.jpg")
            cv2.imwrite(str(snapshot), frame)
        return self.path
