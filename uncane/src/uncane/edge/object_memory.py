"""Short-lived, in-memory sightings for answering local object-location queries."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from ..common.spatial import bearing_from_x, clock_from_bearing
from .detector import Detection


ALIASES = {
    "phone": {"cell phone", "phone", "mobile phone"},
    "water bottle": {"bottle", "water bottle"},
    "bag": {"backpack", "handbag", "suitcase"},
    "keys": {"keys", "key"},
}


@dataclass
class Sighting:
    label: str
    clock: int
    bearing_deg: float
    distance_m: Optional[float]
    seen_at: float


class ObjectMemory:
    def __init__(self, retention_s: float = 120.0, clock=time.monotonic):
        self.retention_s = retention_s
        self.clock = clock
        self._sightings: Dict[str, Sighting] = {}

    def update(self, detections: Iterable[Detection], hfov_deg: float,
               depth: Optional[np.ndarray] = None, now: Optional[float] = None) -> None:
        seen_at = self.clock() if now is None else now
        for detection in detections:
            bearing = bearing_from_x(detection.cx, hfov_deg)
            self._sightings[detection.label.lower()] = Sighting(
                detection.label, clock_from_bearing(bearing), bearing,
                self._distance(detection, depth), seen_at,
            )
        self.prune(seen_at)

    def find(self, requested: str, now: Optional[float] = None) -> Optional[Sighting]:
        self.prune(self.clock() if now is None else now)
        query = requested.lower().strip(" .,!?")
        if query.startswith("my "):
            query = query[3:]
        matching = set(ALIASES.get(query, {query}))
        candidates = [sighting for label, sighting in self._sightings.items()
                      if label in matching or any(alias in label for alias in matching)]
        return max(candidates, key=lambda sighting: sighting.seen_at) if candidates else None

    def prune(self, now: Optional[float] = None) -> None:
        current = self.clock() if now is None else now
        self._sightings = {label: sighting for label, sighting in self._sightings.items()
                           if current - sighting.seen_at <= self.retention_s}

    @staticmethod
    def describe(sighting: Optional[Sighting]) -> str:
        if sighting is None:
            return "I haven't seen that recently."
        elapsed = max(0, int(time.monotonic() - sighting.seen_at))
        clock = "12" if sighting.clock == 12 else str(sighting.clock)
        distance = f", about {sighting.distance_m:.1f} meters away" if sighting.distance_m is not None else ""
        age = "just now" if elapsed < 10 else f"{elapsed} seconds ago"
        return f"I last saw the {sighting.label} at {clock} o'clock{distance}, {age}."

    @staticmethod
    def _distance(detection: Detection, depth: Optional[np.ndarray]) -> Optional[float]:
        if depth is None or depth.size == 0:
            return None
        h, w = depth.shape[:2]
        x0, y0, x1, y1 = detection.box
        roi = depth[max(0, int(y0 * h)):min(h, max(1, int(y1 * h))),
                    max(0, int(x0 * w)):min(w, max(1, int(x1 * w)))]
        valid = roi[np.isfinite(roi) & (roi > 0.1) & (roi < 20.0)]
        return float(np.median(valid)) if valid.size else None
