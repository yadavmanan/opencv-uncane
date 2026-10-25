"""Reflex arbiter: turns noisy per-frame detections into calm, trustworthy alerts.

Keeps the false-positive rate low (success criterion from the brief): a hazard
must be seen in ``confirm_frames`` of the last ``window`` frames before it
alerts, and must be gone for ``release_frames`` before it is released. Only
the single most urgent hazard is sonified so the user isn't flooded.
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from ..common.schemas import HazardItem, ThreatItem

Key = Tuple[str, str]  # (zone, level)
_RANK = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass
class ToneCue:
    """What the spatial beeper should play right now."""

    pan: float          # -1 left .. +1 right
    freq_hz: float      # pitch encodes height: low = ground, high = head
    interval_s: float   # beep period: shorter = closer
    urgency: str


@dataclass
class ReflexOutput:
    tone: Optional[ToneCue]
    speech: List[str]
    active_hazards: List[HazardItem]
    active_threats: List[ThreatItem]


class ReflexArbiter:
    PAN = {"LEFT": -0.85, "CENTER": 0.0, "RIGHT": 0.85}
    PITCH = {"GROUND": 440.0, "TORSO": 660.0, "HEAD": 1100.0}

    def __init__(self, confirm_frames: int = 2, window: int = 3, release_frames: int = 4,
                 speech_cooldown_s: float = 4.0, clock=time.monotonic):
        self.confirm, self.window, self.release = confirm_frames, window, release_frames
        self.cooldown = speech_cooldown_s
        self.clock = clock
        self.history: Dict[Key, Deque[bool]] = collections.defaultdict(lambda: collections.deque(maxlen=window))
        self.absent: Dict[Key, int] = collections.defaultdict(int)
        self.active: Dict[Key, HazardItem] = {}
        self.last_spoken: Dict[str, float] = {}

    def update(self, hazards: Sequence[HazardItem], threats: Sequence[ThreatItem]) -> ReflexOutput:
        seen = {(h.zone, h.level): h for h in hazards}
        for key in set(self.history) | set(seen):
            self.history[key].append(key in seen)
            if key in seen:
                self.absent[key] = 0
                if sum(self.history[key]) >= self.confirm:
                    self.active[key] = seen[key]
            else:
                self.absent[key] += 1
                if self.absent[key] >= self.release:
                    self.active.pop(key, None)

        speech: List[str] = []
        active = sorted(self.active.values(), key=self._severity, reverse=True)
        tone = self._tone(active[0]) if active else None

        for h in active:
            if h.urgency != "CRITICAL":
                continue
            if h.zone == "CENTER" and h.level == "HEAD":
                self._say(speech, "head", f"Head height {h.label or 'obstacle'} ahead. Duck or step aside.")
            elif h.zone == "CENTER":
                self._say(speech, "stop", f"Stop. {(h.label or 'Obstacle').capitalize()} ahead.")

        for t in sorted(threats, key=lambda t: t.ttc_s):
            side = {"LEFT": "left", "RIGHT": "right", "CENTER": "ahead"}[t.zone]
            what = (t.label or "something").capitalize()
            if t.urgency == "CRITICAL":
                self._say(speech, f"threat-{t.zone}", f"{what} coming, {side}!")
                tone = ToneCue(pan=self.PAN[t.zone], freq_hz=1500.0, interval_s=0.08, urgency="CRITICAL")
            break
        return ReflexOutput(tone, speech, active, list(threats))

    def _severity(self, h: HazardItem) -> Tuple[int, int, float]:
        dist = h.distance_m if h.distance_m is not None else 3.0 * (1 - h.proximity)
        return (_RANK[h.urgency], h.zone == "CENTER", -dist)

    def _tone(self, h: HazardItem) -> ToneCue:
        dist = h.distance_m if h.distance_m is not None else 3.0 * (1.0 - h.proximity)
        interval = min(0.8, max(0.08, 0.1 + 0.3 * dist))
        return ToneCue(self.PAN[h.zone], self.PITCH[h.level], interval, h.urgency)

    def _say(self, out: List[str], key: str, text: str) -> None:
        now = self.clock()
        if now - self.last_spoken.get(key, -1e9) >= self.cooldown:
            self.last_spoken[key] = now
            out.append(text)
