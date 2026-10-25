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

Key = Tuple[str, str, str]  # (zone, level, obstacle or drop-off)
_RANK = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass
class ToneCue:
    """What the spatial beeper should play right now."""

    pan: float          # -1 left .. +1 right
    freq_hz: float      # pitch encodes height: low = ground, high = head
    interval_s: float   # beep period: shorter = closer
    urgency: str
    kind: str = "obstacle"


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
                 speech_cooldown_s: float = 4.0, clock=time.monotonic,
                 dropoff_release_frames: int = 8, quiet_mode: bool = False,
                 crowd_mode: bool = False, crowd_alert_distance_m: float = 1.0,
                 dropoff_confirm_frames: Optional[int] = None, dropoff_window: Optional[int] = None):
        self.confirm, self.window, self.release = confirm_frames, window, release_frames
        self.dropoff_confirm = dropoff_confirm_frames or confirm_frames
        self.dropoff_window = dropoff_window or window
        self.cooldown = speech_cooldown_s
        self.dropoff_release = max(release_frames, dropoff_release_frames)
        self.quiet_mode, self.crowd_mode = quiet_mode, crowd_mode
        self.crowd_alert_distance_m = crowd_alert_distance_m
        self.clock = clock
        history_len = max(window, self.dropoff_window)
        self.history: Dict[Key, Deque[bool]] = collections.defaultdict(
            lambda: collections.deque(maxlen=history_len))
        self.absent: Dict[Key, int] = collections.defaultdict(int)
        self.active: Dict[Key, HazardItem] = {}
        self.last_spoken: Dict[str, float] = {}

    def set_modes(self, quiet_mode: Optional[bool] = None, crowd_mode: Optional[bool] = None,
                  crowd_alert_distance_m: Optional[float] = None) -> None:
        if quiet_mode is not None:
            self.quiet_mode = quiet_mode
        if crowd_mode is not None:
            self.crowd_mode = crowd_mode
        if crowd_alert_distance_m is not None:
            self.crowd_alert_distance_m = crowd_alert_distance_m

    def update(self, hazards: Sequence[HazardItem], threats: Sequence[ThreatItem]) -> ReflexOutput:
        if self.crowd_mode:
            hazards = [h for h in hazards if not (
                h.label == "person" and h.distance_m is not None
                and h.distance_m > self.crowd_alert_distance_m
            )]
        seen = {(h.zone, h.level, "drop-off" if h.label == "drop-off" else "obstacle"): h
            for h in hazards}
        for key in set(self.history) | set(seen):
            self.history[key].append(key in seen)
            if key in seen:
                self.absent[key] = 0
                need, span = ((self.dropoff_confirm, self.dropoff_window) if key[2] == "drop-off"
                              else (self.confirm, self.window))
                if sum(list(self.history[key])[-span:]) >= need:
                    self.active[key] = seen[key]
            else:
                self.absent[key] += 1
                active_hazard = self.active.get(key)
                required = (self.dropoff_release if active_hazard and active_hazard.label == "drop-off"
                            else self.release)
                if self.absent[key] >= required:
                    self.active.pop(key, None)

        speech: List[str] = []
        active = sorted(self.active.values(), key=self._severity, reverse=True)
        tone = self._tone(active[0]) if active else None

        for h in active:
            if h.label == "drop-off":
                side = {"LEFT": "on your left", "RIGHT": "on your right", "CENTER": "ahead"}[h.zone]
                urgency_key = "dropoff-critical" if h.urgency == "CRITICAL" else "dropoff-warning"
                if not self.quiet_mode or h.urgency == "CRITICAL":
                    self._say(speech, urgency_key, f"Drop-off {side}. Stop and check your footing.")
                continue
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
        if self.quiet_mode and tone is not None and tone.urgency != "CRITICAL":
            tone = None
        return ReflexOutput(tone, speech, active, list(threats))

    def _severity(self, h: HazardItem) -> Tuple[int, int, float]:
        dist = h.distance_m if h.distance_m is not None else 3.0 * (1 - h.proximity)
        return (_RANK[h.urgency], h.zone == "CENTER", -dist)

    def _tone(self, h: HazardItem) -> ToneCue:
        dist = h.distance_m if h.distance_m is not None else 3.0 * (1.0 - h.proximity)
        interval = min(0.8, max(0.08, 0.1 + 0.3 * dist))
        if h.label == "drop-off":
            return ToneCue(self.PAN[h.zone], 520.0, max(0.25, interval), h.urgency, "dropoff")
        return ToneCue(self.PAN[h.zone], self.PITCH[h.level], interval, h.urgency)

    def _say(self, out: List[str], key: str, text: str) -> None:
        now = self.clock()
        if now - self.last_spoken.get(key, -1e9) >= self.cooldown:
            self.last_spoken[key] = now
            out.append(text)
