"""Local homing guidance for a cloud-identified target."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..common.spatial import STEP_LENGTH_M, bearing_from_x, clock_from_bearing, steps_phrase
from .detector import Detection


@dataclass
class GuidanceCue:
    message: str
    bearing_deg: Optional[float] = None
    distance_m: Optional[float] = None
    state: str = "tracking"


class GuidanceTracker:
    def __init__(self, timeout_s: float = 180.0, lost_after_s: float = 5.0,
                 arrival_distance_m: float = 0.7, clock=time.monotonic):
        self.timeout_s, self.lost_after_s = timeout_s, lost_after_s
        self.arrival_distance_m, self.clock = arrival_distance_m, clock
        self.label = ""
        self.started_at = 0.0
        self.last_seen_at = 0.0
        self.bbox: Optional[Sequence[float]] = None
        self.template: Optional[np.ndarray] = None
        self.active = False
        self._lost_reported = False

    def start(self, label: str, bbox: Optional[Sequence[float]], frame: Optional[np.ndarray] = None) -> None:
        now = self.clock()
        self.label, self.bbox = label.lower(), bbox
        self.started_at = self.last_seen_at = now
        self.active = True
        self._lost_reported = False
        self.template = self._crop(frame, bbox) if frame is not None and bbox else None

    def stop(self) -> None:
        self.active = False
        self.template = None

    def update(self, frame: np.ndarray, detections: Sequence[Detection], hfov_deg: float,
               depth: Optional[np.ndarray] = None) -> Optional[GuidanceCue]:
        if not self.active:
            return None
        now = self.clock()
        if now - self.started_at > self.timeout_s:
            self.stop()
            return GuidanceCue("Target guidance timed out.", state="timeout")
        match = self._match_detection(detections)
        center_x = None
        if match is not None:
            center_x = match.cx
            self.bbox = match.box
            self.last_seen_at = now
        elif self.template is not None:
            found = self._match_template(frame)
            if found is not None:
                center_x, self.bbox = found
                self.last_seen_at = now
        if center_x is None:
            if now - self.last_seen_at > self.lost_after_s and not self._lost_reported:
                self._lost_reported = True
                return GuidanceCue(f"I lost sight of the {self.label}.", state="lost")
            return None
        self._lost_reported = False
        bearing = bearing_from_x(center_x, hfov_deg)
        distance = self._distance(self.bbox, depth)
        if distance is not None and distance <= self.arrival_distance_m:
            self.stop()
            return GuidanceCue(f"You have reached the {self.label}.", bearing, distance, "arrived")
        side = "slightly left" if bearing < -8 else "slightly right" if bearing > 8 else "straight ahead"
        clock = clock_from_bearing(bearing)
        distance_text = f", about {distance:.1f} meters" if distance is not None else ""
        steps_text = f", {steps_phrase(round(distance / STEP_LENGTH_M))}" if distance is not None else ""
        return GuidanceCue(f"The {self.label} is at {clock} o'clock, {side}{distance_text}{steps_text}.",
                   bearing, distance)

    def _match_detection(self, detections: Sequence[Detection]) -> Optional[Detection]:
        target = self.label.split()[-1]
        candidates = [d for d in detections if target in d.label.lower() or d.label.lower() in self.label]
        return max(candidates, key=lambda d: d.score) if candidates else None

    def _match_template(self, frame: np.ndarray):
        if self.template is None or min(self.template.shape[:2]) < 8:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
            return None
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score < 0.55:
            return None
        x = (location[0] + template.shape[1] / 2) / gray.shape[1]
        y0, x0 = location[1] / gray.shape[0], location[0] / gray.shape[1]
        y1 = (location[1] + template.shape[0]) / gray.shape[0]
        x1 = (location[0] + template.shape[1]) / gray.shape[1]
        return x, [x0, y0, x1, y1]

    @staticmethod
    def _crop(frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = [max(0.0, min(1.0, float(v))) for v in bbox]
        crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        return crop.copy() if crop.size else None

    @staticmethod
    def _distance(bbox: Optional[Sequence[float]], depth: Optional[np.ndarray]) -> Optional[float]:
        if bbox is None or depth is None or depth.size == 0:
            return None
        h, w = depth.shape[:2]
        x0, y0, x1, y1 = bbox
        roi = depth[max(0, int(y0 * h)):min(h, max(1, int(y1 * h))),
                    max(0, int(x0 * w)):min(w, max(1, int(x1 * w)))]
        valid = roi[np.isfinite(roi) & (roi > 0.1) & (roi < 20.0)]
        return float(np.median(valid)) if valid.size else None
