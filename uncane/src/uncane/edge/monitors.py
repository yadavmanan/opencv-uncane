"""Rate-limited camera health checks and an independent frame-loop watchdog."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np


class FrameMonitor:
    def __init__(self, cooldown_s: float = 20.0, clock=time.monotonic):
        self.cooldown_s, self.clock = cooldown_s, clock
        self._last_message: Dict[str, float] = {}
        self._previous_gray: Optional[np.ndarray] = None
        self._still_frames = 0

    def inspect(self, frame: np.ndarray, calibrated: bool = True) -> List[str]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        glare = float(np.mean(gray >= 250))
        issues = []
        if brightness < 28:
            issues.append(("dark", "The camera image is very dark."))
        elif brightness > 235:
            issues.append(("bright", "The camera image is overexposed."))
        if contrast < 12:
            issues.append(("contrast", "The camera image has very low contrast."))
        if glare > 0.18:
            issues.append(("glare", "There is strong glare in the camera image."))
        if self._previous_gray is not None:
            change = float(np.mean(cv2.absdiff(gray, self._previous_gray)))
            self._still_frames = self._still_frames + 1 if change < 0.05 else 0
            if self._still_frames >= 30:
                issues.append(("frozen", "The camera image appears frozen."))
        if not calibrated:
            issues.append(("calibration", "Floor distance calibration is not ready."))
        now = self.clock()
        messages = []
        for key, message in issues:
            if now - self._last_message.get(key, -1e9) >= self.cooldown_s:
                self._last_message[key] = now
                messages.append(message)
        self._previous_gray = gray
        return messages


class FrameWatchdog:
    def __init__(self, max_stall_s: float, on_stall: Callable[[], None], interval_s: float = 1.0):
        self.max_stall_s, self.on_stall, self.interval_s = max_stall_s, on_stall, interval_s
        self._last_frame = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="frame-watchdog", daemon=True)

    def start(self) -> "FrameWatchdog":
        self._thread.start()
        return self

    def beat(self) -> None:
        self._last_frame = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 0.5)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            if time.monotonic() - self._last_frame > self.max_stall_s:
                self.on_stall()
                self._last_frame = time.monotonic()
