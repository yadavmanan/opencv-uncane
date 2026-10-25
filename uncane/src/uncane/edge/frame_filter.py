"""Keyframe selection: only spend bandwidth, battery and Bedrock tokens on frames
that carry new information, and send the sharpest recent frame with a question.
"""
from __future__ import annotations

import collections
import time
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity of two same-size grayscale images."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    blur = lambda x: cv2.GaussianBlur(x, (7, 7), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a * mu_a
    var_b = blur(b * b) - mu_b * mu_b
    cov = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + _C1) * (2 * cov + _C2)
    den = (mu_a ** 2 + mu_b ** 2 + _C1) * (var_a + var_b + _C2)
    return float(np.mean(num / den))


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low = motion blur / defocus."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


class KeyframeFilter:
    """Decides when the scene changed enough to be worth a cloud call."""

    def __init__(self, ssim_threshold: float = 0.80, min_interval_s: float = 3.0, size=(160, 120)):
        self.ssim_threshold = ssim_threshold
        self.min_interval_s = min_interval_s
        self.size = size
        self.last_sent: Optional[np.ndarray] = None
        self.last_sent_t = -1e9
        self.last_score = 1.0

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(g, self.size, interpolation=cv2.INTER_AREA)

    def should_send(self, frame: np.ndarray, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        g = self._prep(frame)
        if self.last_sent is None:
            self.last_sent, self.last_sent_t, self.last_score = g, now, 0.0
            return True
        if now - self.last_sent_t < self.min_interval_s:
            return False
        self.last_score = ssim(g, self.last_sent)
        if self.last_score < self.ssim_threshold:
            self.last_sent, self.last_sent_t = g, now
            return True
        return False


class SharpFrameBuffer:
    """Ring buffer of recent frames; ``best()`` returns the least blurry one.

    Photos taken while walking are often blurred (the VizWiz dataset shows how
    common unusable photos are); picking the sharpest of the last ~0.3 s is a
    cheap, large quality win for OCR and object lookup.
    """

    def __init__(self, size: int = 8):
        self.frames: Deque[Tuple[float, float, np.ndarray]] = collections.deque(maxlen=size)

    def push(self, frame: np.ndarray, t: float) -> None:
        g = cv2.cvtColor(cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        self.frames.append((sharpness(g), t, frame))

    def best(self) -> Optional[Tuple[np.ndarray, float]]:
        if not self.frames:
            return None
        score, t, frame = max(self.frames, key=lambda f: f[0])
        return frame, t


def encode_jpeg(frame: np.ndarray, quality: int = 80, max_side: int = 768) -> bytes:
    h, w = frame.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s < 1.0:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
