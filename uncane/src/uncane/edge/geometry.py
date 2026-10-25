"""Body-mounted camera geometry and floor-plane scale recovery.

Monocular networks like MiDaS predict inverse depth only up to an unknown
scale and shift: 1/Z = a * r + b. A chest camera at a known height and tilt
sees the floor in the lower part of the image, and the floor's true depth at
every row follows from trigonometry. Fitting (a, b) to those floor pixels each
frame turns relative depth into meters without a depth sensor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Points3D:
    Z: np.ndarray        # depth along the optical axis (m)
    X: np.ndarray        # lateral offset, + right (m)
    height: np.ndarray   # height above the floor (m)
    forward: np.ndarray  # horizontal distance ahead of the body (m)


class CameraGeometry:
    def __init__(self, hfov_deg: float, frame_w: int, frame_h: int, height_m: float, pitch_deg: float):
        self.tan_hx = math.tan(math.radians(hfov_deg) / 2.0)
        self.tan_hy = self.tan_hx * frame_h / frame_w
        self.height_m = height_m
        self.pitch = math.radians(pitch_deg)
        self._cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def rays(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-pixel normalized ray slopes (tx right, ty down) and expected floor depth."""
        key = (h, w)
        if key not in self._cache:
            tx = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * self.tan_hx
            ty = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * 2.0 * self.tan_hy
            denom = ty * math.cos(self.pitch) + math.sin(self.pitch)
            with np.errstate(divide="ignore"):
                z_floor = np.where(denom > 0.02, self.height_m / denom, np.inf).astype(np.float32)
            self._cache[key] = (tx[None, :], ty[:, None], z_floor[:, None])
        return self._cache[key]

    def backproject(self, Z: np.ndarray) -> Points3D:
        h, w = Z.shape
        tx, ty, _ = self.rays(h, w)
        X = Z * tx
        Y = Z * ty
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        height = self.height_m - (Y * c + Z * s)
        forward = Z * c - Y * s
        return Points3D(Z=Z, X=X, height=height, forward=forward)


class GroundPlaneCalibrator:
    """Online estimate of (a, b) in 1/Z = a * r + b from floor pixels.

    Robust to obstacles covering part of the floor band: fits, trims the worst
    residuals (obstacles, walls), refits, and blends into a running estimate.
    """

    def __init__(
        self,
        geometry: CameraGeometry,
        band: Tuple[float, float] = (0.70, 0.98),
        cols: Tuple[float, float] = (0.2, 0.8),
        z_range: Tuple[float, float] = (0.6, 8.0),
        ema: float = 0.25,
        min_samples: int = 60,
    ):
        self.geometry = geometry
        self.band, self.cols, self.z_range = band, cols, z_range
        self.ema, self.min_samples = ema, min_samples
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.last_fit_ok = False
        self.frames_ok = 0

    @property
    def calibrated(self) -> bool:
        return self.a is not None and self.frames_ok >= 3

    def update(self, r: np.ndarray) -> bool:
        h, w = r.shape
        _, _, z_floor = self.geometry.rays(h, w)
        r0, r1 = int(self.band[0] * h), int(self.band[1] * h)
        c0, c1 = int(self.cols[0] * w), int(self.cols[1] * w)
        zf = np.broadcast_to(z_floor[r0:r1], (r1 - r0, c1 - c0))
        rr = r[r0:r1, c0:c1]
        valid = np.isfinite(zf) & (zf > self.z_range[0]) & (zf < self.z_range[1])
        x = rr[valid].astype(np.float64)
        y = 1.0 / zf[valid].astype(np.float64)
        self.last_fit_ok = False
        if x.size < self.min_samples or np.ptp(x) < 1e-6:
            return False
        keep = np.ones_like(x, dtype=bool)
        a = b = 0.0
        for _ in range(3):
            A = np.stack([x[keep], np.ones(keep.sum())], axis=1)
            (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
            resid = np.abs(a * x + b - y)
            keep = resid <= np.percentile(resid, 60)
            if keep.sum() < self.min_samples // 2:
                return False
        if a <= 0:
            return False
        rel_err = np.median(np.abs(a * x[keep] + b - y[keep]) / y[keep])
        if rel_err > 0.15:
            return False
        if self.a is not None and self.frames_ok >= 3:
            # Reject sudden jumps (e.g. a person steps into the floor band).
            ratio = a / self.a
            if not 0.5 < ratio < 2.0:
                return False
            self.a += self.ema * (a - self.a)
            self.b += self.ema * (b - self.b)
        else:
            self.a, self.b = float(a), float(b)
        self.frames_ok += 1
        self.last_fit_ok = True
        return True

    def to_metric(self, r: np.ndarray, max_range_m: float = 20.0) -> np.ndarray:
        if self.a is None:
            raise RuntimeError("calibrator has no estimate yet")
        inv = self.a * r + self.b
        return np.where(inv > 1.0 / max_range_m, 1.0 / np.maximum(inv, 1e-6), max_range_m).astype(np.float32)
