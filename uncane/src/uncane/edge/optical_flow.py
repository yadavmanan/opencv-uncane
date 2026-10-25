"""Approaching-object detection from sparse optical flow (looming / time-to-contact).

Plain flow magnitude fires constantly on a walking body because the whole image
moves, and per-point thresholds miss real threats: a cyclist 3 m away grows by
~5 % per frame, i.e. only ~1 px of motion. Instead we:
  1. spread corners over a grid and track them with pyramidal Lucas-Kanade,
     rejecting tracks that fail a forward-backward consistency check;
  2. fit the camera's own motion (similarity transform, RANSAC);
  3. in overlapping windows, measure how much faster the tracked points spread
     apart than the camera model predicts (local divergence = looming);
  4. subtract the scene-wide median divergence (walking forward makes the
     whole scene expand), smooth per zone, and require persistence.
Divergence rate e (1/s) gives time-to-contact TTC ~= 1 / e.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..common.schemas import ThreatItem
from ..common.spatial import bearing_from_x

ZONES = ("LEFT", "CENTER", "RIGHT")


@dataclass
class _ZoneState:
    ema_rate: float = 0.0
    hits: int = 0


@dataclass
class FlowDebug:
    movers: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    ego_scale: float = 1.0
    background_rate: float = 0.0
    zone_rates: Dict[str, float] = field(default_factory=dict)


class ThreatTracker:
    GRID = (8, 6)        # feature buckets (cols, rows)
    WINDOW = 2           # window = WINDOW x WINDOW buckets, stride 1 bucket

    def __init__(
        self,
        hfov_deg: float = 70.0,
        proc_width: int = 320,
        max_corners: int = 500,
        redetect_every: int = 4,
        ttc_warning_s: float = 3.0,
        ttc_critical_s: float = 1.5,
        min_cluster_points: int = 8,
        persist_frames: int = 2,
        fb_max_err: float = 0.7,
    ):
        self.hfov_deg = hfov_deg
        self.proc_width = proc_width
        self.max_corners = max(max_corners, 100)
        self.redetect_every = redetect_every
        self.ttc_warning_s = ttc_warning_s
        self.ttc_critical_s = ttc_critical_s
        self.min_points = min_cluster_points
        self.persist_frames = persist_frames
        self.fb_max_err = fb_max_err
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_t: Optional[float] = None
        self.frame_idx = 0
        self.zones: Dict[str, _ZoneState] = {z: _ZoneState() for z in ZONES}
        self.debug = FlowDebug()

    def reset(self) -> None:
        self.prev_gray = self.prev_pts = self.prev_t = None
        for z in self.zones.values():
            z.ema_rate, z.hits = 0.0, 0

    # ---------------------------------------------------------------- features
    def _gray(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        size = (self.proc_width, int(round(h * self.proc_width / float(w))))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    def _detect(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Corners spread evenly over a grid so small objects still get tracked."""
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners * 4, qualityLevel=0.005,
                                      minDistance=4, blockSize=5)
        if pts is None:
            return None
        pts = pts.reshape(-1, 2)
        gx, gy = self.GRID
        h, w = gray.shape
        bucket = (np.minimum(pts[:, 0] * gx / w, gx - 1).astype(int)
                  + gx * np.minimum(pts[:, 1] * gy / h, gy - 1).astype(int))
        per_bucket = max(4, self.max_corners // (gx * gy))
        keep = np.zeros(len(pts), bool)
        counts = np.zeros(gx * gy, int)
        for i, b in enumerate(bucket):         # pts are sorted by corner quality
            if counts[b] < per_bucket:
                counts[b] += 1
                keep[i] = True
        return pts[keep].reshape(-1, 1, 2).astype(np.float32)

    # ------------------------------------------------------------------ update
    def update(self, frame: np.ndarray, t: float) -> List[ThreatItem]:
        gray = self._gray(frame)
        self.frame_idx += 1
        threats: List[ThreatItem] = []
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= 10:
            dt = max(1e-3, t - (self.prev_t if self.prev_t is not None else t - 1 / 30))
            threats = self._step(self.prev_gray, gray, self.prev_pts, dt)
        if self.prev_pts is None or len(self.prev_pts) < self.max_corners // 3 \
                or self.frame_idx % self.redetect_every == 0:
            self.prev_pts = self._detect(gray)
        self.prev_gray, self.prev_t = gray, t
        return threats

    def _step(self, g0: np.ndarray, g1: np.ndarray, p0: np.ndarray, dt: float) -> List[ThreatItem]:
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **self.lk_params)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **self.lk_params)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_max_err)
        a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        self.prev_pts = b.reshape(-1, 1, 2).astype(np.float32)
        if len(a) < 20:
            self._decay_all()
            return []
        M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                           maxIters=500, confidence=0.99)
        if M is None:
            self._decay_all()
            return []
        pred = a @ M[:, :2].T + M[:, 2]

        windows = self._window_rates(pred, b, g1.shape, dt)
        if not windows:
            self._decay_all()
            return []
        rates = np.array([w[0] for w in windows])
        background = float(np.median(rates))
        w_img = g1.shape[1]
        zone_best: Dict[str, Tuple[float, float]] = {}
        for rate, cx, _ in windows:
            zone = ZONES[min(2, int(3 * cx / w_img))]
            excess = rate - max(background, 0.0)
            if excess > zone_best.get(zone, (0.0, 0.0))[0]:
                zone_best[zone] = (excess, cx)

        threats: List[ThreatItem] = []
        movers = []
        for zone in ZONES:
            state = self.zones[zone]
            excess, cx = zone_best.get(zone, (0.0, (ZONES.index(zone) + 0.5) * w_img / 3))
            state.ema_rate = 0.6 * state.ema_rate + 0.4 * excess
            ttc = 1.0 / state.ema_rate if state.ema_rate > 1e-3 else math.inf
            state.hits = state.hits + 1 if ttc < self.ttc_warning_s else 0
            if state.hits >= self.persist_frames:
                threats.append(ThreatItem(
                    bearing_deg=round(bearing_from_x(cx / w_img, self.hfov_deg), 1),
                    ttc_s=round(ttc, 2),
                    urgency="CRITICAL" if ttc < self.ttc_critical_s else "WARNING",
                    zone=zone,
                ))
                movers.extend(w[2] for w in windows if ZONES[min(2, int(3 * w[1] / w_img))] == zone
                              and w[0] - background > 0.5 / self.ttc_warning_s)
        self.debug = FlowDebug(
            movers=np.concatenate(movers) if movers else np.empty((0, 2), np.float32),
            ego_scale=math.sqrt(abs(np.linalg.det(M[:, :2]))), background_rate=background,
            zone_rates={z: self.zones[z].ema_rate for z in ZONES},
        )
        return threats

    def _window_rates(self, pred: np.ndarray, obs: np.ndarray, shape, dt: float):
        """Least-squares divergence of observed vs camera-predicted points per window.

        Returns [(rate_per_s, window_center_x, points), ...].
        """
        gx, gy = self.GRID
        h, w = shape
        bx, by = w / gx, h / gy
        out = []
        for j in range(gy - self.WINDOW + 1):
            for i in range(gx - self.WINDOW + 1):
                x0, y0 = i * bx, j * by
                sel = ((obs[:, 0] >= x0) & (obs[:, 0] < x0 + self.WINDOW * bx)
                       & (obs[:, 1] >= y0) & (obs[:, 1] < y0 + self.WINDOW * by))
                if sel.sum() < self.min_points:
                    continue
                p, o = pred[sel], obs[sel]
                dp, do = p - p.mean(0), o - o.mean(0)
                denom = float((dp * dp).sum())
                if denom < 25.0:                   # points too clustered to measure scale
                    continue
                scale = float((dp * do).sum()) / denom
                out.append(((scale - 1.0) / dt, x0 + self.WINDOW * bx / 2, o))
        return out

    def _decay_all(self) -> None:
        for z in self.zones.values():
            z.ema_rate *= 0.5
            z.hits = 0
