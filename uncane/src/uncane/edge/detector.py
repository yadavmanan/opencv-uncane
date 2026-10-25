"""COCO object detector (YOLOX from the OpenCV Model Zoo) for naming hazards.

The depth pipeline says *where* something is; the detector says *what* it is
("chair ahead", "bicycle from the right"). It runs in its own thread at a few
Hz so it never slows the reflex loop.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..common.schemas import HazardItem, ThreatItem
from ..common.spatial import bearing_from_x, zone_from_bearing

log = logging.getLogger(__name__)

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)
# Classes that matter while walking (others are still returned, but ranked lower).
NAV_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "dog", "chair", "couch", "potted plant", "dining table",
    "suitcase", "skateboard", "bed", "refrigerator", "tv",
}
MOVING_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "skateboard"}


@dataclass
class Detection:
    label: str
    score: float
    box: Sequence[float]      # normalized x0, y0, x1, y1

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


class YoloXDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.45, nms_threshold: float = 0.5,
                 input_size: int = 640):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf, self.nms, self.size = conf_threshold, nms_threshold, input_size
        grids, strides = [], []
        for s in (8, 16, 32):
            n = input_size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            g = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(g)
            strides.append(np.full((g.shape[0], 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(h * r), int(w * r)
        pad = np.full((self.size, self.size, 3), 114.0, np.float32)
        pad[:nh, :nw] = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (nw, nh))
        self.net.setInput(np.ascontiguousarray(pad.transpose(2, 0, 1)[None]))
        out = self.net.forward()[0].copy()
        out[:, :2] = (out[:, :2] + self.grids) * self.strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.strides
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        if not keep.any():
            return []
        boxes, conf, cls = out[keep, :4], conf[keep], cls[keep]
        xywh = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]], 1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(), self.conf, self.nms)
        dets = []
        for i in np.array(idx).reshape(-1):
            x, y, bw, bh = xywh[i] / r
            dets.append(Detection(
                label=COCO_CLASSES[int(cls[i])], score=float(conf[i]),
                box=[float(max(0.0, x / w)), float(max(0.0, y / h)),
                     float(min(1.0, (x + bw) / w)), float(min(1.0, (y + bh) / h))],
            ))
        return dets


class DetectorThread:
    """Runs the detector on the latest frame at a fixed rate."""

    def __init__(self, detector: YoloXDetector, hz: float = 4.0):
        self.detector, self.period = detector, 1.0 / max(hz, 0.1)
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.latest: List[Detection] = []
        self.latest_t = 0.0
        self.last_ms = 0.0
        self._thread = threading.Thread(target=self._run, name="detector", daemon=True)

    def start(self) -> "DetectorThread":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is not None:
                try:
                    self.latest = self.detector.detect(frame)
                    self.latest_t = time.monotonic()
                    self.last_ms = (self.latest_t - t0) * 1000
                except cv2.error as exc:
                    log.warning("detector error: %s", exc)
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))


def label_hazards(hazards: List[HazardItem], dets: Sequence[Detection], hfov_deg: float) -> None:
    """Attach the most plausible class name to each hazard, in place."""
    for h in hazards:
        if h.label == "drop-off":
            continue
        cands = [d for d in dets if d.label in NAV_CLASSES
                 and zone_from_bearing(bearing_from_x(d.cx, hfov_deg)) == h.zone]
        if h.level == "HEAD":
            cands = [d for d in cands if d.box[1] < 0.5]
        if cands:
            h.label = max(cands, key=lambda d: d.area * d.score).label


def label_threats(threats: List[ThreatItem], dets: Sequence[Detection], hfov_deg: float) -> None:
    for t in threats:
        cands = [d for d in dets if d.label in MOVING_CLASSES
                 and abs(bearing_from_x(d.cx, hfov_deg) - t.bearing_deg) < 20]
        if cands:
            t.label = max(cands, key=lambda d: d.area).label
