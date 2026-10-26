"""OpenCV tools the cloud agent can call on the photo it was sent.

Each call runs real OpenCV analysis and returns evidence the model could not have guessed (sharpness,
where text is, a zoomed crop, a colour reading, metric distance), so its next step depends on it.
If OpenCV is not in the Lambda package the tools are simply not offered (``AVAILABLE`` is False).
"""
from __future__ import annotations

import base64
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from ..common.schemas import NavigateRequest
from ..common.spatial import locate_target

log = logging.getLogger(__name__)

AVAILABLE = cv2 is not None and int(cv2.__version__.split(".")[0]) >= 5   # the competition requires OpenCV 5
OPENCV_VERSION = cv2.__version__ if cv2 is not None else None

BBOX_SCHEMA = {
    "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
    "description": "[x_min, y_min, x_max, y_max], each 0-1000 relative to the photo width/height.",
}

VISION_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "assess_image_quality",
        "description": (
            "OpenCV check of the photo: sharpness, brightness, contrast and glare. Call it before saying you "
            "cannot read or see something, or when the photo looks blurry or dark. If `usable` is false, do not "
            "guess: set needs_better_view and tell the person how to re-aim."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_text_regions",
        "description": (
            "OpenCV search for text-like regions (signs, labels, door numbers). Returns boxes ranked by "
            "likelihood. Follow with zoom_region on the best box to read it."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "zoom_region",
        "description": (
            "Crop a box from the original photo, upscale it and sharpen it with OpenCV, and return the zoomed "
            "image. Use it to read small or distant text or to inspect a candidate object. Look at the returned "
            "image before answering."),
        "input_schema": {
            "type": "object",
            "properties": {"bbox": BBOX_SCHEMA,
                           "enhance": {"type": "boolean", "description": "Contrast and sharpness boost. Default true."}},
            "required": ["bbox"],
        },
    },
    {
        "name": "analyze_region_color",
        "description": (
            "Measure the dominant colours inside a box with OpenCV. Use it to verify a colour or object you are "
            "about to report (for example a 'red bottle') before giving its box. If it disagrees, pick another "
            "candidate or say you are unsure."),
        "input_schema": {
            "type": "object",
            "properties": {"bbox": BBOX_SCHEMA,
                           "expected_color": {"type": "string", "description": "Colour the person asked for, if any."}},
            "required": ["bbox"],
        },
    },
    {
        "name": "measure_region_depth",
        "description": (
            "Metric distance, bearing and step count to a box, from the device's depth sensor, plus whether a "
            "critical obstacle is ahead. Use it instead of estimating distances yourself."),
        "input_schema": {"type": "object", "properties": {"bbox": BBOX_SCHEMA}, "required": ["bbox"]},
    },
]
VISION_TOOL_NAMES = {t["name"] for t in VISION_TOOLS}

# Laplacian variance at 640 px wide; below this text edges are too soft to read.
SHARPNESS_MIN = 40.0
_COLOR_HUES = (("red", 0, 8), ("orange", 8, 20), ("yellow", 20, 35), ("green", 35, 85), ("cyan", 85, 100),
               ("blue", 100, 130), ("purple", 130, 160), ("pink", 160, 172), ("red", 172, 180))


class VisionToolError(Exception):
    pass


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    image_b64: Optional[str] = None


def _json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _error(message: str) -> ToolResult:
    return ToolResult(_json({"error": message}), True)


class VisionToolbox:
    """Per-request tool runner. Decodes the photo once, caps the number of calls, records a trace."""

    MAX_CALLS = 4

    def __init__(self, req: NavigateRequest):
        self.req = req
        self.trace: List[Dict[str, Any]] = []
        self._img = None
        self._calls = 0
        self._tools: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
            "assess_image_quality": self._assess_image_quality,
            "find_text_regions": self._find_text_regions,
            "zoom_region": self._zoom_region,
            "analyze_region_color": self._analyze_region_color,
            "measure_region_depth": self._measure_region_depth,
        }

    def record(self, name: str, args: Dict[str, Any], result: ToolResult, started: float) -> None:
        self.trace.append({
            "step": len(self.trace) + 1, "tool": name, "args": args, "ok": not result.is_error,
            "opencv": OPENCV_VERSION,
            "returned_image": result.image_b64 is not None, "result": result.content[:300],
            "ms": round((time.perf_counter() - started) * 1000, 1),
        })

    def call(self, name: str, args: Dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        if name not in self._tools:
            result = _error(f"unknown tool {name}")
        elif not AVAILABLE:
            result = _error("OpenCV 5 is not available in this deployment.")
        elif self._calls >= self.MAX_CALLS:
            result = _error("Vision tool budget used up. Call final_answer now.")
        else:
            self._calls += 1
            try:
                result = self._tools[name](args or {})
            except VisionToolError as exc:
                result = _error(str(exc))
            except Exception as exc:
                log.exception("vision tool %s failed", name)
                result = _error(f"{name} failed: {type(exc).__name__}")
        self.record(name, args or {}, result, started)
        return result

    # ------------------------------------------------------------------ helpers
    @property
    def image(self):
        if self._img is None:
            raw = base64.b64decode(self.req.image_jpeg_b64)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise VisionToolError("The photo could not be decoded.")
            self._img = img
        return self._img

    @staticmethod
    def _box(args: Dict[str, Any]) -> Tuple[float, float, float, float]:
        raw = args.get("bbox")
        if not (isinstance(raw, (list, tuple)) and len(raw) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)):
            raise VisionToolError("bbox must be [x_min, y_min, x_max, y_max], each 0-1000.")
        x0, y0, x1, y1 = [min(1000.0, max(0.0, float(v))) / 1000.0 for v in raw]
        if x1 - x0 < 0.02 or y1 - y0 < 0.02:
            raise VisionToolError("bbox is inverted or too small.")
        return x0, y0, x1, y1

    def _crop(self, box: Tuple[float, float, float, float], pad: float = 0.0):
        h, w = self.image.shape[:2]
        x0, y0, x1, y1 = box
        dx, dy = pad * (x1 - x0), pad * (y1 - y0)
        px0, px1 = int(max(0.0, x0 - dx) * w), int(math.ceil(min(1.0, x1 + dx) * w))
        py0, py1 = int(max(0.0, y0 - dy) * h), int(math.ceil(min(1.0, y1 + dy) * h))
        crop = self.image[py0:py1, px0:px1]
        if crop.size == 0:
            raise VisionToolError("bbox falls outside the photo.")
        return crop

    @staticmethod
    def _sharpness(gray) -> float:
        h, w = gray.shape[:2]
        if w > 640:
            gray = cv2.resize(gray, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # -------------------------------------------------------------------- tools
    def _assess_image_quality(self, _args: Dict[str, Any]) -> ToolResult:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        brightness, contrast = float(gray.mean()), float(gray.std())
        glare = float((gray >= 250).mean())
        sharpness = self._sharpness(gray)
        if brightness < 35:
            verdict = "too_dark"
        elif brightness > 225 or glare > 0.25:
            verdict = "overexposed"
        elif contrast < 18:
            verdict = "low_contrast"
        elif sharpness < SHARPNESS_MIN:
            verdict = "blurry"
        else:
            verdict = "ok"
        return ToolResult(_json({
            "verdict": verdict, "usable": verdict == "ok", "sharpness": round(sharpness, 1),
            "sharpness_min": SHARPNESS_MIN, "brightness": round(brightness, 1),
            "contrast": round(contrast, 1), "glare_fraction": round(glare, 3)}))

    def _find_text_regions(self, _args: Dict[str, Any]) -> ToolResult:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > 640:
            gray = cv2.resize(gray, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA)
        sh, sw = gray.shape
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if int(grad.max()) < 30:
            return ToolResult(_json({"regions": [], "note": "No text-like structure. The photo may be blurry or empty."}))
        _, edges = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, sw // 14), 5)))
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area_frac = (bw * bh) / float(sw * sh)
            if bw < 0.05 * sw or bh < 0.015 * sh or bw / bh < 1.2 or area_frac > 0.6:
                continue
            density = float(cv2.countNonZero(edges[y:y + bh, x:x + bw])) / (bw * bh)
            if density < 0.04:
                continue
            found.append((density * math.sqrt(area_frac), x, y, bw, bh))
        found.sort(reverse=True)
        regions = [{"bbox": [int(1000 * x / sw), int(1000 * y / sh),
                             int(1000 * (x + bw) / sw), int(1000 * (y + bh) / sh)], "score": round(score, 3)}
                   for score, x, y, bw, bh in found[:5]]
        return ToolResult(_json({"regions": regions}))

    def _zoom_region(self, args: Dict[str, Any]) -> ToolResult:
        box = self._box(args)
        crop = self._crop(box, pad=0.04)
        ch, cw = crop.shape[:2]
        scale = min(6.0, 1024.0 / max(ch, cw))
        out = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else crop
        if args.get("enhance", True):
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            out = cv2.addWeighted(out, 1.5, cv2.GaussianBlur(out, (0, 0), 1.5), -0.5, 0)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise VisionToolError("Could not encode the zoomed image.")
        info = {"bbox": [round(v * 1000) for v in box], "crop_px": [cw, ch], "output_px": [out.shape[1], out.shape[0]],
                "scale": round(scale, 2), "note": "The zoomed image follows. Look at it before answering."}
        return ToolResult(_json(info), False, base64.b64encode(buf.tobytes()).decode("ascii"))

    def _analyze_region_color(self, args: Dict[str, Any]) -> ToolResult:
        crop = self._crop(self._box(args))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0].astype(np.int32), hsv[:, :, 1], hsv[:, :, 2]
        black = val < 50
        white = (sat < 40) & (val >= 180) & ~black
        gray = (sat < 40) & (val >= 50) & (val < 180)
        colored = ~(black | white | gray)
        total = float(hue.size)
        fractions = {"black": float(black.sum()) / total, "white": float(white.sum()) / total,
                     "gray": float(gray.sum()) / total}
        for name, lo, hi in _COLOR_HUES:
            share = float((colored & (hue >= lo) & (hue < hi)).sum()) / total
            fractions[name] = fractions.get(name, 0.0) + share
        ranked = sorted(fractions.items(), key=lambda kv: -kv[1])
        result: Dict[str, Any] = {
            "dominant_color": ranked[0][0],
            "fractions": {k: round(v, 3) for k, v in ranked if v >= 0.02}}
        expected = str(args.get("expected_color") or "").strip().lower()
        if expected:
            result["expected_color"] = expected
            result["matches_expected"] = fractions.get(expected, 0.0) >= 0.25
        return ToolResult(_json(result))

    def _measure_region_depth(self, args: Dict[str, Any]) -> ToolResult:
        grid = self.req.depth_grid
        if grid is None or grid.kind != "metric_m":
            raise VisionToolError("No metric depth was sent with this photo.")
        x0, y0, x1, y1 = self._box(args)
        target = locate_target("region", [x0 * 1000, y0 * 1000, x1 * 1000, y1 * 1000],
                               self.req.camera.hfov_deg, grid)
        if target is None or target.distance_m is None:
            raise VisionToolError("The depth sensor has no reading under that box.")
        critical = [h for h in self.req.hazards if h.urgency == "CRITICAL" and h.zone == "CENTER"]
        nearest = min((h.distance_m for h in critical if h.distance_m is not None), default=None)
        return ToolResult(_json({
            "distance_m": target.distance_m, "steps": target.steps, "bearing_deg": target.bearing_deg,
            "clock": target.clock, "critical_obstacle_ahead": bool(critical),
            "nearest_critical_obstacle_m": nearest}))
