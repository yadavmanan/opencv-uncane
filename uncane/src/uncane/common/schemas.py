"""JSON contracts shared by the edge device and the AWS cloud tier.

Plain dataclasses (no third-party deps) so the exact same module ships inside
the Lambda package and runs on the edge device.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

ZONES = ("LEFT", "CENTER", "RIGHT")
LEVELS = ("GROUND", "TORSO", "HEAD")
URGENCIES = ("INFO", "WARNING", "CRITICAL")
DIRECTIONS = ("left", "slightly_left", "ahead", "slightly_right", "right", "behind", "stop", "none")

MAX_IMAGE_B64_CHARS = 4_000_000  # ~3 MB JPEG; API Gateway caps the body at 10 MB
MAX_QUERY_CHARS = 500


class ValidationError(ValueError):
    """Raised when a payload does not match the contract."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)


def _urgency_rank(u: str) -> int:
    return URGENCIES.index(u) if u in URGENCIES else -1


@dataclass
class HazardItem:
    """A static obstacle found by the edge depth pipeline."""

    zone: str                              # LEFT | CENTER | RIGHT (3D walking corridor)
    level: str                             # GROUND | TORSO | HEAD (height above floor)
    proximity: float                       # 0..1, 1 = touching
    urgency: str                           # INFO | WARNING | CRITICAL
    distance_m: Optional[float] = None     # metric estimate when calibrated
    label: Optional[str] = None            # e.g. "chair" when the detector agrees

    def validate(self) -> "HazardItem":
        _require(self.zone in ZONES, f"bad zone {self.zone!r}")
        _require(self.level in LEVELS, f"bad level {self.level!r}")
        _require(self.urgency in URGENCIES, f"bad urgency {self.urgency!r}")
        _require(0.0 <= float(self.proximity) <= 1.0, "proximity must be in [0, 1]")
        if self.distance_m is not None:
            _require(0.0 <= float(self.distance_m) <= 100.0, "distance_m out of range")
        return self

    def describe(self) -> str:
        where = {"LEFT": "on your left", "CENTER": "directly ahead", "RIGHT": "on your right"}[self.zone]
        height = {"GROUND": "at ground level", "TORSO": "at body height", "HEAD": "at head height"}[self.level]
        what = self.label or "obstacle"
        dist = f", about {self.distance_m:.1f} meters" if self.distance_m is not None else ""
        return f"{what} {where} {height}{dist}"


@dataclass
class ThreatItem:
    """A moving object on a collision course, from the optical-flow tracker."""

    bearing_deg: float                     # negative = left, positive = right
    ttc_s: float                           # estimated time to contact
    urgency: str
    zone: str = "CENTER"
    label: Optional[str] = None

    def validate(self) -> "ThreatItem":
        _require(-90.0 <= float(self.bearing_deg) <= 90.0, "bearing_deg out of range")
        _require(float(self.ttc_s) >= 0.0, "ttc_s must be >= 0")
        _require(self.urgency in URGENCIES, f"bad urgency {self.urgency!r}")
        _require(self.zone in ZONES, f"bad zone {self.zone!r}")
        return self


@dataclass
class DepthGrid:
    """Coarse depth summary sent with a keyframe so the cloud can ground distances.

    ``values`` is row-major, ``rows * cols`` long. With kind == "metric_m" each
    cell is the median depth in meters (0 = unknown). With kind == "proximity"
    each cell is a unitless 0..1 closeness score.
    """

    rows: int
    cols: int
    kind: str
    values: List[float]

    def validate(self) -> "DepthGrid":
        _require(self.kind in ("metric_m", "proximity"), "bad depth grid kind")
        _require(1 <= self.rows <= 64 and 1 <= self.cols <= 64, "depth grid too large")
        _require(len(self.values) == self.rows * self.cols, "depth grid size mismatch")
        return self

    def sample(self, x0: float, y0: float, x1: float, y1: float) -> Optional[float]:
        """Median value of cells overlapping a normalized [0,1] box (None if unknown)."""
        c0, c1 = int(max(0, x0) * self.cols), int(min(0.999, x1) * self.cols)
        r0, r1 = int(max(0, y0) * self.rows), int(min(0.999, y1) * self.rows)
        vals = sorted(
            v for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if (v := self.values[r * self.cols + c]) > 0
        )
        return vals[len(vals) // 2] if vals else None


@dataclass
class CameraInfo:
    width: int = 640
    height: int = 480
    hfov_deg: float = 70.0

    def validate(self) -> "CameraInfo":
        _require(10.0 <= float(self.hfov_deg) <= 180.0, "hfov_deg out of range")
        _require(self.width > 0 and self.height > 0, "bad camera size")
        return self


@dataclass
class NavigateRequest:
    """Edge -> cloud. One user question (or ambient keyframe) plus edge context."""

    device_id: str
    session_id: str
    timestamp: float
    query: str
    image_jpeg_b64: str
    camera: CameraInfo = field(default_factory=CameraInfo)
    hazards: List[HazardItem] = field(default_factory=list)
    threats: List[ThreatItem] = field(default_factory=list)
    depth_grid: Optional[DepthGrid] = None
    mode: str = "query"                    # query | ambient
    locale: str = "en-US"
    location: Optional[Dict[str, float]] = None   # {"lat": .., "lon": ..} if the phone shares it

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NavigateRequest":
        _require(isinstance(d, dict), "body must be a JSON object")
        try:
            req = cls(
                device_id=str(d["device_id"])[:128],
                session_id=str(d.get("session_id") or d["device_id"])[:128],
                timestamp=float(d.get("timestamp", 0.0)),
                query=str(d.get("query") or "")[:MAX_QUERY_CHARS],
                image_jpeg_b64=str(d.get("image_jpeg_b64") or ""),
                camera=CameraInfo(**(d.get("camera") or {})),
                hazards=[HazardItem(**h) for h in d.get("hazards") or []],
                threats=[ThreatItem(**t) for t in d.get("threats") or []],
                depth_grid=DepthGrid(**d["depth_grid"]) if d.get("depth_grid") else None,
                mode=str(d.get("mode") or "query"),
                locale=str(d.get("locale") or "en-US"),
                location=d.get("location"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f"malformed request: {exc}") from exc
        return req.validate()

    def validate(self) -> "NavigateRequest":
        _require(bool(self.device_id), "device_id required")
        _require(self.mode in ("query", "ambient"), "mode must be query|ambient")
        _require(self.mode == "ambient" or bool(self.query.strip()), "query required")
        _require(0 < len(self.image_jpeg_b64) <= MAX_IMAGE_B64_CHARS, "image_jpeg_b64 missing or too large")
        _require(len(self.hazards) <= 32 and len(self.threats) <= 16, "too many hazards/threats")
        self.camera.validate()
        for h in self.hazards:
            h.validate()
        for t in self.threats:
            t.validate()
        if self.depth_grid:
            self.depth_grid.validate()
        if self.location is not None:
            _require(
                isinstance(self.location, dict) and {"lat", "lon"} <= set(self.location),
                "location must be {lat, lon}",
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def worst_urgency(self) -> str:
        items = [h.urgency for h in self.hazards] + [t.urgency for t in self.threats]
        return max(items, key=_urgency_rank) if items else "INFO"


@dataclass
class TargetInfo:
    """Where the thing the user asked about is, in egocentric terms."""

    label: str
    bearing_deg: Optional[float] = None
    clock: Optional[int] = None            # 12 = straight ahead, 3 = hard right
    distance_m: Optional[float] = None
    steps: Optional[int] = None
    bbox: Optional[List[float]] = None     # normalized [x0, y0, x1, y1]


@dataclass
class NavigateResponse:
    """Cloud -> edge. ``speech`` is already verified and safe to speak."""

    speech: str
    raw_model_speech: str = ""
    target: Optional[TargetInfo] = None
    safety_override: bool = False
    override_reasons: List[str] = field(default_factory=list)
    needs_better_view: bool = False
    tools_used: List[str] = field(default_factory=list)
    model_id: str = ""
    latency_ms: Dict[str, float] = field(default_factory=dict)
    request_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NavigateResponse":
        target = d.get("target")
        return cls(
            speech=str(d.get("speech", "")),
            raw_model_speech=str(d.get("raw_model_speech", "")),
            target=TargetInfo(**target) if target else None,
            safety_override=bool(d.get("safety_override", False)),
            override_reasons=list(d.get("override_reasons") or []),
            needs_better_view=bool(d.get("needs_better_view", False)),
            tools_used=list(d.get("tools_used") or []),
            model_id=str(d.get("model_id", "")),
            latency_ms=dict(d.get("latency_ms") or {}),
            request_id=str(d.get("request_id", "")),
        )


@dataclass
class ModelAnswer:
    """Structured answer the VLM returns through its ``final_answer`` tool."""

    speech: str
    target_label: Optional[str] = None
    target_bbox: Optional[List[float]] = None       # 0..1000 normalized [x0, y0, x1, y1]
    suggested_direction: str = "none"
    claims_path_clear: bool = False
    needs_better_view: bool = False
    confidence: float = 0.5

    @classmethod
    def from_tool_input(cls, d: Dict[str, Any]) -> "ModelAnswer":
        bbox = d.get("target_bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            bbox = None
        direction = str(d.get("suggested_direction") or "none")
        return cls(
            speech=str(d.get("speech") or "").strip(),
            target_label=(str(d["target_label"]) if d.get("target_label") else None),
            target_bbox=[float(v) for v in bbox] if bbox else None,
            suggested_direction=direction if direction in DIRECTIONS else "none",
            claims_path_clear=bool(d.get("claims_path_clear", False)),
            needs_better_view=bool(d.get("needs_better_view", False)),
            confidence=max(0.0, min(1.0, float(d.get("confidence", 0.5) or 0.5))),
        )
