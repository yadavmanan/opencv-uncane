"""Turn a depth map into hazards inside a 3D walking corridor.

Zones are lateral bands in meters (not image thirds), levels are heights above
the floor. This is what lets the system warn about a branch at head height
that a white cane sweeps right under, while ignoring the floor itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..common.schemas import DepthGrid, HazardItem
from .geometry import CameraGeometry, GroundPlaneCalibrator


@dataclass
class HazardFrame:
    hazards: List[HazardItem]
    depth_grid: Optional[DepthGrid]
    metric_depth: Optional[np.ndarray] = None
    calibrated: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)


class GeometricHazardAnalyzer:
    def __init__(self, geometry: CameraGeometry, cfg: Dict[str, Any], user_height_m: float):
        self.geometry = geometry
        self.cfg = cfg
        self.user_height_m = user_height_m

    def analyze(self, Z: np.ndarray) -> HazardFrame:
        c = self.cfg
        pts = self.geometry.backproject(Z)
        fwd, X, H = pts.forward, pts.X, pts.height
        near = (fwd > 0.15) & (fwd < c["max_range_m"])
        solid = near & (H > c["floor_tolerance_m"]) & (H < self.user_height_m + 0.10)
        min_px = max(12, int(c["min_pixel_fraction"] * Z.size))

        half, side = c["corridor_half_width_m"], c["side_width_m"]
        zones = {
            "LEFT": (X < -half) & (X > -side),
            "CENTER": np.abs(X) <= half,
            "RIGHT": (X > half) & (X < side),
        }
        levels = {
            "GROUND": H < 0.5,
            "TORSO": (H >= 0.5) & (H < 1.3),
            "HEAD": H >= 1.3,
        }
        hazards: List[HazardItem] = []
        for zone, zmask in zones.items():
            for level, lmask in levels.items():
                m = solid & zmask & lmask
                n = int(m.sum())
                if n < min_px:
                    continue
                dist = float(np.percentile(fwd[m], 10))
                urgency = self._urgency(zone, level, dist)
                if urgency is None:
                    continue
                prox = float(np.clip(1.0 - dist / c["max_range_m"], 0.0, 1.0))
                hazards.append(HazardItem(zone=zone, level=level, proximity=round(prox, 3),
                                          urgency=urgency, distance_m=round(dist, 2)))
        dropoff_mask = np.zeros_like(near)
        if c.get("dropoff_enabled", True):
            # Depth error alone puts floor up to ~H_cam * error below the plane, so require a deep
            # region that covers much of what is in range, not a few noisy pixels.
            in_range = (fwd > 0.15) & (fwd < c.get("dropoff_range_m", 4.5))
            dropoff_mask = in_range & (H < -c.get("dropoff_min_depth_m", 0.40))
            for zone, zmask in zones.items():
                m = dropoff_mask & zmask
                share = float(m.sum()) / max(1, int((in_range & zmask).sum()))
                if int(m.sum()) < min_px or share < c.get("dropoff_min_share", 0.25):
                    continue
                # We see the lowered floor beyond the edge, so rescale back to where the edge is.
                below = float(-np.median(H[m]))
                cam_h = self.geometry.height_m
                dist = float(np.percentile(fwd[m], 10)) * cam_h / (cam_h + below)
                if dist < c.get("dropoff_critical_m", 1.7):
                    urgency = "CRITICAL"
                elif dist < c.get("dropoff_warning_m", 2.6):
                    urgency = "WARNING"
                else:
                    continue
                prox = float(np.clip(1.0 - dist / c["max_range_m"], 0.0, 1.0))
                hazards.append(HazardItem(zone=zone, level="GROUND", proximity=round(prox, 3),
                                          urgency=urgency, distance_m=round(dist, 2), label="drop-off"))
        if c.get("merge_adjacent_levels", True):
            hazards = self._merge_adjacent_levels(hazards, c.get("level_merge_distance_m", 0.5))
        grid = cv2.resize(np.minimum(Z, 20.0), (c["grid_cols"], c["grid_rows"]), interpolation=cv2.INTER_AREA)
        depth_grid = DepthGrid(rows=c["grid_rows"], cols=c["grid_cols"], kind="metric_m",
                               values=[round(float(v), 2) for v in grid.ravel()])
        return HazardFrame(hazards, depth_grid, metric_depth=Z, calibrated=True,
                           debug={"solid_mask": solid, "dropoff_mask": dropoff_mask})

    @staticmethod
    def _merge_adjacent_levels(hazards: List[HazardItem], max_distance_delta_m: float) -> List[HazardItem]:
        level_order = {"GROUND": 0, "TORSO": 1, "HEAD": 2}
        urgency_rank = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
        merged: List[HazardItem] = []
        for zone in ("LEFT", "CENTER", "RIGHT"):
            zone_hazards = sorted(
                (hazard for hazard in hazards if hazard.zone == zone),
                key=lambda hazard: (hazard.label == "drop-off", level_order[hazard.level]),
            )
            groups: List[List[HazardItem]] = []
            for hazard in zone_hazards:
                if groups:
                    previous = groups[-1][-1]
                    same_kind = (hazard.label == "drop-off") == (previous.label == "drop-off")
                    adjacent = same_kind and level_order[hazard.level] - level_order[previous.level] == 1
                    nearby = (hazard.distance_m is None or previous.distance_m is None or
                              abs(hazard.distance_m - previous.distance_m) <= max_distance_delta_m)
                    if adjacent and nearby:
                        groups[-1].append(hazard)
                        continue
                groups.append([hazard])
            for group in groups:
                nearest = min((h.distance_m for h in group if h.distance_m is not None), default=None)
                representative = min(
                    group,
                    key=lambda h: (-urgency_rank[h.urgency],
                                   h.distance_m if h.distance_m is not None else float("inf")),
                )
                merged.append(HazardItem(
                    zone=zone,
                    level=representative.level,
                    proximity=max(h.proximity for h in group),
                    urgency=max(group, key=lambda h: urgency_rank[h.urgency]).urgency,
                    distance_m=nearest,
                    label=representative.label,
                ))
        return merged

    def _urgency(self, zone: str, level: str, dist: float) -> Optional[str]:
        c = self.cfg
        if zone == "CENTER":
            crit = c["critical_head_m"] if level == "HEAD" else c["critical_m"]
            if dist < crit:
                return "CRITICAL"
            if dist < c["warning_m"]:
                return "WARNING"
            return None
        return "WARNING" if dist < c["side_warning_m"] else None


class SectorHazardAnalyzer:
    """Legacy image-thirds analysis on relative depth (used before calibration).

    Improves on a plain mean: uses a per-frame robust scale and the share of
    "near" pixels, and skips the bottom rows that are usually floor.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._scale: Optional[float] = None

    def analyze(self, r: np.ndarray) -> HazardFrame:
        h, w = r.shape
        p98 = float(np.percentile(r, 98))
        self._scale = p98 if self._scale is None else 0.8 * self._scale + 0.2 * p98
        norm = np.clip(r / max(self._scale, 1e-6), 0.0, 1.5)
        bands = {"HEAD": (0.0, 0.3), "TORSO": (0.3, 0.7)}
        cols = {"LEFT": (0.0, 0.33), "CENTER": (0.33, 0.67), "RIGHT": (0.67, 1.0)}
        hazards: List[HazardItem] = []
        for zone, (c0, c1) in cols.items():
            for level, (b0, b1) in bands.items():
                region = norm[int(b0 * h):int(b1 * h), int(c0 * w):int(c1 * w)]
                near_share = float((region > 0.85).mean())
                if near_share < 0.25:
                    continue
                prox = float(np.clip(np.percentile(region, 90), 0.0, 1.0))
                urgency = "CRITICAL" if zone == "CENTER" and near_share > 0.5 else "WARNING"
                hazards.append(HazardItem(zone=zone, level=level, proximity=round(prox, 3), urgency=urgency))
        grid = cv2.resize(np.clip(norm, 0, 1), (self.cfg["grid_cols"], self.cfg["grid_rows"]),
                          interpolation=cv2.INTER_AREA)
        depth_grid = DepthGrid(rows=self.cfg["grid_rows"], cols=self.cfg["grid_cols"], kind="proximity",
                               values=[round(float(v), 3) for v in grid.ravel()])
        return HazardFrame(hazards, depth_grid, calibrated=False)


class HazardPipeline:
    """Chooses geometric (metric) or sector analysis depending on calibration."""

    def __init__(self, cfg: Dict[str, Any], depth_output: str, frame_w: int, frame_h: int):
        mount, cam = cfg["mount"], cfg["camera"]
        self.geometry = CameraGeometry(cam["hfov_deg"], frame_w, frame_h, mount["height_m"], mount["pitch_deg"])
        self.calibrator = GroundPlaneCalibrator(self.geometry)
        self.metric_model = depth_output == "metric"
        self.mode = cfg["hazards"]["analyzer"]
        self.geometric = GeometricHazardAnalyzer(self.geometry, cfg["hazards"], mount["user_height_m"])
        self.sector = SectorHazardAnalyzer(cfg["hazards"])

    def process(self, depth: np.ndarray) -> HazardFrame:
        if self.mode == "sector":
            return self.sector.analyze(depth)
        if self.metric_model:
            return self.geometric.analyze(depth)
        self.calibrator.update(depth)
        if not self.calibrator.calibrated:
            return self.sector.analyze(depth)
        Z = self.calibrator.to_metric(depth)
        frame = self.geometric.analyze(Z)
        frame.debug["calib_ok"] = self.calibrator.last_fit_ok
        return frame
