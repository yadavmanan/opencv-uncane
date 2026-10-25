"""Synthetic scenes rendered from the same camera model: floor + box / branch."""
import copy
import math

import numpy as np
import pytest

from uncane.edge.config import DEFAULTS
from uncane.edge.geometry import CameraGeometry
from uncane.edge.hazards import HazardPipeline

H_CAM, PITCH, HFOV, W, H = 1.3, 12.0, 70.0, 640, 480
S = 160


def render(obstacle=None):
    """Metric Z map: floor everywhere, optional vertical plate
    obstacle = (forward_m, x_min, x_max, h_min, h_max)."""
    g = CameraGeometry(HFOV, W, H, H_CAM, PITCH)
    tx, ty, z_floor = g.rays(S, S)
    Z = np.broadcast_to(np.minimum(z_floor, 30.0), (S, S)).copy()
    if obstacle:
        f, x0, x1, h0, h1 = obstacle
        p = math.radians(PITCH)
        z_plate = f / (math.cos(p) - ty * math.sin(p))
        z_plate = np.broadcast_to(z_plate, (S, S))
        X = z_plate * tx
        Hh = H_CAM - z_plate * (ty * math.cos(p) + math.sin(p))
        hit = (z_plate < Z) & (X >= x0) & (X <= x1) & (Hh >= h0) & (Hh <= h1)
        Z = np.where(hit, z_plate, Z)
    return Z.astype(np.float32)


def to_relative(Z, rng):
    r = 3.0 / Z + 0.5                        # unknown affine in inverse depth, like MiDaS
    return (r + rng.normal(0, 0.01, r.shape)).astype(np.float32)


def make_pipeline(output="relative_inverse"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["camera"].update(hfov_deg=HFOV, width=W, height=H)
    cfg["mount"].update(height_m=H_CAM, pitch_deg=PITCH)
    return HazardPipeline(cfg, output, W, H)


def test_floor_only_is_not_a_hazard():
    rng = np.random.default_rng(0)
    pipe = make_pipeline()
    for _ in range(6):
        hf = pipe.process(to_relative(render(), rng))
    assert pipe.calibrator.calibrated
    assert hf.hazards == []


def test_calibration_recovers_metric_scale():
    rng = np.random.default_rng(1)
    pipe = make_pipeline()
    Z = render()
    for _ in range(5):
        pipe.process(to_relative(Z, rng))
    Zm = pipe.calibrator.to_metric(to_relative(Z, rng))
    band = slice(int(0.75 * S), S)
    rel_err = np.abs(Zm[band] - Z[band]) / Z[band]
    assert np.median(rel_err) < 0.05


def test_box_ahead_is_critical_with_distance():
    rng = np.random.default_rng(2)
    pipe = make_pipeline()
    for _ in range(5):                      # walk in with a clear floor first
        pipe.process(to_relative(render(), rng))
    hf = pipe.process(to_relative(render((1.0, -0.25, 0.25, 0.0, 1.0)), rng))
    center = [h for h in hf.hazards if h.zone == "CENTER"]
    assert center, hf.hazards
    assert all(h.urgency == "CRITICAL" for h in center)
    assert min(h.distance_m for h in center) == pytest.approx(1.0, abs=0.2)
    assert {h.level for h in center} <= {"GROUND", "TORSO"}


def test_head_height_branch_detected_but_cane_level_clear():
    pipe = make_pipeline(output="metric")   # metric model path: no calibration needed
    hf = pipe.process(render((1.4, -0.5, 0.5, 1.45, 1.75)))
    levels = {(h.zone, h.level): h for h in hf.hazards}
    assert ("CENTER", "HEAD") in levels
    assert levels[("CENTER", "HEAD")].urgency == "CRITICAL"   # critical_head_m = 1.6
    assert ("CENTER", "GROUND") not in levels


def test_side_obstacle_is_warning_not_critical():
    pipe = make_pipeline(output="metric")
    hf = pipe.process(render((1.0, 0.5, 1.2, 0.0, 1.5)))
    right = [h for h in hf.hazards if h.zone == "RIGHT"]
    assert right and all(h.urgency == "WARNING" for h in right)
    assert not [h for h in hf.hazards if h.zone == "CENTER"]


def test_depth_grid_is_metric():
    pipe = make_pipeline(output="metric")
    hf = pipe.process(render())
    assert hf.depth_grid.kind == "metric_m"
    assert len(hf.depth_grid.values) == hf.depth_grid.rows * hf.depth_grid.cols
