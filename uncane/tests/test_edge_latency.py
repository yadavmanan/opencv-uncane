"""Reflex-loop latency of the real edge pipeline (opt-in: pytest -m latency -s).

Budget comes from UNCANE_LATENCY_BUDGET_MS (default 50, the brief's target).
Uses a real photo if tests/data/sample_indoor.jpg exists, else random frames (worst case).
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from uncane.edge.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.latency
def test_reflex_loop_latency():
    cfg = load_config(str(ROOT / "config" / "edge_config.yaml"), [
        "detector.enabled=false", "cloud.enabled=false", "audio.enabled=false", "voice.backend=none"])
    if not Path(cfg["depth"]["model_path"]).exists():
        pytest.skip("depth model not downloaded (python scripts/download_models.py)")
    from uncane.edge.app import EdgeApp

    app = EdgeApp(cfg, headless=True)
    sample = ROOT / "tests" / "data" / "sample_indoor.jpg"
    if sample.exists():
        base = cv2.resize(cv2.imread(str(sample)), (640, 480))
        frames = [np.roll(base, 3 * i, axis=1) for i in range(8)]
    else:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(8)]
    for i in range(60):
        app.process_frame(frames[i % len(frames)], i / 30)
    app.reflex_ms = app.reflex_ms[10:]      # drop warm-up
    p50, p95 = app.latency_stats()
    budget = float(os.environ.get("UNCANE_LATENCY_BUDGET_MS", 50))
    print(f"reflex p50 = {p50:.1f} ms, p95 = {p95:.1f} ms (budget {budget} ms, "
          f"depth input {cfg['depth']['input_size']}px, OpenCV {cv2.__version__})")
    assert p50 < budget
