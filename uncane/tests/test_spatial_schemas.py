import pytest

from uncane.common.schemas import DepthGrid, ModelAnswer, NavigateRequest, ValidationError
from uncane.common.spatial import bearing_from_x, clock_from_bearing, guidance_sentence, locate_target


def test_bearing_geometry():
    assert bearing_from_x(0.5, 70) == pytest.approx(0.0)
    assert bearing_from_x(1.0, 70) == pytest.approx(35.0)
    assert bearing_from_x(0.0, 70) == pytest.approx(-35.0)


def test_clock_positions():
    assert clock_from_bearing(0) == 12
    assert clock_from_bearing(30) == 1
    assert clock_from_bearing(-30) == 11
    assert clock_from_bearing(90) == 3


def test_locate_target_demo_case():
    # Bottle right of centre, 2.1 m away on the edge depth grid -> "Turn 30 degrees right, walk three steps."
    grid = DepthGrid(rows=2, cols=4, kind="metric_m", values=[3, 3, 3, 3, 3, 3, 3, 2.1])
    t = locate_target("red water bottle", [820, 700, 950, 950], hfov_deg=70, depth_grid=grid)
    assert t.bearing_deg == pytest.approx(28.3, abs=0.5)
    assert t.steps == 3
    assert guidance_sentence(t) == "Turn 30 degrees right, walk three steps."


def test_locate_target_without_metric_depth():
    t = locate_target("door", [0, 0, 200, 1000], 70, DepthGrid(1, 1, "proximity", [0.5]))
    assert t.distance_m is None and t.steps is None
    assert guidance_sentence(t).startswith("Turn")


def _req(**kw):
    d = {"device_id": "d1", "query": "where is the door?", "image_jpeg_b64": "/9j/AAAA"}
    d.update(kw)
    return d


def test_request_roundtrip_and_validation():
    req = NavigateRequest.from_dict(_req(hazards=[{"zone": "LEFT", "level": "HEAD", "proximity": 0.5,
                                                    "urgency": "WARNING", "distance_m": 1.1}]))
    assert req.session_id == "d1"
    again = NavigateRequest.from_dict(req.to_dict())
    assert again.hazards[0].level == "HEAD"
    with pytest.raises(ValidationError):
        NavigateRequest.from_dict(_req(query=""))
    with pytest.raises(ValidationError):
        NavigateRequest.from_dict(_req(hazards=[{"zone": "UP", "level": "HEAD", "proximity": 0.5, "urgency": "WARNING"}]))
    with pytest.raises(ValidationError):
        NavigateRequest.from_dict({"query": "x"})
    # ambient mode needs no question
    assert NavigateRequest.from_dict(_req(query="", mode="ambient")).mode == "ambient"


def test_model_answer_sanitises_tool_input():
    a = ModelAnswer.from_tool_input({"speech": " hi ", "target_bbox": [1, 2, "x", 4],
                                     "suggested_direction": "upwards", "confidence": 7})
    assert a.speech == "hi" and a.target_bbox is None
    assert a.suggested_direction == "none" and a.confidence == 1.0
