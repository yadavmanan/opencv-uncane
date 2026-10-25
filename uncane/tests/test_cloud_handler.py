"""Lambda handler + Bedrock agent loop, fully offline (mock VLM / fake Anthropic client)."""
import base64
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from uncane.cloud import handler as H
from uncane.cloud.vlm import BedrockClaudeAgent, MockVLM
from uncane.common.schemas import NavigateRequest

JPEG = base64.b64encode(cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()).decode()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for k in ("SESSIONS_TABLE", "TELEMETRY_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLM_BACKEND", "mock")
    H._backend = H._memory = H._telemetry = None
    yield
    H._backend = H._memory = H._telemetry = None


def call(body, method="POST", path="/v1/navigate"):
    ev = {"httpMethod": method, "resource": path, "body": json.dumps(body) if body is not None else None}
    res = H.lambda_handler(ev, SimpleNamespace(aws_request_id="req-1"))
    return res["statusCode"], json.loads(res["body"])


def body(**kw):
    d = {"device_id": "dev", "session_id": "s", "query": "Where is the red water bottle?", "image_jpeg_b64": JPEG,
         "camera": {"width": 640, "height": 480, "hfov_deg": 70},
         "depth_grid": {"rows": 1, "cols": 4, "kind": "metric_m", "values": [3, 3, 2.0, 2.2]}}
    d.update(kw)
    return d


def test_health():
    code, b = call(None, "GET", "/v1/health")
    assert code == 200 and b["status"] == "ok"


def test_find_object_gets_grounded_guidance(capsys):
    code, b = call(body())
    assert code == 200, b
    assert b["speech"].startswith("Turn ")
    assert "step" in b["speech"]
    assert b["target"]["clock"] in (12, 1)
    assert b["latency_ms"]["total"] >= 0
    assert '"CloudWatchMetrics"' in capsys.readouterr().out   # EMF metrics emitted


def test_hallucinated_clear_path_is_overridden():
    hz = [{"zone": "CENTER", "level": "TORSO", "proximity": 0.8, "urgency": "CRITICAL", "distance_m": 0.7}]
    code, b = call(body(query="Can I walk forward?", hazards=hz))
    assert code == 200
    assert b["safety_override"] is True
    assert b["speech"].startswith("Stop.")
    assert b["raw_model_speech"] == "The path ahead is clear."


def test_bad_requests():
    assert call(body(image_jpeg_b64="notbase64!!"))[0] == 400
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 20).decode()
    assert call(body(image_jpeg_b64=png))[0] == 400
    assert call(body(query=""))[0] == 400
    assert call(None, "GET", "/nope")[0] == 404


def test_backend_failure_returns_502(monkeypatch):
    class Boom:
        def run(self, *a):
            raise RuntimeError("bedrock down")
    H._deps()
    H._backend = Boom()
    assert call(body())[0] == 502


# ---------------------------------------------------------------- agent loop
class FakeMessages:
    """Scripted Anthropic client: first asks for a tool, then answers."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def block(**kw):
    return SimpleNamespace(**kw)


def test_bedrock_agent_tool_loop(monkeypatch):
    monkeypatch.setenv("DEMO_TRANSIT", "true")
    agent = BedrockClaudeAgent.__new__(BedrockClaudeAgent)
    agent.model_id, agent.effort = "anthropic.claude-sonnet-5", "low"
    agent.client = SimpleNamespace(messages=FakeMessages([
        SimpleNamespace(stop_reason="tool_use", content=[
            block(type="tool_use", id="t1", name="get_transit_arrivals", input={"route": "12"})]),
        SimpleNamespace(stop_reason="tool_use", content=[
            block(type="tool_use", id="t2", name="final_answer",
                  input={"speech": "Bus 12 in 4 minutes (demo data).", "claims_path_clear": False,
                         "needs_better_view": False})]),
    ]))
    req = NavigateRequest.from_dict(body(query="When is the next 12 bus?"))
    res = agent.run(req, history=[])
    assert res.answer.speech.startswith("Bus 12")
    assert res.tools_used == ["get_transit_arrivals", "final_answer"]
    calls = agent.client.messages.calls
    assert calls[0]["output_config"] == {"effort": "low"}
    assert calls[0]["messages"][0]["content"][0]["type"] == "image"
    tool_result = calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["is_error"] is False
    assert json.loads(tool_result["content"])["demo_data"] is True


def test_bedrock_agent_unconfigured_tool_is_error_and_haiku_has_no_effort():
    agent = BedrockClaudeAgent.__new__(BedrockClaudeAgent)
    agent.model_id, agent.effort = "anthropic.claude-haiku-4-5", "low"
    agent.client = SimpleNamespace(messages=FakeMessages([
        SimpleNamespace(stop_reason="tool_use", content=[block(type="tool_use", id="t1", name="get_location_context", input={})]),
        SimpleNamespace(stop_reason="end_turn", content=[block(type="text", text="I can't tell where you are.")]),
    ]))
    res = agent.run(NavigateRequest.from_dict(body(query="Where am I?")), history=[])
    calls = agent.client.messages.calls
    assert "output_config" not in calls[0]
    assert calls[1]["messages"][-1]["content"][0]["is_error"] is True
    assert res.answer.speech == "I can't tell where you are."


def test_refusal_gives_fallback():
    agent = BedrockClaudeAgent.__new__(BedrockClaudeAgent)
    agent.model_id, agent.effort = "anthropic.claude-sonnet-5", "low"
    agent.client = SimpleNamespace(messages=FakeMessages([SimpleNamespace(stop_reason="refusal", content=[])]))
    res = agent.run(NavigateRequest.from_dict(body()), history=[])
    assert res.stop_reason == "refusal" and "try again" in res.answer.speech


def test_mock_backend_is_deterministic():
    r = MockVLM().run(NavigateRequest.from_dict(body()), [])
    assert r.answer.target_bbox
