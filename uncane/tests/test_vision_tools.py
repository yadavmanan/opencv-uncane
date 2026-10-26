"""OpenCV agent tools: real images in, measured evidence out, and that evidence changes what the agent does next."""
import base64
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from uncane.cloud import handler as H
from uncane.cloud.vision_tools import VisionToolbox
from uncane.cloud.vlm import BedrockClaudeAgent, MantleOpenAIAgent, MockVLM
from uncane.common.schemas import NavigateRequest


def b64(img) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def test_opencv_5_is_what_runs_the_tools():
    assert int(cv2.__version__.split(".")[0]) >= 5
    box = VisionToolbox(make_req(sign_image()))
    box.call("assess_image_quality", {})
    assert box.trace[0]["opencv"] == cv2.__version__


def sign_image():
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.putText(img, "EXIT", (60, 150), cv2.FONT_HERSHEY_DUPLEX, 3.0, (20, 20, 20), 6)
    cv2.putText(img, "STAIRS LEFT", (60, 300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (20, 20, 20), 4)
    return img


def bottle_image(bgr):
    img = np.full((480, 640, 3), 200, np.uint8)
    img[216:297, 384:435] = bgr            # the mock's bottle box: [600, 450, 680, 620] of 1000
    return img


def make_req(img, query="What does the sign say?", **extra):
    d = {"device_id": "dev", "session_id": "s", "query": query, "image_jpeg_b64": b64(img),
         "camera": {"width": 640, "height": 480, "hfov_deg": 70}}
    d.update(extra)
    return NavigateRequest.from_dict(d)


def run(img, name, **args):
    res = VisionToolbox(make_req(img)).call(name, args)
    assert not res.is_error, res.content
    return json.loads(res.content), res


# ----------------------------------------------------------------- individual tools
def test_quality_separates_sharp_blurred_and_dark():
    assert run(sign_image(), "assess_image_quality")[0]["verdict"] == "ok"
    blurry = cv2.GaussianBlur(sign_image(), (0, 0), 9)
    verdict = run(blurry, "assess_image_quality")[0]
    assert verdict["verdict"] == "blurry" and verdict["usable"] is False
    assert run(np.full((480, 640, 3), 10, np.uint8), "assess_image_quality")[0]["verdict"] == "too_dark"


def test_find_text_regions_locates_the_sign_text():
    regions = run(sign_image(), "find_text_regions")[0]["regions"]
    assert regions
    x0, y0, x1, y1 = regions[0]["bbox"]
    assert x1 > x0 and y1 > y0 and 0 <= x0 and x1 <= 1000 and 0 <= y0 and y1 <= 1000
    assert run(np.full((480, 640, 3), 128, np.uint8), "find_text_regions")[0]["regions"] == []


def test_zoom_region_returns_a_bigger_decodable_image():
    info, res = run(sign_image(), "zoom_region", bbox=[50, 100, 500, 400])
    out = cv2.imdecode(np.frombuffer(base64.b64decode(res.image_b64), np.uint8), cv2.IMREAD_COLOR)
    assert out is not None and out.shape[1] > 450 * 640 // 1000
    assert info["output_px"] == [out.shape[1], out.shape[0]] and info["scale"] > 1.0


def test_color_tool_confirms_or_rejects_the_expected_colour():
    red = bottle_image((30, 30, 200))
    assert run(red, "analyze_region_color", bbox=[600, 450, 680, 620], expected_color="red")[0]["matches_expected"]
    blue = bottle_image((200, 60, 30))
    info = run(blue, "analyze_region_color", bbox=[600, 450, 680, 620], expected_color="red")[0]
    assert info["dominant_color"] == "blue" and info["matches_expected"] is False


def test_depth_tool_uses_edge_depth_and_flags_critical_obstacles():
    hazard = {"zone": "CENTER", "level": "TORSO", "proximity": 0.8, "urgency": "CRITICAL", "distance_m": 0.9}
    req = make_req(sign_image(), depth_grid={"rows": 1, "cols": 4, "kind": "metric_m", "values": [3, 3, 2.1, 2.1]},
                   hazards=[hazard])
    out = json.loads(VisionToolbox(req).call("measure_region_depth", {"bbox": [500, 300, 900, 700]}).content)
    assert out["distance_m"] == pytest.approx(2.1) and out["steps"] == 3
    assert out["critical_obstacle_ahead"] is True and out["nearest_critical_obstacle_m"] == 0.9


@pytest.mark.parametrize("name,args", [
    ("zoom_region", {}), ("zoom_region", {"bbox": [10, 10, 10, 10]}), ("zoom_region", {"bbox": "nope"}),
    ("analyze_region_color", {"bbox": [0, 0]}), ("measure_region_depth", {"bbox": [0, 0, 500, 500]}),
    ("no_such_tool", {}),
])
def test_bad_arguments_are_errors_not_crashes(name, args):
    res = VisionToolbox(make_req(sign_image())).call(name, args)
    assert res.is_error and "error" in json.loads(res.content)


def test_tool_budget_is_enforced_and_every_call_is_traced():
    box = VisionToolbox(make_req(sign_image()))
    results = [box.call("assess_image_quality", {}) for _ in range(VisionToolbox.MAX_CALLS + 1)]
    assert results[-1].is_error and "budget" in results[-1].content
    assert [t["step"] for t in box.trace] == list(range(1, VisionToolbox.MAX_CALLS + 2))
    assert box.trace[0]["tool"] == "assess_image_quality" and box.trace[0]["ok"] is True


# ------------------------------------------------- the evidence changes what happens next
def test_mock_agent_reads_sign_only_after_quality_check_passes():
    ok = MockVLM().run(make_req(sign_image()), [])
    assert ok.tools_used == ["assess_image_quality", "find_text_regions", "zoom_region", "final_answer"]
    assert ok.answer.needs_better_view is False and ok.answer.speech.startswith("The sign says")
    assert [t["tool"] for t in ok.trace] == ["assess_image_quality", "find_text_regions", "zoom_region"]

    blurry = MockVLM().run(make_req(cv2.GaussianBlur(sign_image(), (0, 0), 9)), [])
    assert blurry.tools_used == ["assess_image_quality", "final_answer"]       # never zoomed
    assert blurry.answer.needs_better_view is True and "blurry" in blurry.answer.speech


def test_mock_agent_confirms_target_colour_before_claiming_it():
    q = "Where is the red water bottle?"
    red = MockVLM().run(make_req(bottle_image((30, 30, 200)), q), [])
    assert red.answer.speech == "The red water bottle is on the table." and red.answer.confidence == 0.8
    blue = MockVLM().run(make_req(bottle_image((200, 60, 30)), q), [])
    assert "could not confirm its colour" in blue.answer.speech and blue.answer.confidence < 0.5


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def tool_use(id_, name, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def claude_agent(script):
    agent = BedrockClaudeAgent.__new__(BedrockClaudeAgent)
    agent.model_id, agent.effort = "anthropic.claude-sonnet-5", "low"
    agent.client = SimpleNamespace(messages=FakeMessages(script))
    return agent


def test_claude_loop_feeds_the_zoomed_image_and_trace_back():
    agent = claude_agent([
        SimpleNamespace(stop_reason="tool_use", content=[tool_use("t1", "zoom_region", bbox=[50, 100, 500, 400])]),
        SimpleNamespace(stop_reason="tool_use", content=[tool_use(
            "t2", "final_answer", speech="The sign says Exit.", claims_path_clear=False, needs_better_view=False)]),
    ])
    res = agent.run(make_req(sign_image()), [])
    calls = agent.client.messages.calls
    assert "zoom_region" in {t["name"] for t in calls[0]["tools"]}
    content = calls[1]["messages"][-1]["content"][0]["content"]
    assert [b["type"] for b in content] == ["text", "image"]                  # the model sees the zoomed crop
    assert res.tools_used == ["zoom_region", "final_answer"]
    assert res.trace[0]["tool"] == "zoom_region" and res.trace[0]["returned_image"] is True


def test_claude_loop_quality_result_changes_the_next_tool_call():
    agent = claude_agent([
        SimpleNamespace(stop_reason="tool_use", content=[tool_use("t1", "assess_image_quality")]),
        SimpleNamespace(stop_reason="tool_use", content=[tool_use(
            "t2", "final_answer", speech="Too blurry. Hold still and ask again.", claims_path_clear=False,
            needs_better_view=True)]),
    ])
    res = agent.run(make_req(cv2.GaussianBlur(sign_image(), (0, 0), 9)), [])
    seen = json.loads(agent.client.messages.calls[1]["messages"][-1]["content"][0]["content"])
    assert seen["verdict"] == "blurry"
    assert res.answer.needs_better_view is True


def test_openai_style_loop_sends_the_image_as_a_follow_up_user_turn():
    def completion(*calls):
        msg = SimpleNamespace(content="", tool_calls=list(calls))
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=msg)])

    def call(id_, name, args):
        return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    recorded = []
    script = [
        completion(call("c1", "zoom_region", {"bbox": [50, 100, 500, 400]})),
        completion(call("c2", "final_answer", {"speech": "Exit.", "claims_path_clear": False,
                                               "needs_better_view": False})),
    ]
    agent = MantleOpenAIAgent.__new__(MantleOpenAIAgent)
    agent.model_id = "qwen"
    agent._tools = []
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: (recorded.append(kw), script.pop(0))[1])))
    res = agent.run(make_req(sign_image()), [])
    last = recorded[1]["messages"][-1]
    assert last["role"] == "user" and any(b["type"] == "image_url" for b in last["content"])
    assert recorded[1]["messages"][-2]["role"] == "tool"
    assert res.trace[0]["tool"] == "zoom_region"


# ------------------------------------------------------------------ end to end via the Lambda
@pytest.fixture
def offline(monkeypatch):
    for k in ("SESSIONS_TABLE", "TELEMETRY_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLM_BACKEND", "mock")
    H._backend = H._memory = H._telemetry = None
    yield
    H._backend = H._memory = H._telemetry = None


def post(img, query):
    ev = {"httpMethod": "POST", "resource": "/v1/navigate",
          "body": json.dumps({"device_id": "d", "session_id": "s", "query": query, "image_jpeg_b64": b64(img),
                              "camera": {"width": 640, "height": 480, "hfov_deg": 70}})}
    res = H.lambda_handler(ev, SimpleNamespace(aws_request_id="req-1"))
    return res["statusCode"], json.loads(res["body"])


def test_response_exposes_the_agent_trace(offline):
    code, body = post(sign_image(), "What does the sign say?")
    assert code == 200 and body["needs_better_view"] is False
    assert [t["tool"] for t in body["agent_trace"]] == ["assess_image_quality", "find_text_regions", "zoom_region"]
    assert body["tools_used"][-1] == "final_answer"

    code, body = post(cv2.GaussianBlur(sign_image(), (0, 0), 9), "What does the sign say?")
    assert body["needs_better_view"] is True and len(body["agent_trace"]) == 1
