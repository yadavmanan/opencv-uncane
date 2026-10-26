"""VLM backends for Hemisphere 2 (cloud reasoning).

* MantleOpenAIAgent - open VLM (Qwen3-VL) on Bedrock Mantle, API-key auth. Set VLM_BACKEND=mantle.
* BedrockClaudeAgent - Claude in Amazon Bedrock (Messages API via the
  `AnthropicBedrockMantle` client) with a short tool-use loop. Default.
* SageMakerVLM       - a VizWiz fine-tuned open VLM on a SageMaker endpoint
  (see training/sagemaker). Returns the same ModelAnswer.
* MockVLM            - deterministic answers for tests and offline development.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..common.prompts import SYSTEM_PROMPT, user_turn_text
from ..common.schemas import ModelAnswer, NavigateRequest
from .tools import agent_tools, run_tool
from .vision_tools import VISION_TOOL_NAMES, ToolResult, VisionToolbox

log = logging.getLogger(__name__)

MAX_AGENT_TURNS = 5
FALLBACK_SPEECH = "Sorry, I couldn't work that out. Please try again."


@dataclass
class AgentResult:
    answer: ModelAnswer
    model_id: str
    model_ms: float
    tools_used: List[str] = field(default_factory=list)
    stop_reason: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)   # one entry per tool call, for observability


def image_block(jpeg_b64: str) -> Dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64}}


def _run_agent_tool(name: str, args: Dict[str, Any], req: NavigateRequest, toolbox: VisionToolbox) -> ToolResult:
    """Runs one tool call and records it in the trace. OpenCV tools may return an image for the model."""
    if name in VISION_TOOL_NAMES:
        return toolbox.call(name, args)
    started = time.perf_counter()
    content, is_error = run_tool(name, args, req)
    result = ToolResult(content, is_error)
    toolbox.record(name, args, result, started)
    return result


class _ToolLoopMixin:
    """Gives every tool-calling backend a per-request OpenCV toolbox and attaches its trace to the result."""

    def run(self, req: NavigateRequest, history: List[Dict[str, Any]]) -> AgentResult:
        toolbox = VisionToolbox(req)
        result = self._run_loop(req, history, toolbox)
        result.trace = toolbox.trace
        return result


class BedrockClaudeAgent(_ToolLoopMixin):
    def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None,
                 effort: Optional[str] = None, timeout_s: float = 12.0):
        from anthropic import AnthropicBedrockMantle

        self.model_id = model_id or os.environ.get("MODEL_ID", "anthropic.claude-sonnet-5")
        region = region or os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "us-east-1")
        self.effort = effort or os.environ.get("EFFORT", "low")
        self.client = AnthropicBedrockMantle(aws_region=region, timeout=timeout_s, max_retries=1)

    def _extra(self) -> Dict[str, Any]:
        # Haiku 4.5 rejects the effort parameter; newer models use it to trade
        # depth for latency (low effort keeps answers well under the 1.5 s budget).
        if "haiku" in self.model_id:
            return {}
        return {"output_config": {"effort": self.effort}}

    def _run_loop(self, req: NavigateRequest, history: List[Dict[str, Any]], toolbox: VisionToolbox) -> AgentResult:
        messages: List[Dict[str, Any]] = list(history) + [{
            "role": "user",
            "content": [image_block(req.image_jpeg_b64), {"type": "text", "text": user_turn_text(req)}],
        }]
        system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        tools_used: List[str] = []
        t0 = time.perf_counter()
        stop_reason = ""
        for _ in range(MAX_AGENT_TURNS):
            resp = self.client.messages.create(
                model=self.model_id, max_tokens=4096, system=system, tools=agent_tools(),
                messages=messages, **self._extra(),
            )
            stop_reason = resp.stop_reason or ""
            if stop_reason == "refusal":
                log.warning("model refused: %s", getattr(resp, "stop_details", None))
                break
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            final = next((b for b in tool_uses if b.name == "final_answer"), None)
            if final is not None:
                tools_used.append("final_answer")
                ans = ModelAnswer.from_tool_input(final.input if isinstance(final.input, dict) else {})
                return AgentResult(ans, self.model_id, (time.perf_counter() - t0) * 1000, tools_used, stop_reason)
            if not tool_uses:
                text = " ".join(b.text for b in resp.content if b.type == "text").strip()
                ans = ModelAnswer(speech=text or FALLBACK_SPEECH)
                return AgentResult(ans, self.model_id, (time.perf_counter() - t0) * 1000, tools_used, stop_reason)
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in tool_uses:  # all results go back in ONE user message
                tools_used.append(b.name)
                out = _run_agent_tool(b.name, b.input if isinstance(b.input, dict) else {}, req, toolbox)
                content = out.content if out.image_b64 is None else [
                    {"type": "text", "text": out.content}, image_block(out.image_b64)]
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content,
                                "is_error": out.is_error})
            messages.append({"role": "user", "content": results})
        return AgentResult(ModelAnswer(speech=FALLBACK_SPEECH), self.model_id,
                           (time.perf_counter() - t0) * 1000, tools_used, stop_reason or "max_turns")


class MantleOpenAIAgent(_ToolLoopMixin):
    """Open-weight VLM (default Qwen3-VL) on Bedrock Mantle's OpenAI-compatible
    chat-completions route, authenticated with a Mantle API key. Same tool loop
    and the same final_answer contract as BedrockClaudeAgent.
    """

    def __init__(self, model_id: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout_s: float = 12.0):
        from openai import OpenAI

        self.model_id = model_id or os.environ.get("MODEL_ID", "qwen.qwen3-vl-235b-a22b-instruct")
        region = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "us-east-1")
        self.base_url = base_url or os.environ.get("MANTLE_BASE_URL") or f"https://bedrock-mantle.{region}.api.aws/v1"
        key = api_key or os.environ["MANTLE_API_KEY"]
        self.client = OpenAI(base_url=self.base_url, api_key=key, timeout=timeout_s, max_retries=1)
        self._tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in agent_tools()]

    def _run_loop(self, req: NavigateRequest, history: List[Dict[str, Any]], toolbox: VisionToolbox) -> AgentResult:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_jpeg_b64}"}},
            {"type": "text", "text": user_turn_text(req)},
        ]})
        tools_used: List[str] = []
        t0 = time.perf_counter()
        stop_reason = ""
        for _ in range(MAX_AGENT_TURNS):
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=messages, tools=self._tools, tool_choice="auto",
                max_tokens=1024, temperature=0.1,
            )
            choice = resp.choices[0]
            stop_reason = choice.finish_reason or ""
            msg = choice.message
            calls = list(msg.tool_calls or [])
            final = next((c for c in calls if c.function.name == "final_answer"), None)
            if final is not None:
                tools_used.append("final_answer")
                ans = ModelAnswer.from_tool_input(_parse_args(final.function.arguments))
                return AgentResult(ans, self.model_id, (time.perf_counter() - t0) * 1000, tools_used, stop_reason)
            if not calls:
                ans = ModelAnswer(speech=(msg.content or "").strip() or FALLBACK_SPEECH)
                return AgentResult(ans, self.model_id, (time.perf_counter() - t0) * 1000, tools_used, stop_reason)
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name,
                                                              "arguments": c.function.arguments}} for c in calls]})
            images: List[Dict[str, Any]] = []
            for c in calls:  # every tool result goes back before the next model turn
                tools_used.append(c.function.name)
                out = _run_agent_tool(c.function.name, _parse_args(c.function.arguments), req, toolbox)
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": out.content if not out.is_error else f"ERROR: {out.content}"})
                if out.image_b64:   # tool messages are text-only here, so the image follows as a user turn
                    images += [{"type": "text", "text": f"Image returned by {c.function.name}:"},
                               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{out.image_b64}"}}]
            if images:
                messages.append({"role": "user", "content": images})
        return AgentResult(ModelAnswer(speech=FALLBACK_SPEECH), self.model_id,
                           (time.perf_counter() - t0) * 1000, tools_used, stop_reason or "max_turns")


def _parse_args(raw: Optional[str]) -> Dict[str, Any]:
    try:
        val = json.loads(raw or "{}")
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        return {}


_JSON_RE = re.compile(r"\{.*\}", re.S)


class SageMakerVLM:
    """Calls a SageMaker endpoint that speaks the OpenAI-style chat 'messages' API
    (Hugging Face TGI / LMI containers do). The model is asked for JSON matching
    the final_answer schema. No tool use on this path.
    """

    def __init__(self, endpoint: Optional[str] = None, region: Optional[str] = None):
        import boto3

        self.endpoint = endpoint or os.environ["SAGEMAKER_ENDPOINT"]
        self.client = boto3.client("sagemaker-runtime", region_name=region or os.environ.get("AWS_REGION"))
        self.model_id = f"sagemaker:{self.endpoint}"

    def run(self, req: NavigateRequest, history: List[Dict[str, Any]]) -> AgentResult:
        instructions = (
            SYSTEM_PROMPT.split("Use the other tools")[0]
            + 'Reply with only a JSON object: {"speech": str, "target_label": str|null, '
              '"target_bbox": [x0,y0,x1,y1] 0-1000 or null, "suggested_direction": str, '
              '"claims_path_clear": bool, "needs_better_view": bool}'
        )
        payload = {
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_jpeg_b64}"}},
                    {"type": "text", "text": user_turn_text(req)},
                ]},
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }
        t0 = time.perf_counter()
        res = self.client.invoke_endpoint(EndpointName=self.endpoint, ContentType="application/json",
                                          Body=json.dumps(payload))
        body = json.loads(res["Body"].read())
        text = body["choices"][0]["message"]["content"] if "choices" in body else str(body)
        m = _JSON_RE.search(text)
        try:
            ans = ModelAnswer.from_tool_input(json.loads(m.group(0))) if m else ModelAnswer(speech=text.strip())
        except (json.JSONDecodeError, ValueError):
            ans = ModelAnswer(speech=text.strip())
        return AgentResult(ans, self.model_id, (time.perf_counter() - t0) * 1000, [], "end_turn")


class MockVLM:
    """Offline stand-in. It drives the real OpenCV tools, so what they measure decides its answer."""

    model_id = "mock"
    _QUALITY_WORDS = {"too_dark": "too dark", "overexposed": "overexposed", "blurry": "too blurry",
                      "low_contrast": "too washed out"}

    def run(self, req: NavigateRequest, history: List[Dict[str, Any]]) -> AgentResult:
        base64.b64decode(req.image_jpeg_b64[:64] + "==", validate=False)
        toolbox = VisionToolbox(req)
        tools_used: List[str] = []

        def use(name: str, **args: Any) -> Dict[str, Any]:
            tools_used.append(name)
            res = toolbox.call(name, args)
            return {} if res.is_error else json.loads(res.content)   # no OpenCV: behave as before

        q = req.query.lower()
        if "bottle" in q:
            bbox = [600, 450, 680, 620]
            confirmed = use("analyze_region_color", bbox=bbox, expected_color="red").get("matches_expected", True)
            speech = ("The red water bottle is on the table." if confirmed else
                      "I think the red water bottle is on the table, but I could not confirm its colour.")
            ans = ModelAnswer(speech=speech, target_label="red water bottle", target_bbox=bbox,
                              suggested_direction="slightly_right", confidence=0.8 if confirmed else 0.4)
        elif "clear" in q or "walk" in q:
            ans = ModelAnswer(speech="The path ahead is clear.", claims_path_clear=True)
        elif "sign" in q or "read" in q:
            quality = use("assess_image_quality")
            if quality and not quality.get("usable", True):
                why = self._QUALITY_WORDS.get(quality.get("verdict"), "unclear")
                ans = ModelAnswer(speech=f"The photo is {why} to read. Hold still, aim at the sign and ask again.",
                                  needs_better_view=True)
            else:
                regions = use("find_text_regions").get("regions") or []
                if regions:
                    use("zoom_region", bbox=regions[0]["bbox"])
                ans = ModelAnswer(speech="The sign says: Exit, stairs to the left.")
        else:
            ans = ModelAnswer(speech="You are in a room with a table ahead and a chair on your left.")
        return AgentResult(ans, self.model_id, 1.0, tools_used + ["final_answer"], "end_turn", trace=toolbox.trace)


def make_backend(name: Optional[str] = None):
    name = (name or os.environ.get("VLM_BACKEND", "bedrock")).lower()
    if name == "mock":
        return MockVLM()
    if name == "sagemaker":
        return SageMakerVLM()
    if name == "mantle":
        return MantleOpenAIAgent()
    return BedrockClaudeAgent()
