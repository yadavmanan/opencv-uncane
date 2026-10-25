"""AWS Lambda entry point behind API Gateway (REST API, API-key protected).

Routes:
  GET  /v1/health    -> {"status": "ok", ...}
  POST /v1/navigate  -> NavigateRequest in, NavigateResponse out

Pipeline per request: validate -> load session memory -> VLM agent (Bedrock)
-> ground target with camera geometry + edge depth -> safety verifier ->
save memory -> telemetry -> respond.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from ..common.schemas import NavigateRequest, NavigateResponse, ValidationError
from ..common.spatial import guidance_sentence, locate_target
from ..common.verifier import SafetyVerifier
from .memory import SessionMemory
from .telemetry import Telemetry, emit_metrics
from .vlm import make_backend

log = logging.getLogger()
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

_backend = None
_memory: Optional[SessionMemory] = None
_telemetry: Optional[Telemetry] = None
_verifier = SafetyVerifier()


def _deps():
    """Create clients once per Lambda container (reused across warm invocations)."""
    global _backend, _memory, _telemetry
    if _backend is None:
        _backend = make_backend()
        _memory = SessionMemory()
        _telemetry = Telemetry()
    return _backend, _memory, _telemetry


def _response(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(body),
    }


def _check_jpeg(b64: str) -> None:
    try:
        head = base64.b64decode(b64[:16], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("image_jpeg_b64 is not valid base64") from exc
    if not head.startswith(b"\xff\xd8"):
        raise ValidationError("image must be a JPEG")


def navigate(req: NavigateRequest, request_id: str) -> NavigateResponse:
    t0 = time.perf_counter()
    backend, memory, telemetry = _deps()
    history = memory.load(req.session_id) if req.mode == "query" else []
    result = backend.run(req, history)
    answer = result.answer
    raw_speech = answer.speech

    target = None
    if answer.target_bbox and answer.target_label:
        target = locate_target(answer.target_label, answer.target_bbox, req.camera.hfov_deg, req.depth_grid)
        if target is not None and answer.speech:
            # Deterministic turn/steps first (from geometry + edge depth), then the model's description.
            answer.speech = f"{guidance_sentence(target)} {answer.speech}".strip()

    verified = _verifier.verify(answer, req.hazards, req.threats, target)
    total_ms = (time.perf_counter() - t0) * 1000
    resp = NavigateResponse(
        speech=verified.speech if answer.speech else "",
        raw_model_speech=raw_speech,
        target=target,
        safety_override=verified.overridden,
        override_reasons=verified.reasons,
        needs_better_view=answer.needs_better_view,
        tools_used=result.tools_used,
        model_id=result.model_id,
        latency_ms={"model": round(result.model_ms, 1), "total": round(total_ms, 1)},
        request_id=request_id,
    )
    if req.mode == "query":
        memory.save(req.session_id, req.query, resp.speech)
    telemetry.record(request_id, req.device_id, {
        "request_id": request_id, "device_id": req.device_id, "session_id": req.session_id,
        "timestamp": req.timestamp, "mode": req.mode, "query": req.query,
        "hazards": [h.__dict__ for h in req.hazards], "threats": [t.__dict__ for t in req.threats],
        "response": resp.to_dict(), "stop_reason": result.stop_reason,
    }, req.image_jpeg_b64)
    emit_metrics(
        {"Service": "navigator", "Model": result.model_id},
        {"ModelLatencyMs": result.model_ms, "TotalLatencyMs": total_ms,
         "SafetyOverride": 1.0 if verified.overridden else 0.0,
         "NeedsBetterView": 1.0 if answer.needs_better_view else 0.0},
        {"ModelLatencyMs": "Milliseconds", "TotalLatencyMs": "Milliseconds", "SafetyOverride": "Count",
         "NeedsBetterView": "Count"},
    )
    return resp


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    request_id = getattr(context, "aws_request_id", None) or uuid.uuid4().hex
    method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "")
    path = event.get("resource") or event.get("rawPath") or event.get("path") or ""

    if method == "GET" and path.endswith("/health"):
        return _response(200, {"status": "ok", "backend": os.environ.get("VLM_BACKEND", "bedrock"),
                               "model": os.environ.get("MODEL_ID", "anthropic.claude-sonnet-5")})
    if method != "POST" or not path.endswith("/navigate"):
        return _response(404, {"error": "not found"})

    try:
        raw = event.get("body") or ""
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8")
        req = NavigateRequest.from_dict(json.loads(raw))
        _check_jpeg(req.image_jpeg_b64)
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _response(400, {"error": str(exc), "request_id": request_id})

    try:
        resp = navigate(req, request_id)
    except Exception as exc:  # model/AWS failure: tell the device to degrade gracefully
        log.exception("navigate failed")
        return _response(502, {"error": f"reasoning unavailable: {type(exc).__name__}", "request_id": request_id})
    return _response(200, resp.to_dict())
