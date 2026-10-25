"""Agent tools the VLM can call, plus their executors.

`final_answer` is how the model returns structured output (Claude in Amazon
Bedrock does not support `output_config.format`, so a tool schema carries the
structure instead). The other tools give the agent context the camera cannot:
transit arrivals and the user's street location.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from ..common.schemas import DIRECTIONS, NavigateRequest

log = logging.getLogger(__name__)

FINAL_ANSWER_TOOL: Dict[str, Any] = {
    "name": "final_answer",
    "description": (
        "Deliver the spoken answer to the user. Call exactly once, as your last action. "
        "`speech` is read aloud verbatim, so keep it to one or two short sentences except when the user explicitly asks you to read text aloud; then include all legible requested text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "speech": {"type": "string", "description": "What to say to the user. Empty string only in ambient mode when nothing is worth saying."},
            "target_label": {"type": "string", "description": "Short name of the object the user asked to find, if any (e.g. 'red water bottle')."},
            "target_bbox": {
                "type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                "description": "Bounding box of that object in the photo as [x_min, y_min, x_max, y_max], each 0-1000 relative to image width/height. Omit if the object is not visible or no object was asked for.",
            },
            "suggested_direction": {"type": "string", "enum": list(DIRECTIONS),
                                    "description": "Which way you are suggesting the user move or turn, if any."},
            "claims_path_clear": {"type": "boolean", "description": "True if your speech says or implies the way ahead is free of obstacles."},
            "needs_better_view": {"type": "boolean", "description": "True if the photo is too blurry/dark/misaimed to answer."},
            "confidence": {"type": "number", "description": "0-1 confidence in the answer."},
        },
        "required": ["speech", "claims_path_clear", "needs_better_view"],
    },
}

TRANSIT_TOOL: Dict[str, Any] = {
    "name": "get_transit_arrivals",
    "description": "Upcoming public-transit arrivals near the user's current location. Use when asked about buses, trams or trains.",
    "input_schema": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "description": "Route number or name to filter by, if the user named one."},
            "stop_name": {"type": "string", "description": "Stop name read from a sign in the photo, if visible."},
        },
    },
}

LOCATION_TOOL: Dict[str, Any] = {
    "name": "get_location_context",
    "description": "Street address and nearest intersection for the user's GPS position. Use when asked 'where am I' or for orientation.",
    "input_schema": {"type": "object", "properties": {}},
}

ALL_TOOLS: List[Dict[str, Any]] = [LOCATION_TOOL, TRANSIT_TOOL, FINAL_ANSWER_TOOL]


class ToolError(Exception):
    pass


def _http_json(url: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_transit_arrivals(args: Dict[str, Any], req: NavigateRequest) -> Dict[str, Any]:
    """Calls TRANSIT_API_URL (your GTFS-realtime adapter). Never invents times."""
    if os.environ.get("DEMO_TRANSIT", "").lower() == "true":
        return {"demo_data": True, "stop": args.get("stop_name") or "Main St & 5th Ave",
                "arrivals": [{"route": args.get("route") or "12", "minutes": 4},
                             {"route": args.get("route") or "12", "minutes": 17}],
                "note": "DEMO DATA - tell the user these times are a demo."}
    base = os.environ.get("TRANSIT_API_URL")
    if not base:
        raise ToolError("Transit data is not configured for this deployment.")
    if not req.location:
        raise ToolError("The user's location is not available.")
    q = {"lat": req.location["lat"], "lon": req.location["lon"]}
    if args.get("route"):
        q["route"] = str(args["route"])
    return _http_json(f"{base}?{urllib.parse.urlencode(q)}")


def get_location_context(_args: Dict[str, Any], req: NavigateRequest) -> Dict[str, Any]:
    """Reverse-geocodes with Amazon Location Service (PLACE_INDEX_NAME)."""
    index = os.environ.get("PLACE_INDEX_NAME")
    if not index:
        raise ToolError("Location lookup is not configured for this deployment.")
    if not req.location:
        raise ToolError("The phone has not shared a GPS position.")
    import boto3

    client = boto3.client("location")
    res = client.search_place_index_for_position(
        IndexName=index, Position=[req.location["lon"], req.location["lat"]], MaxResults=1)
    if not res.get("Results"):
        raise ToolError("No address found for this position.")
    place = res["Results"][0]["Place"]
    return {"label": place.get("Label"), "street": place.get("Street"),
            "neighborhood": place.get("Neighborhood"), "municipality": place.get("Municipality")}


EXECUTORS = {
    "get_transit_arrivals": get_transit_arrivals,
    "get_location_context": get_location_context,
}


def run_tool(name: str, args: Dict[str, Any], req: NavigateRequest) -> tuple[str, bool]:
    """Returns (json_content, is_error)."""
    fn = EXECUTORS.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {name}"}), True
    try:
        return json.dumps(fn(args or {}, req)), False
    except ToolError as exc:
        return json.dumps({"error": str(exc)}), True
    except Exception as exc:  # network, AWS errors
        log.exception("tool %s failed", name)
        return json.dumps({"error": f"{name} failed: {type(exc).__name__}"}), True


def tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    return next((t for t in ALL_TOOLS if t["name"] == name), None)
