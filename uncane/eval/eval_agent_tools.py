"""Agentic-vision evaluation: does OpenCV evidence change what the agent does next?

    python eval/eval_agent_tools.py                      # mock backend, offline
    VLM_BACKEND=mantle MANTLE_API_KEY=... python eval/eval_agent_tools.py --backend mantle

Each case renders a photo, runs it through the same handler path as production, and checks both the final
answer and which OpenCV tools were called. Traces are written to eval/agent_traces.json. With the mock
backend this proves the tools, loop and plumbing. Run it with a real model to measure the model's own
tool-use behaviour, and report those numbers separately.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncane.cloud import handler as H  # noqa: E402
from uncane.common.schemas import NavigateRequest  # noqa: E402

CRITICAL_CHAIR = {"zone": "CENTER", "level": "TORSO", "proximity": 0.8, "urgency": "CRITICAL",
                  "distance_m": 0.8, "label": "chair"}


def sign(blur=0.0, dark=False):
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.putText(img, "EXIT", (60, 150), cv2.FONT_HERSHEY_DUPLEX, 3.0, (20, 20, 20), 6)
    cv2.putText(img, "STAIRS LEFT", (60, 300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (20, 20, 20), 4)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return (img * 0.05).astype(np.uint8) if dark else img


def bottle(bgr):
    img = np.full((480, 640, 3), 200, np.uint8)
    img[216:297, 384:435] = bgr
    return img


GRID = {"rows": 1, "cols": 4, "kind": "metric_m", "values": [3, 3, 2.1, 2.1]}
READ = "What does the sign say?"
FIND = "Where is the red water bottle?"
CASES = [
    # name, image, query, extras, expectation(tools, answer, response) -> bool
    ("sharp sign is read", sign(), READ, {},
     lambda t, a, r: "zoom_region" in t and not r["needs_better_view"]),
    ("blurry sign: re-aim, no guess", sign(blur=9), READ, {},
     lambda t, a, r: "zoom_region" not in t and r["needs_better_view"]),
    ("dark sign: re-aim, no guess", sign(dark=True), READ, {},
     lambda t, a, r: "zoom_region" not in t and r["needs_better_view"]),
    ("red bottle is confirmed", bottle((30, 30, 200)), FIND, {"depth_grid": GRID},
     lambda t, a, r: "analyze_region_color" in t and r["target"] is not None and "could not confirm" not in a),
    ("blue bottle is not called red", bottle((200, 60, 30)), FIND, {"depth_grid": GRID},
     lambda t, a, r: "analyze_region_color" in t and "could not confirm" in a),
    ("clear-path claim is overridden", sign(), "Can I walk forward?", {"hazards": [CRITICAL_CHAIR]},
     lambda t, a, r: r["safety_override"] and a.startswith("Stop.")),
]


def run_case(img, query, extras):
    body = {"device_id": "eval", "session_id": "eval", "query": query,
            "image_jpeg_b64": base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode(),
            "camera": {"width": 640, "height": 480, "hfov_deg": 70}, **extras}
    with contextlib.redirect_stdout(io.StringIO()):   # the handler prints CloudWatch metric lines
        return H.navigate(NavigateRequest.from_dict(body), "eval").to_dict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("VLM_BACKEND", "mock"))
    ap.add_argument("--out", default=str(Path(__file__).with_name("agent_traces.json")))
    args = ap.parse_args()
    for key in ("SESSIONS_TABLE", "TELEMETRY_BUCKET"):
        os.environ.pop(key, None)
    os.environ["VLM_BACKEND"] = args.backend
    H._backend = H._memory = H._telemetry = None
    H._deps()

    rows, traces = [], []
    for name, img, query, extras, check in CASES:
        resp = run_case(img, query, extras)
        used = [t["tool"] for t in resp["agent_trace"]]
        ok = bool(check(used, resp["speech"], resp))
        rows.append((name, ok, used, resp["speech"], resp["latency_ms"].get("model", 0.0)))
        traces.append({"case": name, "query": query, "passed": ok, "speech": resp["speech"],
                       "safety_override": resp["safety_override"], "needs_better_view": resp["needs_better_view"],
                       "agent_trace": resp["agent_trace"]})
    width = max(len(r[0]) for r in rows)
    for name, ok, used, speech, ms in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  tools={used or '-'}  {ms:.0f} ms\n      -> {speech}")
    passed = sum(r[1] for r in rows)
    print(f"\n{passed}/{len(rows)} cases passed with backend '{args.backend}'")
    payload = {"backend": args.backend, "opencv": cv2.__version__, "cases": traces}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"traces written to {args.out} (OpenCV {cv2.__version__})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
