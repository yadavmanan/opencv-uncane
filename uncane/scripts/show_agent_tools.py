"""Shows whether the agent really calls the OpenCV tools, against any UnCane cloud endpoint.

    # local server (python -m uncane.cloud.local_server running)
    python scripts/show_agent_tools.py

    # deployed AWS stack (key from `aws apigateway get-api-key ... --include-value`)
    $env:UNCANE_API_KEY = "<key>"
    python scripts/show_agent_tools.py --url https://<id>.execute-api.us-east-1.amazonaws.com/prod --insecure

It sends a sharp and a blurred sign, prints the spoken answer and every tool call, and exits 1 if no
tool was called at all.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def sign(blur: float = 0.0):
    img = np.full((480, 640, 3), 235, np.uint8)
    cv2.putText(img, "EXIT", (60, 150), cv2.FONT_HERSHEY_DUPLEX, 3.0, (20, 20, 20), 6)
    cv2.putText(img, "STAIRS LEFT", (60, 300), cv2.FONT_HERSHEY_DUPLEX, 2.0, (20, 20, 20), 4)
    return cv2.GaussianBlur(img, (0, 0), blur) if blur else img


def call(url: str, key: str, path: str, body, timeout: float, context):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data, method="POST" if data else "GET",
                                 headers={"content-type": "application/json", "x-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        return json.load(resp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("UNCANE_API_URL", "http://127.0.0.1:8787"))
    ap.add_argument("--key", default=os.environ.get("UNCANE_API_KEY", "local"),
                    help="API Gateway key (or set UNCANE_API_KEY). The local server uses 'local'.")
    ap.add_argument("--query", default="What does the sign say?")
    ap.add_argument("--image", help="send this photo instead of the generated signs")
    ap.add_argument("--insecure", action="store_true", help="skip TLS certificate checks (corporate proxy)")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "eval" / "live_traces.json"))
    args = ap.parse_args()

    context = ssl._create_unverified_context() if args.insecure else None
    print(f"OpenCV {cv2.__version__} | endpoint {args.url}" + ("  (TLS NOT verified)" if args.insecure else ""))
    try:
        health = call(args.url, args.key, "/v1/health", None, 15, context)
    except (urllib.error.URLError, OSError) as exc:
        print(f"Cannot reach the endpoint: {exc}\nIs the server running, and are the URL and key right?")
        return 2
    print(f"health: {health}\n")

    if args.image:
        photo = cv2.imread(args.image)
        if photo is None:
            print(f"Could not read {args.image}")
            return 2
        cases = [("your photo", photo)]
    else:
        cases = [("sharp sign", sign()), ("blurred sign", sign(blur=9))]

    saved, total_calls = [], 0
    for label, img in cases:
        body = {"device_id": "show-tools", "session_id": f"show-tools-{label}".replace(" ", "-"),
                "query": args.query, "camera": {"width": 640, "height": 480, "hfov_deg": 70},
                "image_jpeg_b64": base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()}
        try:
            resp = call(args.url, args.key, "/v1/navigate", body, args.timeout, context)
        except urllib.error.HTTPError as exc:
            print(f"--- {label}: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:300]}")
            return 2
        trace = resp.get("agent_trace") or []
        total_calls += len(trace)
        print(f"--- {label}")
        print(f"  answer         : {resp.get('speech')}")
        print(f"  needs re-aim   : {resp.get('needs_better_view')}")
        print(f"  model / total  : {resp.get('latency_ms')}")
        print(f"  tools_used     : {resp.get('tools_used')}")
        if not trace:
            print("  agent_trace    : EMPTY - no OpenCV tool was called")
        for step in trace:
            mark = "ok " if step.get("ok") else "ERR"
            print(f"  {step['step']}. [{mark}] {step['tool']:<22} {step.get('ms', 0):>6.1f} ms  "
                  f"opencv={step.get('opencv')}  image_back={step.get('returned_image')}")
            print(f"       args   {json.dumps(step.get('args'))}")
            print(f"       result {step.get('result', '')[:110]}")
        print()
        saved.append({"case": label, "response": resp})

    Path(args.out).write_text(json.dumps({"endpoint": args.url, "opencv": cv2.__version__, "cases": saved},
                                         indent=2), encoding="utf-8")
    print(f"full responses saved to {args.out}")
    if total_calls == 0:
        print("RESULT: no tool calls. The endpoint is running old code, OpenCV 5 is missing there, or the model "
              "chose not to use tools.")
        return 1
    print(f"RESULT: {total_calls} OpenCV tool call(s) recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
