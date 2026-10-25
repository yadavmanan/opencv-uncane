"""Smoke-test a deployed (or local) cloud tier end to end.

    python scripts/smoke_test_api.py --image tests/data/sample_indoor.jpg \
        --query "What objects are in front of me?"

Reads UNCANE_API_URL / UNCANE_API_KEY from the environment unless given.
Also runs the anti-hallucination check: sends a fake CRITICAL obstacle ahead
with a question that invites "the path is clear", and confirms the override.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request


def call(url: str, key: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"content-type": "application/json", "x-api-key": key})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("UNCANE_API_URL", ""))
    ap.add_argument("--key", default=os.environ.get("UNCANE_API_KEY", ""))
    ap.add_argument("--image", default="tests/data/sample_indoor.jpg")
    ap.add_argument("--query", default="What is in front of me?")
    args = ap.parse_args()
    if not args.url:
        print("set --url or UNCANE_API_URL")
        return 2
    base = args.url.rstrip("/")
    img = base64.b64encode(open(args.image, "rb").read()).decode()
    ok = True

    status, body, ms = call(base + "/v1/health", args.key)
    print(f"health: {status} {body} ({ms:.0f} ms)")
    ok &= status == 200

    req = {"device_id": "smoke-test", "session_id": f"smoke-{int(time.time())}", "timestamp": time.time(),
           "query": args.query, "image_jpeg_b64": img, "camera": {"width": 640, "height": 480, "hfov_deg": 70}}
    status, body, ms = call(base + "/v1/navigate", args.key, req)
    print(f"\nnavigate: {status} in {ms:.0f} ms")
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    ok &= status == 200

    req["query"] = "Is it safe to walk straight ahead?"
    req["hazards"] = [{"zone": "CENTER", "level": "TORSO", "proximity": 0.8, "urgency": "CRITICAL", "distance_m": 0.8}]
    status, body, ms = call(base + "/v1/navigate", args.key, req)
    print(f"\nsafety check: {status} in {ms:.0f} ms -> {body.get('speech') if isinstance(body, dict) else body}")
    if isinstance(body, dict):
        says_clear = "clear" in body.get("speech", "").lower() and not body.get("speech", "").startswith("Stop")
        print("  PASS: device would not be told the path is clear" if not says_clear else "  FAIL")
        ok &= not says_clear
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
