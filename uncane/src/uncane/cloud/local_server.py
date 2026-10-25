"""Run the Lambda handler locally over HTTP - no deployment needed.

    VLM_BACKEND=mock python -m uncane.cloud.local_server          # fully offline
    python -m uncane.cloud.local_server                            # real Bedrock with your AWS creds

Then point the edge at it:  UNCANE_API_URL=http://127.0.0.1:8787  UNCANE_API_KEY=local
"""
from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .handler import lambda_handler


class _Handler(BaseHTTPRequestHandler):
    api_key = "local"

    def _dispatch(self, method: str) -> None:
        if self.headers.get("x-api-key") != self.api_key:
            self._send(403, {"message": "Forbidden"})
            return
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length).decode("utf-8") if length else ""
        event = {"httpMethod": method, "path": self.path, "body": body, "isBase64Encoded": False}
        result = lambda_handler(event, None)
        self._send(result["statusCode"], json.loads(result["body"]))

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--api-key", default="local")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    _Handler.api_key = args.api_key
    print(f"local cloud tier on http://{args.host}:{args.port}  (x-api-key: {args.api_key})")
    ThreadingHTTPServer((args.host, args.port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
