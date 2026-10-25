"""Edge -> AWS client. Never blocks the reflex loop.

Requests run on a worker thread; results come back through a callback. After
``offline_after_failures`` consecutive failures the client reports itself
offline (the app tells the user once) and keeps probing in the background, so
physical safety never depends on connectivity.
"""
from __future__ import annotations

import json
import logging
import queue
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from ..common.schemas import NavigateRequest, NavigateResponse

log = logging.getLogger(__name__)


@dataclass
class CloudResult:
    request: NavigateRequest
    response: Optional[NavigateResponse]
    error: Optional[str]
    sent_at: float
    round_trip_ms: float


ResultCallback = Callable[[CloudResult], None]


class CloudClient:
    def __init__(self, api_url: str, api_key: str, on_result: ResultCallback, timeout_s: float = 8.0,
                 offline_after_failures: int = 3, verify_ssl: bool = True):
        self.url = api_url.rstrip("/") + "/v1/navigate" if api_url else ""
        self.health_url = api_url.rstrip("/") + "/v1/health" if api_url else ""
        self.api_key = api_key
        self._ssl_context = None
        if not verify_ssl:
            log.warning("cloud.verify_ssl is false: TLS certificates are NOT being verified")
            self._ssl_context = ssl._create_unverified_context()
        self.on_result = on_result
        self.timeout_s = timeout_s
        self.offline_after = offline_after_failures
        self.failures = 0
        self.in_flight = False
        self._q: "queue.Queue[NavigateRequest]" = queue.Queue(maxsize=2)
        threading.Thread(target=self._run, name="cloud", daemon=True).start()

    @property
    def configured(self) -> bool:
        return bool(self.url)

    @property
    def online(self) -> bool:
        return self.configured and self.failures < self.offline_after

    def submit(self, req: NavigateRequest) -> bool:
        """Queue a request; drops the oldest if the user fires questions faster than the cloud answers."""
        if not self.configured:
            return False
        try:
            self._q.put_nowait(req)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(req)
        return True

    def _post(self, url: str, body: Optional[bytes]) -> dict:
        headers = {"content-type": "application/json", "x-api-key": self.api_key}
        req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=self._ssl_context) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> bool:
        try:
            return self._post(self.health_url, None).get("status") == "ok"
        except Exception:
            return False

    def _run(self) -> None:
        while True:
            req = self._q.get()
            self.in_flight = True
            t0 = time.monotonic()
            resp, err = None, None
            try:
                data = self._post(self.url, json.dumps(req.to_dict()).encode("utf-8"))
                resp = NavigateResponse.from_dict(data)
                self.failures = 0
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                err = f"HTTP {exc.code}: {detail}"
                # 4xx is our bug, not connectivity; don't flip to offline for it.
                if exc.code >= 500 or exc.code == 429:
                    self.failures += 1
            except Exception as exc:  # timeout, DNS, no network
                err = f"{type(exc).__name__}: {exc}"
                self.failures += 1
            finally:
                self.in_flight = False
            rtt = (time.monotonic() - t0) * 1000
            if err:
                log.warning("cloud request failed after %.0f ms: %s", rtt, err)
            self.on_result(CloudResult(req, resp, err, t0, rtt))
