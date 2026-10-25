"""Telemetry to S3 and metrics to CloudWatch (Embedded Metric Format).

S3 layout (partitioned for Athena):
  s3://<bucket>/telemetry/dt=YYYY-MM-DD/<device>/<request_id>.json
  s3://<bucket>/keyframes/dt=YYYY-MM-DD/<device>/<request_id>.jpg   (only if SAVE_KEYFRAMES=true)

Keyframes are photos of the user's surroundings; keep them off unless you have
consent and need them for evaluation/fine-tuning.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Dict

log = logging.getLogger(__name__)
NAMESPACE = "UnCane"


class Telemetry:
    def __init__(self, bucket: str = ""):
        self.bucket = bucket or os.environ.get("TELEMETRY_BUCKET", "")
        self.save_keyframes = os.environ.get("SAVE_KEYFRAMES", "false").lower() == "true"
        self._s3 = None
        if self.bucket:
            import boto3

            self._s3 = boto3.client("s3")

    def record(self, request_id: str, device_id: str, record: Dict[str, Any], image_b64: str = "") -> None:
        if self._s3 is None:
            return
        day = time.strftime("%Y-%m-%d", time.gmtime())
        prefix = f"dt={day}/{device_id}/{request_id}"
        try:
            self._s3.put_object(Bucket=self.bucket, Key=f"telemetry/{prefix}.json",
                                Body=json.dumps(record).encode("utf-8"), ContentType="application/json")
            if self.save_keyframes and image_b64:
                self._s3.put_object(Bucket=self.bucket, Key=f"keyframes/{prefix}.jpg",
                                    Body=base64.b64decode(image_b64), ContentType="image/jpeg")
        except Exception as exc:
            log.warning("telemetry upload failed: %s", exc)


def emit_metrics(dimensions: Dict[str, str], metrics: Dict[str, float], units: Dict[str, str]) -> None:
    """Print one CloudWatch EMF line; Lambda's log agent turns it into metrics."""
    doc: Dict[str, Any] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": NAMESPACE,
                "Dimensions": [list(dimensions)],
                "Metrics": [{"Name": k, "Unit": units.get(k, "None")} for k in metrics],
            }],
        },
        **dimensions,
        **metrics,
    }
    print(json.dumps(doc))
