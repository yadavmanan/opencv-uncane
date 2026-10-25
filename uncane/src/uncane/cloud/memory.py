"""Multi-turn conversational memory in DynamoDB (text only, auto-expiring).

Table: PK `session_id` (S), SK `ts` (N), attribute `expires_at` is the TTL.
Only question/answer text is stored - never images - and items expire after
MEMORY_TTL_HOURS (default 12).
"""
from __future__ import annotations

import logging
import os
import time
from decimal import Decimal
from typing import Any, Dict, List

log = logging.getLogger(__name__)


class SessionMemory:
    def __init__(self, table_name: str = "", max_turns: int = 4, ttl_hours: float = 12.0):
        self.table_name = table_name or os.environ.get("SESSIONS_TABLE", "")
        self.max_turns = int(os.environ.get("MEMORY_TURNS", max_turns))
        self.ttl_s = int(float(os.environ.get("MEMORY_TTL_HOURS", ttl_hours)) * 3600)
        self._table = None
        if self.table_name:
            import boto3

            self._table = boto3.resource("dynamodb").Table(self.table_name)

    @property
    def enabled(self) -> bool:
        return self._table is not None and self.max_turns > 0

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        """Previous turns as alternating user/assistant text messages (oldest first)."""
        if not self.enabled:
            return []
        from boto3.dynamodb.conditions import Key

        try:
            res = self._table.query(KeyConditionExpression=Key("session_id").eq(session_id),
                                    ScanIndexForward=False, Limit=self.max_turns)
        except Exception as exc:  # memory is a nice-to-have; never fail the request
            log.warning("memory load failed: %s", exc)
            return []
        messages: List[Dict[str, Any]] = []
        for item in reversed(res.get("Items", [])):
            messages.append({"role": "user", "content": f"(Earlier) {item['question']}"})
            messages.append({"role": "assistant", "content": item["answer"]})
        return messages

    def save(self, session_id: str, question: str, answer: str) -> None:
        if not self.enabled or not question or not answer:
            return
        now = time.time()
        try:
            self._table.put_item(Item={
                "session_id": session_id, "ts": Decimal(str(round(now, 3))),
                "question": question[:500], "answer": answer[:1000],
                "expires_at": int(now + self.ttl_s),
            })
        except Exception as exc:
            log.warning("memory save failed: %s", exc)
