"""Parse device-local commands before sending unmatched speech to the cloud."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LocalCommand:
    name: str
    value: str = ""


_RULES = (
    (re.compile(r"^(?:repeat|say that again|repeat that)$"), "repeat"),
    (re.compile(r"^(?:stop talking|be quiet|silence)$"), "stop_speech"),
    (re.compile(r"^(?:quiet mode|quiet)(?:\s+(on|off))?$"), "quiet"),
    (re.compile(r"^(?:indoor|indoors|outdoor|outdoors) mode$"), "environment"),
    (re.compile(r"^(?:speak|speech) (faster|slower)$"), "speech_rate"),
    (re.compile(r"^volume (up|down)$"), "speech_volume"),
    (re.compile(r"^(?:read|read this|read the text)(?:\s+(.+))?$"), "read"),
    (re.compile(r"^(?:show|read|list) notes$"), "show_notes"),
    (re.compile(r"^(?:clear|delete) notes$"), "clear_notes"),
    (re.compile(r"^(?:add|save|remember) note\s+(.+)$"), "add_note"),
    (re.compile(r"^(?:status|device status|how are you)$"), "status"),
    (re.compile(r"^(?:recalibrate|calibrate floor)$"), "recalibrate"),
    (re.compile(r"^(?:stop|cancel) guidance$"), "stop_guidance"),
    (re.compile(r"^(?:save|export) (?:event )?log$"), "export_blackbox"),
    (re.compile(r"^(?:feedback|that was wrong|good answer|bad answer)$"), "feedback"),
    (re.compile(r"^(?:where did you last see|where is my)\s+(.+?)[?.!]*$"), "find_object"),
)


def parse_command(text: str) -> Optional[LocalCommand]:
    normalized = re.sub(r"\s+", " ", text.strip().lower()).strip(" .,!?;:")
    if not normalized:
        return None
    for pattern, name in _RULES:
        match = pattern.fullmatch(normalized)
        if match:
            value = next((group for group in match.groups() if group), "")
            if name == "environment":
                value = "indoor" if normalized.startswith(("indoor", "indoors")) else "outdoor"
            elif name == "quiet" and not value:
                value = "toggle"
            return LocalCommand(name, value)
    return None
