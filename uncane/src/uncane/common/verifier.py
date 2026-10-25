"""Anti-hallucination safety verifier.

The VLM is never allowed to contradict the deterministic edge sensors. Every
cloud answer passes through ``SafetyVerifier.verify`` twice: once in Lambda
(against the hazards captured with the keyframe) and again on the device
right before it is spoken (against the *current* hazards, because the user
kept walking during the ~1 s round trip).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .schemas import HazardItem, ModelAnswer, TargetInfo, ThreatItem
from .spatial import zone_from_bearing

# Claims that the way forward is free. Checked with word boundaries on
# lower-cased text; negated forms ("path is not clear") don't match.
_CLEAR_PATH_PATTERNS = [
    r"\b(the )?(path|way|route|sidewalk|walkway|hallway|road) (ahead |in front( of you)? )?(is|looks|seems) (completely |totally |fully )?(clear|free|open|empty|unobstructed)\b",
    r"\bclear (path|way|route)\b",
    r"\bno (obstacles?|obstructions?|hazards?|barriers?)\b",
    r"\bnothing (is )?(in front of you|in your (way|path)|ahead|blocking)\b",
    r"\b(it is |it's )?safe to (walk|proceed|go|continue|move) (forward|ahead|straight)\b",
    r"\byou can (walk|go|move|continue) (straight|forward|ahead)\b",
    r"\bkeep (walking|going) straight\b",
]
_CROSSING_PATTERNS = [
    r"\b(it is |it's )?safe to cross\b",
    r"\byou (can|may) (now )?cross\b",
    r"\bgo ahead and cross\b",
]
_CLEAR_RE = re.compile("|".join(_CLEAR_PATH_PATTERNS))
_CROSS_RE = re.compile("|".join(_CROSSING_PATTERNS))

_DIRECTION_TO_ZONE = {
    "left": "LEFT", "slightly_left": "LEFT", "ahead": "CENTER",
    "slightly_right": "RIGHT", "right": "RIGHT",
}
CROSSING_DISCLAIMER = "Listen for traffic and confirm before you cross."


@dataclass
class VerificationResult:
    speech: str
    overridden: bool = False
    reasons: List[str] = field(default_factory=list)


def claims_clear_path(text: str) -> bool:
    return bool(_CLEAR_RE.search(text.lower()))


def claims_safe_crossing(text: str) -> bool:
    return bool(_CROSS_RE.search(text.lower()))


def _critical(hazards: Sequence[HazardItem], zone: Optional[str] = None) -> List[HazardItem]:
    out = [h for h in hazards if h.urgency == "CRITICAL" and (zone is None or h.zone == zone)]
    return sorted(out, key=lambda h: (h.distance_m if h.distance_m is not None else 1.0 - h.proximity))


def _threat_prefix(threats: Sequence[ThreatItem], ttc_limit_s: float) -> Optional[str]:
    urgent = [t for t in threats if t.ttc_s <= ttc_limit_s and t.urgency in ("WARNING", "CRITICAL")]
    if not urgent:
        return None
    t = min(urgent, key=lambda t: t.ttc_s)
    side = {"LEFT": "from your left", "RIGHT": "from your right", "CENTER": "from ahead"}[t.zone]
    what = t.label or "something"
    return f"Caution, {what} approaching {side}."


class SafetyVerifier:
    """Deterministic rules; order matters (most severe first)."""

    def __init__(self, threat_ttc_limit_s: float = 3.0):
        self.threat_ttc_limit_s = threat_ttc_limit_s

    def verify(
        self,
        answer: ModelAnswer,
        hazards: Sequence[HazardItem],
        threats: Sequence[ThreatItem] = (),
        target: Optional[TargetInfo] = None,
    ) -> VerificationResult:
        text = answer.speech.strip() or "I could not understand the scene."
        reasons: List[str] = []
        center_critical = _critical(hazards, "CENTER")

        # Rule 1: the model says "clear" but the edge sees a critical obstacle ahead.
        if center_critical and (answer.claims_path_clear or claims_clear_path(text)):
            reasons.append("clear_path_contradicts_edge_depth")
            text = f"Stop. {center_critical[0].describe().capitalize()}."
            return VerificationResult(self._prefix_threat(text, threats), True, reasons)

        # Rules 2-3 prefix a warning; skip if an earlier pass (Lambda) already did, so
        # the device's second verification is idempotent.
        already_warned = text.startswith("Careful:") or text.startswith("Caution,")

        # Rule 2: the model steers the user into a zone the edge flags as critical.
        steer_zone = _DIRECTION_TO_ZONE.get(answer.suggested_direction)
        if already_warned:
            pass
        elif steer_zone and _critical(hazards, steer_zone):
            h = _critical(hazards, steer_zone)[0]
            reasons.append(f"direction_into_critical_{steer_zone.lower()}")
            text = f"Careful: {h.describe()}. " + text

        # Rule 3: the target lies behind a critical obstacle in the same direction.
        elif target is not None and target.bearing_deg is not None:
            tz = zone_from_bearing(target.bearing_deg)
            blocking = [h for h in _critical(hazards, tz)
                        if target.distance_m is None or h.distance_m is None or h.distance_m < target.distance_m]
            if blocking:
                reasons.append(f"target_path_blocked_{tz.lower()}")
                text = f"Careful: {blocking[0].describe()} before you reach it. " + text

        # Rule 4: never certify a street crossing; the VLM may only describe the signal.
        if claims_safe_crossing(text):
            reasons.append("crossing_safety_claim")
            text = _CROSS_RE.sub("the signal appears to allow crossing", text.lower(), count=1)
            text = text[0].upper() + text[1:]
            if CROSSING_DISCLAIMER not in text:
                text = text.rstrip(". ") + ". " + CROSSING_DISCLAIMER

        text = self._prefix_threat(text, threats)
        if text.startswith("Caution,"):
            reasons.append("approaching_threat")
        return VerificationResult(text, bool(reasons), reasons)

    def _prefix_threat(self, text: str, threats: Sequence[ThreatItem]) -> str:
        prefix = _threat_prefix(threats, self.threat_ttc_limit_s)
        return f"{prefix} {text}" if prefix and not text.startswith("Caution,") else text
