"""Egocentric geometry helpers: image coordinates -> bearings, clock positions, steps."""
from __future__ import annotations

import math
from typing import List, Optional

from .schemas import DepthGrid, TargetInfo

STEP_LENGTH_M = 0.7


def bearing_from_x(x_norm: float, hfov_deg: float) -> float:
    """Horizontal angle (deg) of a normalized image x (0..1) for a pinhole camera.

    Negative = left of the optical axis, positive = right.
    """
    half = math.radians(hfov_deg) / 2.0
    return math.degrees(math.atan((x_norm - 0.5) * 2.0 * math.tan(half)))


def clock_from_bearing(bearing_deg: float) -> int:
    """Map a bearing to a clock face (12 = ahead, 3 = right, 9 = left)."""
    hour = round(bearing_deg / 30.0) % 12
    return 12 if hour == 0 else int(hour)


def zone_from_bearing(bearing_deg: float, center_half_width_deg: float = 12.0) -> str:
    if bearing_deg < -center_half_width_deg:
        return "LEFT"
    if bearing_deg > center_half_width_deg:
        return "RIGHT"
    return "CENTER"


def turn_phrase(bearing_deg: float) -> str:
    """Human turn instruction, rounded to 15 degrees: 'Turn 30 degrees right'."""
    rounded = int(round(abs(bearing_deg) / 15.0) * 15)
    if rounded == 0:
        return "It is straight ahead"
    side = "right" if bearing_deg > 0 else "left"
    return f"Turn {rounded} degrees {side}"


def steps_phrase(steps: int) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    if steps <= 0:
        return "within arm's reach"
    n = words[steps] if steps < len(words) else str(steps)
    return f"walk {n} step" + ("" if steps == 1 else "s")


def locate_target(
    label: str,
    bbox_1000: Optional[List[float]],
    hfov_deg: float,
    depth_grid: Optional[DepthGrid],
) -> Optional[TargetInfo]:
    """Turn the VLM's 0..1000 bounding box into a grounded egocentric target.

    The bearing comes from camera geometry and the distance from the edge depth
    grid, not from the language model, so both are deterministic.
    """
    if not bbox_1000:
        return None
    x0, y0, x1, y1 = [max(0.0, min(1000.0, v)) / 1000.0 for v in bbox_1000]
    if x1 <= x0 or y1 <= y0:
        return None
    cx = (x0 + x1) / 2.0
    bearing = bearing_from_x(cx, hfov_deg)
    distance = steps = None
    if depth_grid is not None and depth_grid.kind == "metric_m":
        distance = depth_grid.sample(x0, y0, x1, y1)
        if distance is not None:
            steps = max(0, int(round(distance / STEP_LENGTH_M)))
    return TargetInfo(
        label=label,
        bearing_deg=round(bearing, 1),
        clock=clock_from_bearing(bearing),
        distance_m=round(distance, 2) if distance is not None else None,
        steps=steps,
        bbox=[round(v, 3) for v in (x0, y0, x1, y1)],
    )


def guidance_sentence(target: TargetInfo) -> str:
    """'Turn 30 degrees right, walk three steps.'"""
    if target.bearing_deg is None:
        return ""
    parts = [turn_phrase(target.bearing_deg)]
    if target.steps is not None:
        parts.append(steps_phrase(target.steps))
    return ", ".join(parts) + "."
