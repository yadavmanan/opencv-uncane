import json

import numpy as np
import pytest

from uncane.common.schemas import HazardItem
from uncane.edge.blackbox import BlackBox
from uncane.edge.commands import parse_command
from uncane.edge.detector import Detection
from uncane.edge.guidance import GuidanceTracker
from uncane.edge.monitors import FrameMonitor
from uncane.edge.object_memory import ObjectMemory
from uncane.edge.user_profile import UserProfile
from uncane.edge.arbiter import ReflexArbiter


def test_local_command_parser_separates_controls_from_queries():
    assert parse_command("quiet mode on").name == "quiet"
    assert parse_command("where did you last see my phone?").value == "my phone"
    assert parse_command("what does the sign say") is None


def test_profile_persists_preferences_and_notes(tmp_path):
    path = tmp_path / "profile.json"
    profile = UserProfile(str(path))
    profile.update(environment="indoor", quiet_mode=True)
    profile.add_note("Front door code is in my wallet")
    loaded = UserProfile(str(path))
    assert loaded.data["environment"] == "indoor"
    assert loaded.data["quiet_mode"] is True
    assert loaded.data["notes"] == ["Front door code is in my wallet"]


def test_profile_corruption_falls_back_to_defaults(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{broken", encoding="utf-8")
    assert UserProfile(str(path)).data["notes"] == []


def test_object_memory_tracks_alias_and_depth():
    memory = ObjectMemory(retention_s=10, clock=lambda: 1.0)
    memory.update([Detection("cell phone", 0.9, [0.4, 0.3, 0.6, 0.7])], 70.0,
                  np.full((8, 8), 2.4, dtype=np.float32), now=1.0)
    sighting = memory.find("phone", now=2.0)
    assert sighting is not None
    assert sighting.distance_m == pytest.approx(2.4)


def test_guidance_uses_detector_match():
    tracker = GuidanceTracker(clock=lambda: 1.0)
    tracker.start("water bottle", [0.2, 0.2, 0.4, 0.5])
    cue = tracker.update(np.zeros((32, 32, 3), dtype=np.uint8),
                         [Detection("bottle", 0.9, [0.55, 0.4, 0.75, 0.8])],
                         70.0)
    assert cue is not None
    assert cue.state == "tracking"
    assert cue.bearing_deg > 0


def test_monitor_rate_limits_camera_warnings():
    clock = [0.0]
    monitor = FrameMonitor(cooldown_s=10, clock=lambda: clock[0])
    dark = np.zeros((32, 32, 3), dtype=np.uint8)
    assert monitor.inspect(dark, calibrated=False)
    assert monitor.inspect(dark, calibrated=False) == []
    clock[0] = 11.0
    assert monitor.inspect(dark, calibrated=False)


def test_blackbox_exports_jsonl_without_snapshots_by_default(tmp_path):
    path = tmp_path / "events.jsonl"
    box = BlackBox(str(path), capacity=2)
    box.record("feedback", {"count": 1}, export=True)
    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["event"] == "feedback"
    assert not list(tmp_path.glob("*.jpg"))


def test_arbiter_quiet_crowd_and_dropoff_rules():
    arbiter = ReflexArbiter(confirm_frames=1, window=1, quiet_mode=True, crowd_mode=True,
                            crowd_alert_distance_m=1.0, clock=lambda: 1.0)
    distant_person = HazardItem("CENTER", "TORSO", 0.4, "WARNING", 2.0, "person")
    assert arbiter.update([distant_person], []).tone is None
    close_person = HazardItem("CENTER", "TORSO", 0.8, "CRITICAL", 0.5, "person")
    assert arbiter.update([close_person], []).tone is not None
    dropoff = HazardItem("CENTER", "GROUND", 0.8, "CRITICAL", 0.3, "drop-off")
    result = arbiter.update([dropoff], [])
    assert result.tone.kind == "dropoff"
    assert any("Drop-off" in message for message in result.speech)


def test_dropoff_needs_longer_confirmation_than_obstacles():
    arbiter = ReflexArbiter(confirm_frames=2, window=3, dropoff_confirm_frames=4, dropoff_window=5,
                            clock=lambda: 1.0)
    dropoff = HazardItem("CENTER", "GROUND", 0.8, "CRITICAL", 1.0, "drop-off")
    for _ in range(3):
        assert arbiter.update([dropoff], []).tone is None
    assert arbiter.update([dropoff], []).tone.kind == "dropoff"
