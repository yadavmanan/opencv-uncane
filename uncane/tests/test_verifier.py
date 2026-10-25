from uncane.common.schemas import HazardItem, ModelAnswer, TargetInfo, ThreatItem
from uncane.common.verifier import SafetyVerifier, claims_clear_path, claims_safe_crossing

V = SafetyVerifier()
WALL = HazardItem(zone="CENTER", level="TORSO", proximity=0.7, urgency="CRITICAL", distance_m=0.9)
LEFT_POLE = HazardItem(zone="LEFT", level="HEAD", proximity=0.8, urgency="CRITICAL", distance_m=0.6)


def test_clear_path_phrases():
    assert claims_clear_path("The path is clear, go ahead.")
    assert claims_clear_path("There are no obstacles in front of you.")
    assert claims_clear_path("You can walk straight ahead.")
    assert claims_clear_path("It's safe to walk forward")
    assert not claims_clear_path("The path is not clear.")
    assert not claims_clear_path("The door is to your left.")


def test_overrides_clear_claim_when_edge_sees_wall():
    r = V.verify(ModelAnswer(speech="The path ahead is clear."), [WALL])
    assert r.overridden
    assert r.speech.startswith("Stop.")
    assert "0.9 meters" in r.speech
    assert "clear_path_contradicts_edge_depth" in r.reasons


def test_structured_flag_alone_triggers_override():
    r = V.verify(ModelAnswer(speech="Go on.", claims_path_clear=True), [WALL])
    assert r.overridden and r.speech.startswith("Stop.")


def test_clear_claim_allowed_without_critical_hazard():
    warn = HazardItem(zone="CENTER", level="TORSO", proximity=0.3, urgency="WARNING", distance_m=2.0)
    r = V.verify(ModelAnswer(speech="The hallway looks clear."), [warn])
    assert not r.overridden
    assert r.speech == "The hallway looks clear."


def test_steering_into_critical_zone_is_prefixed():
    r = V.verify(ModelAnswer(speech="The door is on your left.", suggested_direction="left"), [LEFT_POLE])
    assert r.overridden
    assert r.speech.startswith("Careful:")
    assert "head height" in r.speech


def test_target_behind_obstacle():
    target = TargetInfo(label="bottle", bearing_deg=2.0, distance_m=2.5)
    r = V.verify(ModelAnswer(speech="The bottle is ahead on the table."), [WALL], target=target)
    assert r.overridden and "before you reach it" in r.speech


def test_never_certifies_crossing():
    assert claims_safe_crossing("It's safe to cross now.")
    r = V.verify(ModelAnswer(speech="The walk signal is on. It's safe to cross."), [])
    assert r.overridden
    assert "safe to cross" not in r.speech.lower()
    assert "Listen for traffic" in r.speech


def test_threat_prefix():
    threat = ThreatItem(bearing_deg=30.0, ttc_s=1.2, urgency="CRITICAL", zone="RIGHT", label="bicycle")
    r = V.verify(ModelAnswer(speech="The shop is ahead."), [], [threat])
    assert r.speech.startswith("Caution, bicycle approaching from your right.")
    assert "approaching_threat" in r.reasons


def test_benign_answer_untouched():
    r = V.verify(ModelAnswer(speech="The sign says Exit."), [])
    assert not r.overridden and r.speech == "The sign says Exit."


def test_second_pass_on_device_is_idempotent():
    target = TargetInfo(label="bottle", bearing_deg=2.0, distance_m=2.5)
    threat = ThreatItem(bearing_deg=-30.0, ttc_s=1.0, urgency="CRITICAL", zone="LEFT")
    first = V.verify(ModelAnswer(speech="The bottle is ahead on the table."), [WALL], [threat], target)
    second = V.verify(ModelAnswer(speech=first.speech), [WALL], [threat], target)
    assert second.speech == first.speech
    assert first.speech.count("Careful:") == 1 and first.speech.count("Caution,") == 1
