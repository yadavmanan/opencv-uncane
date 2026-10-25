import numpy as np

from uncane.common.schemas import HazardItem, ThreatItem
from uncane.edge.arbiter import ReflexArbiter
from uncane.edge.frame_filter import KeyframeFilter, SharpFrameBuffer, encode_jpeg, ssim


def _img(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def test_ssim_identity_and_difference():
    a = _img(0)[:, :, 0]
    assert ssim(a, a) > 0.99
    assert ssim(a, _img(1)[:, :, 0]) < 0.3


def test_keyframe_filter_respects_interval_and_change():
    kf = KeyframeFilter(ssim_threshold=0.8, min_interval_s=1.0)
    a, b = _img(0), _img(1)
    assert kf.should_send(a, now=0.0)          # first frame always
    assert not kf.should_send(b, now=0.5)      # too soon
    assert not kf.should_send(a, now=2.0)      # same scene
    assert kf.should_send(b, now=2.1)          # changed scene


def test_sharp_buffer_prefers_sharp_frame():
    import cv2

    sharp = _img(3)
    blurry = cv2.GaussianBlur(sharp, (21, 21), 8)
    buf = SharpFrameBuffer(4)
    buf.push(blurry, 0.0)
    buf.push(sharp, 0.1)
    buf.push(blurry, 0.2)
    frame, t = buf.best()
    assert t == 0.1


def test_jpeg_encode_downscales():
    data = encode_jpeg(np.zeros((1080, 1920, 3), np.uint8), max_side=768)
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 768 and data[:2] == b"\xff\xd8"


def test_arbiter_debounces_and_releases():
    clock = [0.0]
    arb = ReflexArbiter(confirm_frames=2, window=3, release_frames=2, speech_cooldown_s=4, clock=lambda: clock[0])
    wall = HazardItem(zone="CENTER", level="TORSO", proximity=0.7, urgency="CRITICAL", distance_m=0.8)
    out = arb.update([wall], [])
    assert out.tone is None and not out.speech           # one frame is not enough
    out = arb.update([wall], [])
    assert out.tone is not None and out.tone.pan == 0.0
    assert out.speech and out.speech[0].startswith("Stop.")
    clock[0] = 1.0
    out = arb.update([wall], [])
    assert not out.speech                                  # cooldown
    arb.update([], [])
    out = arb.update([], [])
    assert out.tone is None                                # released


def test_arbiter_head_level_and_threat_speech():
    arb = ReflexArbiter(confirm_frames=1, window=1, clock=lambda: 0.0)
    branch = HazardItem(zone="CENTER", level="HEAD", proximity=0.6, urgency="CRITICAL", distance_m=1.2, label=None)
    bike = ThreatItem(bearing_deg=25, ttc_s=1.0, urgency="CRITICAL", zone="RIGHT", label="bicycle")
    out = arb.update([branch], [bike])
    assert any("Head height" in s for s in out.speech)
    assert any(s.startswith("Bicycle coming, right") for s in out.speech)
    assert out.tone.freq_hz == 1500.0 and out.tone.pan > 0
