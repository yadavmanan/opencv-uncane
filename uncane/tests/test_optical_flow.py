import cv2
import numpy as np

from uncane.edge.optical_flow import ThreatTracker

W, H = 640, 480


def textured(h, w, seed, blur=1.0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), blur)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


BG = textured(H + 200, W + 200, 0)
OBJ = textured(300, 300, 1, blur=3.0)   # smooth texture, like a real surface


def frame(bg_shift=0, obj_scale=None, obj_cx=W // 2):
    f = BG[100:100 + H, 100 + bg_shift:100 + bg_shift + W].copy()
    if obj_scale:
        size = int(110 * obj_scale)            # ~ a cyclist's width at 3-4 m
        patch = cv2.resize(OBJ, (size, size))
        y0, x0 = H // 2 - size // 2, obj_cx - size // 2
        f[y0:y0 + size, x0:x0 + size] = patch
    return f


def run(frames, fps=30.0):
    tr = ThreatTracker(hfov_deg=70)
    out = []
    for i, f in enumerate(frames):
        out.append(tr.update(f, i / fps))
    return out


def test_looming_object_is_detected_ahead():
    frames = [frame(obj_scale=1.03 ** i) for i in range(25)]   # ~0.9 /s expansion -> TTC ~1.1 s
    results = run(frames)
    hits = [t for step in results[5:] for t in step]
    assert hits, "expected an approaching-object threat"
    assert all(t.zone == "CENTER" for t in hits)
    assert min(t.ttc_s for t in hits) < 2.0
    assert any(t.urgency == "CRITICAL" for t in hits)


def test_camera_pan_is_not_a_threat():
    frames = [frame(bg_shift=4 * i) for i in range(25)]
    assert not any(run(frames))


def test_static_object_is_not_a_threat():
    frames = [frame(obj_scale=1.0) for _ in range(25)]
    assert not any(run(frames))


def test_object_on_right_reports_right_zone():
    frames = [frame(obj_scale=1.03 ** i, obj_cx=W - 130) for i in range(25)]
    hits = [t for step in run(frames)[5:] for t in step]
    assert hits and all(t.zone == "RIGHT" and t.bearing_deg > 10 for t in hits)


def test_walking_forward_whole_scene_expansion_is_not_a_threat():
    # Uniform zoom of the entire view (walking toward a far scene) is ego-motion.
    frames = []
    for i in range(25):
        s = 1.0 + 0.01 * i
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        frames.append(cv2.warpAffine(BG[100:100 + H, 100:100 + W], M, (W, H), borderMode=cv2.BORDER_REFLECT))
    assert not any(run(frames))
