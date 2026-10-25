"""Edge configuration: defaults + YAML file + ${ENV} expansion + --set overrides."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "device_id": "uncane-dev-01",
    "camera": {"source": 0, "width": 640, "height": 480, "hfov_deg": 70.0, "loop_video": True},
    "mount": {"height_m": 1.30, "pitch_deg": 12.0, "user_height_m": 1.75},
    "depth": {
        "model_path": "models/midas_v21_small_256.onnx",
        "output": "relative_inverse",
        "input_size": 128,
        "backend": "opencv",
        "opencv_target": "cpu",
        "input_name": None,
    },
    "hazards": {
        "analyzer": "geometric",
        "corridor_half_width_m": 0.40,
        "side_width_m": 1.50,
        "max_range_m": 3.0,
        "critical_m": 1.2,
        "critical_head_m": 1.6,
        "warning_m": 2.2,
        "side_warning_m": 1.2,
        "floor_tolerance_m": 0.15,
        "min_pixel_fraction": 0.004,
        "grid_rows": 6,
        "grid_cols": 8,
    },
    "flow": {
        "enabled": True, "proc_width": 320, "max_corners": 500, "redetect_every": 4,
        "ttc_warning_s": 3.0, "ttc_critical_s": 1.5, "min_cluster_points": 8, "persist_frames": 2,
    },
    "detector": {
        "enabled": True, "model_path": "models/object_detection_yolox_2022nov.onnx",
        "conf_threshold": 0.45, "nms_threshold": 0.5, "hz": 2.0, "input_size": 416,
    },
    "arbiter": {"confirm_frames": 2, "window": 3, "release_frames": 4, "speech_cooldown_s": 4.0},
    "keyframes": {
        "ssim_threshold": 0.80, "min_interval_s": 3.0, "buffer_frames": 8,
        "jpeg_quality": 80, "max_side_px": 768,
    },
    "audio": {"enabled": True, "volume": 0.35, "tts_rate": 190, "tts_voice": None, "haptics_serial_port": None},
    "voice": {
        "backend": "keyboard", "vosk_model_path": "models/vosk-model-small-en-us-0.15",
        "record_seconds": 4.0, "presets": {},
    },
    "cloud": {
        "enabled": True, "api_url": "", "api_key": "", "timeout_s": 8.0, "ambient_mode": False,
        "max_answer_age_s": 6.0, "offline_after_failures": 3,
        "static_location": None,
    },
    "debug": {"show_windows": True, "log_level": "INFO"},
}

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_env(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), node)
    return node


def _apply_override(cfg: Dict[str, Any], item: str) -> None:
    key, _, raw = item.partition("=")
    if not key or not _:
        raise ValueError(f"--set expects key=value, got {item!r}")
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: Optional[str] = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    for item in overrides:
        _apply_override(cfg, item)
    cfg = _expand_env(cfg)
    # Resolve model paths relative to the repo root so the app runs from anywhere.
    root = Path(path).resolve().parent.parent if path else Path.cwd()
    for section in ("depth", "detector"):
        p = Path(cfg[section]["model_path"])
        if not p.is_absolute() and not p.exists() and (root / p).exists():
            cfg[section]["model_path"] = str(root / p)
    src = cfg["camera"]["source"]
    if isinstance(src, str) and src.isdigit():
        cfg["camera"]["source"] = int(src)
    return cfg
