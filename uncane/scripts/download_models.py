"""Download the edge models into ./models.

    python scripts/download_models.py                 # required models (depth + detector)
    python scripts/download_models.py --all           # + Depth-Anything-V2 small + Vosk speech model
    python scripts/download_models.py --only vosk
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

REGISTRY = {
    "midas": {
        "file": "midas_v21_small_256.onnx",
        "urls": ["https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"],
        "size": 66_764_249, "required": True,
        "note": "MiDaS v2.1 small - relative inverse depth (MIT license)",
    },
    "yolox": {
        "file": "object_detection_yolox_2022nov.onnx",
        "urls": [
            "https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx",
        ],
        "size": 35_858_002, "required": True,
        "note": "YOLOX-S, COCO-80 classes, OpenCV Model Zoo (Apache-2.0)",
    },
    "depth_anything": {
        "file": "depth_anything_v2_small.onnx",
        "urls": ["https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"],
        "size": 99_060_839, "required": False,
        "note": "Depth-Anything-V2 small - sharper relative depth, slower on CPU (Apache-2.0)",
    },
    "vosk": {
        "file": "vosk-model-small-en-us-0.15.zip",
        "urls": ["https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"],
        "size": 41_205_931, "required": False, "unzip": True,
        "note": "Vosk small English offline speech recognition (Apache-2.0)",
    },
}


def _progress(name):
    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r  {name}: {pct:3d}%")
            sys.stdout.flush()
    return hook


def fetch(key: str) -> None:
    spec = REGISTRY[key]
    dest = MODELS / spec["file"]
    if dest.exists() and dest.stat().st_size == spec["size"]:
        print(f"[ok]   {spec['file']} already present")
    else:
        for url in spec["urls"]:
            try:
                print(f"[get]  {spec['note']}\n       {url}")
                tmp = dest.with_suffix(dest.suffix + ".part")
                urllib.request.urlretrieve(url, tmp, _progress(spec["file"]))
                print()
                if tmp.stat().st_size != spec["size"]:
                    raise IOError(f"size mismatch: got {tmp.stat().st_size}, expected {spec['size']}")
                tmp.replace(dest)
                break
            except Exception as exc:
                print(f"\n[warn] {exc}; trying next mirror")
        else:
            raise SystemExit(f"could not download {spec['file']}")
    if spec.get("unzip"):
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(MODELS)
        print(f"[ok]   extracted {dest.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", choices=sorted(REGISTRY))
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    keys = [args.only] if args.only else [k for k, v in REGISTRY.items() if v["required"] or args.all]
    for k in keys:
        fetch(k)


if __name__ == "__main__":
    main()
