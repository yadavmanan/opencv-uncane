"""Detector accuracy (mAP) on the navigation COCO subset. Needs `pip install pycocotools`.

    python scripts/prepare_coco_subset.py --max-images 500
    python eval/eval_detector_coco.py --input-size 416
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uncane.edge.detector import YoloXDetector  # noqa: E402

DATA = ROOT / "data" / "coco_subset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "object_detection_yolox_2022nov.onnx"))
    ap.add_argument("--input-size", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.05, help="low threshold for mAP; the app uses ~0.45")
    args = ap.parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(DATA / "instances_nav_subset.json"))
    name_to_cat = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}
    det = YoloXDetector(args.model, conf_threshold=args.conf, input_size=args.input_size)
    results, times = [], []
    for img_id in gt.getImgIds():
        info = gt.loadImgs(img_id)[0]
        img = cv2.imread(str(DATA / "images" / info["file_name"]))
        t0 = time.perf_counter()
        dets = det.detect(img)
        times.append((time.perf_counter() - t0) * 1000)
        w, h = info["width"], info["height"]
        for d in dets:
            x0, y0, x1, y1 = d.box
            results.append({"image_id": img_id, "category_id": name_to_cat[d.label],
                            "bbox": [x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h], "score": d.score})
    out = ROOT / "eval" / f"yolox_{args.input_size}_detections.json"
    out.write_text(json.dumps(results))
    ev = COCOeval(gt, gt.loadRes(str(out)), "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    print(f"\nlatency p50 {np.median(times):.0f} ms at {args.input_size}px; mAP@[.5:.95] = {ev.stats[0]:.3f}, "
          f"mAP@.5 = {ev.stats[1]:.3f}")


if __name__ == "__main__":
    main()
