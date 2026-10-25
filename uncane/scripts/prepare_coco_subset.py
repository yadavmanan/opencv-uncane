"""Phase 2 data pipeline: a navigation-focused subset of MS COCO val2017.

    python scripts/prepare_coco_subset.py --max-images 500

Downloads COCO 2017 annotations (253 MB, once), keeps images that contain
street/indoor obstacle classes, fetches just those images, and writes a
COCO-format annotation file for eval/eval_detector_coco.py.

COCO annotations are CC BY 4.0; images carry their Flickr licenses.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "coco_subset"
ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"

NAV_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "dog", "chair", "couch", "potted plant", "dining table", "suitcase", "bottle",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)

    zpath = OUT / "annotations_trainval2017.zip"
    ann_file = OUT / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        if not zpath.exists():
            print(f"[get] {ANN_ZIP} (253 MB)")
            urllib.request.urlretrieve(ANN_ZIP, zpath)
        with zipfile.ZipFile(zpath) as zf:
            zf.extract("annotations/instances_val2017.json", OUT)

    coco = json.loads(ann_file.read_text())
    cat_ids = {c["id"] for c in coco["categories"] if c["name"] in NAV_CLASSES}
    img_ids = sorted({a["image_id"] for a in coco["annotations"] if a["category_id"] in cat_ids})
    random.Random(args.seed).shuffle(img_ids)
    keep = set(img_ids[: args.max_images])

    images = [im for im in coco["images"] if im["id"] in keep]
    subset = {
        "info": coco.get("info", {}), "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
        "categories": coco["categories"],
    }
    (OUT / "instances_nav_subset.json").write_text(json.dumps(subset))
    for i, im in enumerate(images, 1):
        dest = OUT / "images" / im["file_name"]
        if not dest.exists():
            urllib.request.urlretrieve(IMG_URL.format(im["file_name"]), dest)
        if i % 50 == 0:
            print(f"  {i}/{len(images)} images")
    print(f"[ok] {len(images)} images, {len(subset['annotations'])} boxes -> {OUT}")


if __name__ == "__main__":
    main()
