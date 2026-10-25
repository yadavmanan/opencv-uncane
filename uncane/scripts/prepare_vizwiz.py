"""Phase 2 data pipeline: VizWiz-VQA (photos taken by blind users + their questions).

    python scripts/prepare_vizwiz.py                       # annotations only (1.7 MB)
    python scripts/prepare_vizwiz.py --images val          # + 3.5 GB of validation images
    python scripts/prepare_vizwiz.py --images train --sft  # + SFT JSONL for fine-tuning
    python scripts/prepare_vizwiz.py --images val --upload s3://<DataBucket>/datasets/vizwiz/

Outputs (data/vizwiz/):
  vizwiz_<split>.jsonl      one record per question: image, question, answers, answerable, nav_relevant
  sft_<split>.jsonl         chat-format targets matching the cloud JSON contract (with --sft)

VizWiz is CC BY 4.0: https://vizwiz.org/tasks-and-datasets/vqa/
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vizwiz"
BASE = "https://vizwiz.cs.colorado.edu/VizWiz_final"
ANN_URL = f"{BASE}/vqa_data/Annotations.zip"
IMG_URL = BASE + "/images/{split}.zip"

NAV_KEYWORDS = re.compile(
    r"\b(door|sign|street|road|cross|bus|train|stop|exit|stairs?|elevator|where|which way|direction|"
    r"left|right|front|ahead|room|building|store|shop|read|say|number|color|colour|light|walk|"
    r"hallway|entrance|path|car|what is this|what's this)\b", re.I)

UNANSWERABLE_SPEECH = ("I can't tell from this photo. Try holding the camera a little further away "
                       "and pointing it straight at the object.")


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[ok] {dest.name} present")
        return dest
    print(f"[get] {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def majority_answer(answers):
    counts = collections.Counter(a["answer"].strip().lower() for a in answers if a.get("answer"))
    return counts.most_common(1)[0][0] if counts else ""


def to_sft(rec: dict) -> dict:
    """Chat example whose target is the same JSON the SageMaker backend parses."""
    if rec["answerable"] and rec["answer_type"] != "unanswerable":
        ans = majority_answer(rec["answers"])
        speech = ans[0].upper() + ans[1:] + "." if ans else UNANSWERABLE_SPEECH
        target = {"speech": speech, "needs_better_view": not ans, "claims_path_clear": False}
    else:
        target = {"speech": UNANSWERABLE_SPEECH, "needs_better_view": True, "claims_path_clear": False}
    return {"image": rec["image"], "messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rec["question"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", choices=["none", "val", "train", "both"], default="none")
    ap.add_argument("--sft", action="store_true", help="also write SFT chat JSONL")
    ap.add_argument("--upload", help="s3://bucket/prefix/ to aws s3 sync the output to")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ann_zip = download(ANN_URL, OUT / "Annotations.zip")
    with zipfile.ZipFile(ann_zip) as zf:
        zf.extractall(OUT / "annotations")

    splits = {"none": [], "val": ["val"], "train": ["train"], "both": ["train", "val"]}[args.images]
    for split in splits:
        z = download(IMG_URL.format(split=split), OUT / f"{split}.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(OUT / "images")

    for split in ("train", "val"):
        ann_path = next((OUT / "annotations").rglob(f"{split}.json"), None)
        if ann_path is None:
            continue
        items = json.loads(ann_path.read_text(encoding="utf-8"))
        recs = [{
            "image": f"images/{split}/{it['image']}", "question": it["question"],
            "answers": it.get("answers", []), "answer_type": it.get("answer_type", ""),
            "answerable": int(it.get("answerable", 1)),
            "nav_relevant": bool(NAV_KEYWORDS.search(it["question"])),
        } for it in items]
        with open(OUT / f"vizwiz_{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        nav = sum(r["nav_relevant"] for r in recs)
        unans = sum(1 - r["answerable"] for r in recs)
        print(f"[ok] {split}: {len(recs)} questions, {nav} navigation-relevant, {unans} unanswerable")
        if args.sft:
            with open(OUT / f"sft_{split}.jsonl", "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(to_sft(r)) + "\n")
            print(f"[ok] wrote sft_{split}.jsonl")

    if args.upload:
        cmd = ["aws", "s3", "sync", str(OUT), args.upload, "--exclude", "*.zip"]
        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
