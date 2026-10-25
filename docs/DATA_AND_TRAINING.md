# Data engine, evaluation and fine-tuning

The brief's data strategy: **pre-train on MS COCO** for robust object recognition, then
**align with VizWiz**, the real photos and questions of blind users, for the messy,
unoptimized reality of a wearable. This is how each dataset is used here.

| Dataset | Used for | Script |
|---|---|---|
| MS COCO 2017 | The on-device detector (YOLOX-S) is COCO-trained. A navigation subset measures its mAP. | `scripts/prepare_coco_subset.py`, `eval/eval_detector_coco.py` |
| VizWiz-VQA | Evaluating the cloud VLM on real blind-user photos: accuracy, and whether it correctly says "I can't tell, re-aim the camera". Optional LoRA fine-tune of an open VLM. | `scripts/prepare_vizwiz.py`, `eval/eval_vizwiz_vlm.py`, `training/sagemaker/` |

## 1. COCO navigation subset â†’ detector mAP

```bash
pip install pycocotools
python scripts/prepare_coco_subset.py --max-images 500     # 253 MB annotations once + images
python eval/eval_detector_coco.py --input-size 416
```

The subset is val2017 images containing at least one of 18 navigation classes: person,
bicycle, car, motorcycle, bus, truck, traffic light, fire hydrant, stop sign, parking meter,
bench, dog, chair, couch, potted plant, dining table, suitcase, bottle. Evaluation uses
standard COCO mAP over all classes present in those images.

**Measured** (60-image subset, 570 boxes, laptop CPU, input 416 px):

| Metric | Value |
|---|---|
| mAP@[.5:.95] | **0.349** |
| mAP@.5 | 0.467 |
| mAP small / medium / large | 0.152 / 0.420 / 0.576 |
| Latency p50 | 96 ms |

This is a small sample; rerun with `--max-images 500` for a stable number, and compare
`--input-size 640` (more accurate, ~340 ms) and `320` (faster). Small-object AP is low, as
expected at 416 px. That is acceptable because the detector only *names* hazards; the depth
pipeline finds them.

## 2. VizWiz-VQA

```bash
python scripts/prepare_vizwiz.py                      # annotations only (1.7 MB)
python scripts/prepare_vizwiz.py --images val         # + 3.5 GB of validation photos
python scripts/prepare_vizwiz.py --images train --sft # + 11.3 GB of training photos and SFT JSONL
```

What the annotations contain (measured):

| Split | Questions | Navigation-relevant* | Unanswerable |
|---|---|---|---|
| train | 20 523 | 9 302 | 5 532 |
| val | 4 319 | 1 910 | 1 385 |

\* keyword filter in `prepare_vizwiz.py`: door, sign, street, bus, read, which way, color, â€¦

About 30 % of VizWiz photos can't be answered because they are blurry, dark or pointed away.
That is exactly why the edge sends the *sharpest* recent frame, and why the model has a
`needs_better_view` flag and is told to say how to re-aim the camera.

### 2.1 Evaluating the cloud VLM

```bash
python eval/eval_vizwiz_vlm.py --limit 100                 # through the deployed API
python eval/eval_vizwiz_vlm.py --limit 100 --direct        # straight to Bedrock with your AWS creds
python eval/eval_vizwiz_vlm.py --limit 100 --nav-only
```

Each sample is one paid model call, so start small. Metrics:

- **Accuracy (lenient VQA).** For answerable questions, `min(1, #annotators whose answer
  appears in our spoken sentence / 3)`, after lower-casing and removing punctuation and
  articles. The official VizWiz metric compares short answers exactly; we speak full
  sentences, so containment is used and the number is not directly comparable to the
  leaderboard.
- **Unanswerable precision / recall.** How well `needs_better_view` matches VizWiz's
  "unanswerable" label. High precision means few needless "please re-aim" prompts (the brief's
  false-positive constraint); high recall means few confident answers on unusable photos.
- **Latency p50 / p90**, which checks the < 1.5 s target.

Results go to `eval/vizwiz_results_<time>.json`, with every row, for error analysis.
*Not yet run in this build*, because it needs Bedrock access and costs money.

A note on the brief's "mAP on the VizWiz validation set": VizWiz-VQA has no bounding boxes,
so mAP isn't defined for it. We report mAP on COCO (Â§1) and VQA accuracy plus answerability
on VizWiz (Â§2.1). VizWiz also publishes "answer grounding" and "few-shot recognition"
tracks with masks and boxes, if you want box-level metrics on blind-user photos.

### 2.2 Prompt engineering loop (Phase 4)

1. Run `eval_vizwiz_vlm.py --limit 100 --seed 0` for a baseline.
2. Edit `SYSTEM_PROMPT` in [src/uncane/common/prompts.py](../uncane/src/uncane/common/prompts.py).
3. Rerun with the **same seed**. Keep the change only if accuracy and unanswerable precision
   don't drop.
4. Before you trust the result, check it once on a fresh seed, so you aren't overfitting to
   the 100 samples.

### 2.3 Fine-tuning (optional): Qwen2-VL-2B + LoRA on SageMaker

`prepare_vizwiz.py --sft` turns each question into a chat example whose target is the
**same JSON contract the Lambda's SageMaker backend parses**:

```json
{"image": "images/train/VizWiz_train_00000001.jpg",
 "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]},
              {"role": "assistant", "content": [{"type": "text",
                "text": "{\"speech\": \"Canned tomatoes.\", \"needs_better_view\": false, \"claims_path_clear\": false}"}]}]}
```

Unanswerable photos train the model to reply
`"I can't tell from this photo. Try holding the camera a little further awayâ€¦"` with
`needs_better_view: true`.

[train_vizwiz_lora.py](../uncane/training/sagemaker/train_vizwiz_lora.py) does the following:

- loads the model in bf16 with gradient checkpointing;
- applies LoRA r=16 on the attention projections;
- **masks the loss to the assistant's answer only**;
- trains with the Hugging Face `Trainer`;
- merges the adapters and saves a full model that Hugging Face TGI can serve.

Launch, deploy and switch-over commands are in [AWS_SETUP.md Â§12](AWS_SETUP.md#12-optional-vizwiz-fine-tune-on-sagemaker).
Suggested first run: `--max-samples 4000`, 1 epoch on ml.g5.2xlarge, about 1 hour.

> The fine-tuning scripts were written for this build but **not executed** (no GPU was
> available). Expect to adjust library versions in `training/sagemaker/requirements.txt` to
> match the container you pick.

## 3. Edge evaluation without datasets

| What | How |
|---|---|
| Reflex latency | `pytest -m latency -s` |
| Floor calibration and 3D corridor | `tests/test_geometry_hazards.py` renders synthetic scenes (floor, box ahead, head-height branch, side obstacle) from the camera model and checks zones, levels, urgencies and distances |
| Looming / TTC accuracy | `tests/test_optical_flow.py`; ARCHITECTURE.md Â§3.4 has the TTC table |
| Real-world false-alarm rate | Record 10 minutes of walking with `--source` videos, count spoken "Stop" warnings with no obstacle within 1.5 m, and tune `arbiter.*` and `hazards.min_pixel_fraction` |
