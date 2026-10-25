# Architecture

UnCane-y AI is a wearable navigation co-pilot for blind and low-vision people, built for the
OpenCV AI Competition 2026 (Use Case 3, powered by AWS). This document describes how the
system is built and why. For setup, see [SETUP.md](SETUP.md) and [AWS_SETUP.md](AWS_SETUP.md).

## 1. The split-intelligence idea

A white cane and GPS are passive: they miss head-height hazards (branches, open truck
doors), fast silent threats (e-scooters, EVs), and "last-yard" questions ("where is the door
handle?"). An LLM can answer the last kind, but a 1â€“2 s cloud round trip is too slow for the
first two. So the work is split:

| | Hemisphere 1 â€” Reflexes (edge) | Hemisphere 2 â€” Reasoning (AWS) |
|---|---|---|
| Runs on | The wearable's computer (laptop, Jetson, Pi 5) | API Gateway â†’ Lambda â†’ Claude in Amazon Bedrock |
| Speed | Every frame, target < 50 ms | On request, target < 1.5 s |
| Does | Depth, 3D obstacle corridor, approaching-object detection, beeps, urgent speech | Reading signs, finding objects, describing scenes, transit/location tools, conversation |
| Needs network | **No** â€” works with zero signal | Yes |
| Trust | Authoritative about obstacles | Must pass the safety verifier before anything is spoken |

Rule from the brief: **real-time obstacle avoidance is never sent to the LLM.**

```
 camera â”€â”¬â”€â–º [reflex loop, every frame, ~47 ms] â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
         â”‚     depth (OpenCV 5 DNN) â”€â–º floor calibration â”€â–º 3D corridor hazards â”‚
         â”‚     optical flow (parallel) â”€â–º looming / time-to-contact threats     â”‚
         â”‚     reflex arbiter (debounce) â”€â–º spatial beeps, haptics, "Stop."     â–¼
         â”‚                                                            speaker/haptics
         â”œâ”€â–º [detector thread, 2 Hz] YOLOX (COCO) â”€â–º names hazards ("chair")    â–²
         â”‚                                                                      â”‚
         â””â”€â–º [on a question] sharpest recent frame + hazards + depth grid       â”‚
               â”‚  HTTPS, x-api-key                                              â”‚
               â–¼                                                                â”‚
         API Gateway (usage plan) â”€â–º Lambda                                     â”‚
               load DynamoDB memory â”€â–º Claude (Bedrock) tool loop â”€â–º final_answerâ”‚
               ground target (geometry + edge depth) â”€â–º SafetyVerifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
               S3 telemetry, CloudWatch metrics                                 â”‚
                                   â—„â”€â”€ answer â”€â”€ edge re-verifies vs. *current* hazards
```

## 2. Repository map

| Path | What it is |
|---|---|
| `src/uncane/common/` | Stdlib-only code shared by edge and Lambda: JSON contracts (`schemas.py`), egocentric geometry (`spatial.py`), the anti-hallucination verifier (`verifier.py`), prompts (`prompts.py`) |
| `src/uncane/edge/` | The wearable runtime: `app.py` (main loop), `depth_engine.py`, `geometry.py`, `hazards.py`, `optical_flow.py`, `detector.py`, `arbiter.py`, `audio_feedback.py`, `voice_input.py`, `frame_filter.py`, `cloud_client.py`, `config.py` |
| `src/uncane/cloud/` | The Lambda: `handler.py`, `vlm.py` (Bedrock agent / SageMaker / mock), `tools.py`, `memory.py` (DynamoDB), `telemetry.py` (S3 + EMF metrics), `local_server.py` |
| `template.yaml` | AWS SAM template â€” every AWS resource |
| `config/edge_config.yaml` | Edge tuning knobs |
| `scripts/` | Model download, dataset prep (COCO, VizWiz), API smoke test |
| `eval/` | Detector mAP on COCO, VLM accuracy on VizWiz |
| `training/sagemaker/` | VizWiz LoRA fine-tune + endpoint deployment (optional) |
| `tests/` | 43 tests (42 offline + 1 opt-in latency benchmark) |

## 3. Hemisphere 1 â€” the edge reflex loop

`EdgeApp.process_frame` in [app.py](../uncane/src/uncane/edge/app.py) runs once per camera frame.

### 3.1 Depth ([depth_engine.py](../uncane/src/uncane/edge/depth_engine.py))

MiDaS v2.1 small (ONNX) through OpenCV 5's DNN module, using the **new graph engine**
(`cv2.dnn.ENGINE_NEW`). The classic engine fails on MiDaS at any input size except its
256 px export size; the new engine handles any multiple of 32 but is CPU-only, so GPU
targets fall back to the classic engine at 256 px. Preprocessing (resize, RGB,
ImageNet mean/std) is done in NumPy because `blobFromImage` cannot divide by a per-channel std.

The output is *relative inverse depth*: bigger means closer, with an unknown scale and shift.

### 3.2 From relative depth to meters ([geometry.py](../uncane/src/uncane/edge/geometry.py))

A chest camera at a known height *h* and downward tilt *Ï†* sees the floor in the lower
part of the frame, and trigonometry gives the floor's true depth at every image row:
`Z_floor(row) = h / (tan Î¸Â·cos Ï† + sin Ï†)`. Monocular networks satisfy
`1/Z â‰ˆ aÂ·r + b`, so each frame `GroundPlaneCalibrator` fits *(a, b)* to floor pixels by
trimmed least squares (dropping the worst 40 % of residuals, i.e. obstacles standing on the
floor). It then rejects sudden jumps and smooths with an exponential moving average. The
result is a metric depth map without a depth sensor. Until calibration settles
(3 good frames), the legacy image-sector analyser is used.

If you export a metric depth model (e.g. Depth-Anything-V2 Metric), set
`depth.output: metric` and calibration is skipped.

### 3.3 The 3D walking corridor ([hazards.py](../uncane/src/uncane/edge/hazards.py))

Every depth pixel is back-projected to *(lateral X, height above floor, forward distance)*.
Hazards are cells of a 3 Ã— 3 grid in **body space**, not image space:

| | LEFT (âˆ’1.5â€¦âˆ’0.4 m) | CENTER (Â±0.4 m) | RIGHT (0.4â€¦1.5 m) |
|---|---|---|---|
| **HEAD** (1.3 m â€¦ user height) | branch, sign | **critical < 1.6 m** | |
| **TORSO** (0.5â€“1.3 m) | | **critical < 1.2 m**, warning < 2.2 m | warning < 1.2 m |
| **GROUND** (0.15â€“0.5 m) | | curb, box, bag | |

Anything below 0.15 m is floor, which is what stops the floor from triggering alarms. Head
level gets extra warning distance because a cane sweeps right under it. Distance is the
10th percentile of forward distance in the cell. A cell must cover at least 0.4 % of the
map, which removes speckle. The analyser also emits a coarse 8 Ã— 6 metric **depth grid** that
travels with keyframes, so the cloud can ground distances.

### 3.4 Approaching objects ([optical_flow.py](../uncane/src/uncane/edge/optical_flow.py))

Flow magnitude alone fires constantly on a walking body. The tracker instead:

1. Spreads corners over an 8 Ã— 6 grid, so small objects get tracks, and follows them with
   pyramidal Lucasâ€“Kanade plus a forwardâ€“backward consistency check.
2. Fits the camera's own motion as a similarity transform, using RANSAC.
3. In overlapping windows, measures by least squares how much faster points spread apart
   than the camera model predicts (*looming*).
4. Subtracts the scene-wide median, because walking forward makes everything expand.
   It then smooths per zone and requires 2 consecutive frames.

A divergence rate *e* (per second) gives **time-to-contact â‰ˆ 1/e**. Below 3 s is a WARNING,
below 1.5 s is CRITICAL. Measured on synthetic sequences with known ground truth:

| True TTC | 0.68 s | 1.13 s | 1.68 s | 4.18 s |
|---|---|---|---|---|
| Detected | 0.68 s | 1.18 s | 2.14 s | not flagged (correct, > 3 s) |

Camera pans, static objects and a whole-scene zoom (walking forward) produce no threats. This
is covered by `tests/test_optical_flow.py`.

The flow step runs **concurrently** with depth inference on a one-thread pool. Both are
OpenCV calls that release the GIL.

### 3.5 Naming things ([detector.py](../uncane/src/uncane/edge/detector.py))

YOLOX-S (COCO-80, OpenCV Model Zoo) runs in its own thread at 2 Hz on 416 px input. It
never blocks the reflex loop; it only attaches labels ("chair", "bicycle") to hazards and
threats. It also powers the **offline answer** when the cloud is unreachable: "I can see
chair at 11 o'clock, table at 12 o'clock."

### 3.6 Calm alerts ([arbiter.py](../uncane/src/uncane/edge/arbiter.py), [audio_feedback.py](../uncane/src/uncane/edge/audio_feedback.py))

The brief asks for an ultra-low false-positive rate. The arbiter:

- only reports a hazard seen in 2 of the last 3 frames, and releases it after 4 absent frames;
- sonifies only the single most urgent hazard;
- rate-limits spoken warnings to one per kind every 4 s.

**Audio encoding.** Stereo pan gives direction. Pitch gives height: 440 Hz ground, 660 Hz
body, 1100 Hz head. Beep rate gives distance, from 0.1 s at contact to 0.8 s far away. An
approaching threat overrides everything with a 1500 Hz rapid beep plus a short phrase
("Bicycle coming, right!").

Speech runs on a priority queue: safety messages flush queued cloud chatter, and cloud
answers older than 8 s are dropped. An optional serial haptic belt receives `L,C,R`
intensities.

### 3.7 Keyframes ([frame_filter.py](../uncane/src/uncane/edge/frame_filter.py))

- **SharpFrameBuffer.** When the user asks something, the least-blurred of the last 8 frames
  is sent, judged by the variance of the Laplacian. VizWiz shows how often blind users'
  photos are blurred, and this cheap step fixes most of it.
- **KeyframeFilter (SSIM).** In ambient mode, which is off by default, a frame goes to the
  cloud only when the scene has structurally changed and at least 3 s have passed. This
  saves bandwidth, battery and tokens.

### 3.8 Measured edge latency

Measured on the development laptop: 8-thread CPU, no GPU, OpenCV 5.0.0.
Reproduce with `pytest -m latency -s`.

| Component | Time |
|---|---|
| MiDaS small @128 px / 160 / 256 | 32 / 48 / ~130 ms |
| Hazard analysis | ~4 ms |
| Optical flow (runs in parallel with depth) | ~8 ms |
| **Reflex loop total @128 px** | **p50 47 ms, p95 53 ms** âœ… < 50 ms p50 |
| Reflex loop with detector thread (416 px @ 2 Hz) sharing the CPU | p50 ~60 ms |
| YOLOX @640 / 416 / 320 (own thread) | 341 / 132 / 88 ms |
| Depth-Anything-V2 small @126 px | 96 ms (for GPU devices) |

On a CPU-only device, either disable the detector or keep it at 1â€“2 Hz. On a Jetson,
build OpenCV with CUDA, set `depth.opencv_target: cuda`, and raise `depth.input_size` to 256.

## 4. Hemisphere 2 â€” cloud reasoning

### 4.1 Request path

`POST /v1/navigate` â†’ API Gateway (REST API; API key required; usage plan of 5 req/s,
burst 10, 5 000/day) â†’ Lambda `uncane.cloud.handler.lambda_handler` (Python 3.13, arm64,
1 GB) â†’ Claude in Amazon Bedrock.

### 4.2 The agent ([vlm.py](../uncane/src/uncane/cloud/vlm.py), [tools.py](../uncane/src/uncane/cloud/tools.py))

- **Client.** `anthropic.AnthropicBedrockMantle` calls the Messages API on the
  `bedrock-mantle` endpoint.
- **Model.** Default `anthropic.claude-sonnet-5`; configurable (for example
  `anthropic.claude-haiku-4-5` for lower latency).
- **Effort.** `low`, to stay near the 1.5 s budget. Haiku 4.5 doesn't accept the effort
  parameter, so it is omitted for Haiku.
- **Input.** The sharpest JPEG, plus a text block with the edge sensor readings marked
  *authoritative*, plus up to 4 earlier Q&A turns from DynamoDB (text only).
- **Tools.**
  - `get_location_context`: Amazon Location Service reverse geocoding.
  - `get_transit_arrivals`: your GTFS-realtime adapter. It never invents times; demo mode
    is labelled as demo data.
  - `final_answer`: the structured answer.
- **Why a `final_answer` tool.** Claude in Amazon Bedrock does not support
  `output_config.format` structured outputs, so a tool schema carries the structure:
  `speech`, `target_label`, `target_bbox` (0â€“1000), `suggested_direction`,
  `claims_path_clear`, `needs_better_view`, `confidence`.
- **Loop.** At most 4 turns. Tool results go back in a single user message, failed tools
  return `is_error: true`, and `stop_reason == "refusal"` leads to a safe fallback phrase.
- The system prompt is marked for prompt caching.

### 4.3 Grounding ([spatial.py](../uncane/src/uncane/common/spatial.py))

The model never states distances. It returns a bounding box, and the Lambda computes:

- **bearing** from camera geometry: `atan((cx âˆ’ 0.5)Â·2Â·tan(HFOV/2))`;
- **distance** from the edge depth grid under the box;
- **steps** as distance Ã· 0.7 m.

That produces the demo line *"Turn 30 degrees right, walk three steps. The red water bottle
is on the table."* Every number in it is deterministic.

### 4.4 The safety verifier ([verifier.py](../uncane/src/uncane/common/verifier.py))

This runs **twice**: in Lambda against the hazards captured with the photo, and on the device
right before speaking against the hazards *now*, because the user kept walking.

| Rule | Trigger | Action |
|---|---|---|
| 1 | Model says or flags the path is clear, but the edge has a CRITICAL CENTER hazard | Replace with "Stop. Chair directly ahead at body height, about 0.9 meters." |
| 2 | Model steers the user toward a zone with a CRITICAL hazard | Prefix "Careful: â€¦" |
| 3 | Target lies behind a CRITICAL hazard in the same direction | Prefix "Careful: â€¦ before you reach it." |
| 4 | Model says it is "safe to cross" | Reword to describe the signal and add "Listen for traffic and confirm before you cross." |
| 5 | An approaching threat has TTC â‰¤ 3 s | Prefix "Caution, bicycle approaching from your right." |

Clear-path detection uses both the model's structured flag and a regex bank that
deliberately does not match negations ("the path is *not* clear").

### 4.5 Storage and observability

- **DynamoDB** `SessionsTable`: question/answer text per session, TTL 12 h, never images.
- **S3** `DataBucket`: `telemetry/` JSON per request, partitioned by `dt=YYYY-MM-DD` for Athena.
  `keyframes/` images are stored only if `SaveKeyframes=true`. Both prefixes expire after
  30 days. The bucket is encrypted, blocks public access, and denies non-TLS requests.
  `datasets/` holds training data.
- **CloudWatch.** The Lambda emits Embedded Metric Format lines: `ModelLatencyMs`,
  `TotalLatencyMs`, `SafetyOverride`, `NeedsBetterView`. A dashboard and alarms (errors;
  p90 latency above 3 s) come with the stack. X-Ray tracing is on.

## 5. Contracts

Request (edge â†’ cloud), abbreviated:

```json
{
  "device_id": "uncane-dev-01", "session_id": "uncane-dev-01-3f2a9c1e", "timestamp": 1790000000.0,
  "query": "Where is the red water bottle?", "mode": "query",
  "image_jpeg_b64": "<~40-90 KB JPEG, max side 768 px>",
  "camera": {"width": 640, "height": 480, "hfov_deg": 70.0},
  "hazards": [{"zone": "CENTER", "level": "TORSO", "proximity": 0.62, "urgency": "CRITICAL",
               "distance_m": 1.1, "label": "chair"}],
  "threats": [{"bearing_deg": 24.0, "ttc_s": 1.3, "urgency": "CRITICAL", "zone": "RIGHT", "label": "bicycle"}],
  "depth_grid": {"rows": 6, "cols": 8, "kind": "metric_m", "values": [3.1, 2.9, "â€¦48 values"]},
  "location": {"lat": 40.74, "lon": -73.99}
}
```

Response (cloud â†’ edge):

```json
{
  "speech": "Careful: chair directly ahead at body height, about 1.1 meters before you reach it. Turn 30 degrees right, walk three steps. The red water bottle is on the table.",
  "raw_model_speech": "The red water bottle is on the table.",
  "target": {"label": "red water bottle", "bearing_deg": 28.3, "clock": 1, "distance_m": 2.1, "steps": 3,
             "bbox": [0.82, 0.7, 0.95, 0.95]},
  "safety_override": true, "override_reasons": ["target_path_blocked_center"],
  "needs_better_view": false, "tools_used": ["final_answer"],
  "model_id": "anthropic.claude-sonnet-5", "latency_ms": {"model": 1034.2, "total": 1071.9}, "request_id": "â€¦"
}
```

The full field list is in [schemas.py](../uncane/src/uncane/common/schemas.py). The latency numbers
above are illustrative; see Â§7.

## 6. Graceful degradation

| Failure | What the user experiences |
|---|---|
| No network / API down | Obstacle beeps and "Stop" warnings continue unchanged. After 3 failed calls the device says once, "Connection lost. Obstacle alerts are still active." Questions get an on-device answer from the detector. |
| Bedrock error or refusal | Lambda returns 502 or a fallback phrase, and the device falls back to the offline answer. |
| Late answer (> 6 s) | Dropped; it would describe a scene the user has already left. |
| Blurry or misaimed photo | The model sets `needs_better_view` and says how to re-aim (VizWiz-style). |
| Depth not yet calibrated | Sector analyser (relative) is used until floor calibration converges (~3 frames). |
| Audio device missing | Beeps and speech are logged instead; the pipeline keeps running. |

## 7. Mapping to the brief's success criteria

| Brief target | Where it's met / measured |
|---|---|
| Edge inference < 50 ms | `pytest -m latency -s` â†’ p50 47 ms on a laptop CPU (Â§3.8) |
| Cloud reasoning < 1.5 s | CloudWatch `TotalLatencyMs` p50/p90 on the dashboard, `LatencyAlarm`, `eval_vizwiz_vlm.py` latency percentiles. **Not yet measured**, because the stack hasn't been deployed; the default is `Effort=low`, and `anthropic.claude-haiku-4-5` is the lever if needed |
| High mAP on VizWiz val | VizWiz-VQA has no boxes, so we report VQA accuracy + unanswerable precision/recall (`eval/eval_vizwiz_vlm.py`), and detector mAP on a COCO navigation subset (`eval/eval_detector_coco.py`) |
| Ultra-low false positives | Arbiter debounce, corridor geometry (floor excluded), min-area filter, background-subtracted looming, `SafetyOverride` metric |
| 40 % faster precision tasks | User study protocol in [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) Â§5 |

## 8. What changed from `developer-guide.md`

The original guide was the starting blueprint. The implementation keeps its structure but
fixes these problems:

| Guide | Problem | Now |
|---|---|---|
| `anthropic.claude-3-5-sonnet-20240620-v1:0` via `invoke_model` | Model is retired | `AnthropicBedrockMantle` + `anthropic.claude-sonnet-5`, tool-based structured output |
| `blobFromImage(mean=(0.485,â€¦), scalefactor=1/255)` | Subtracts 0.485 from 0â€“255 pixels and never divides by std | Explicit NumPy normalization |
| Per-frame minâ€“max normalized depth, mean > 180 in image thirds | Something always reads as "close"; the floor triggers alarms; no heights or meters | Floor-calibrated metric depth, 3D corridor with head/torso/ground levels |
| Any LK vector > 15 px = threat | Fires on every head turn or step; misses slow looming | Ego-motion-compensated looming â†’ time-to-contact |
| MSE keyframe filter | Sensitive to lighting | SSIM + minimum interval, plus sharpest-frame selection |
| Verifier checks 3 phrases for CENTER only | Easy to bypass | Structured flag + regex bank + steering/target/crossing/threat rules, run twice |
| `opencv-python-headless` + `cv2.imshow` | The headless build can't show windows | `opencv-python` (headless only on boards without a display) |
