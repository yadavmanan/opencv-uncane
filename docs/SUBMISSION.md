# UnCane-y AI: a wearable navigation co-pilot that reacts in 47 ms and checks the AI before it speaks

**OpenCV AI Competition 2026, powered by AWS: technical report and final submission**

| | |
|---|---|
| **Team** | `[TEAM NAME]` |
| **Members** | `[Name, role]` · `[Name, role]` · `[Name, role]` |
| **Featured path** | **Agentic Vision** (primary). Not entering Best Use of COOL (see §6.4). |
| **Code** | `[repo URL, with access granted to the judges]` |
| **Demo video (≤ 5 min)** | `[YouTube/Vimeo unlisted URL]` |
| **Live endpoint** | `[ApiUrl]/v1/health`. The full live demo is by arranged screen share, since the edge runs on a body-worn camera. |
| **Domains** | Accessibility · Physical AI · Safety · Smart cities |

> **Safety notice.** UnCane-y AI is a research prototype. It does not replace a white cane, a
> guide dog, or orientation-and-mobility (O&M) training. It never certifies that a street
> crossing or a path is safe, and it is built so that it cannot.

---

## TL;DR for judges

- **Problem.** A white cane only detects what is at ground level and within reach. It misses
  head-height hazards (branches, open truck doors), fast silent threats (e-scooters, EVs), and
  "last-yard" questions such as *where is the door handle?* Cloud vision-language models can
  answer the last of these, but a 1–2 s round trip is far too slow for the first two.
- **Our approach: split intelligence.**
  - **Hemisphere 1, Reflexes.** Runs on the device with OpenCV 5 on every frame. It does not
    need the network. Measured latency is **p50 47 ms, p95 53 ms on a laptop CPU**.
  - **Hemisphere 2, Reasoning.** On request, a Claude agent in Amazon Bedrock, served from
    an arm64 (Graviton) Lambda, reads signs, finds objects and calls tools.
- **The OpenCV 5 work.**
  - Monocular depth runs on OpenCV 5's **new DNN graph engine**.
  - A **floor-plane calibrator** turns relative depth into metres without a depth sensor.
  - Hazards are placed in a **3D walking corridor** split into head, torso and ground levels.
  - **Ego-motion-compensated looming** from optical flow gives **time-to-contact**.
  - YOLOX from the OpenCV Model Zoo names objects.
  - Laplacian-variance selection picks the sharpest frame, and SSIM picks keyframes.
- **The agentic part.**
  - OpenCV output goes into the agent's prompt as *authoritative* evidence.
  - The agent answers with a bounding box, never with a distance. **OpenCV geometry turns
    that box into "turn 30° right, walk three steps".**
  - A **safety verifier** compares every answer with the depth map **twice**: once in Lambda,
    and again on the device just before speaking, because the user kept walking. It rewrites
    or blocks any answer that contradicts the depth map.
  - The visual evidence therefore changes what the system says and does next.
- **What is measured and what is not yet.** Edge latency, TTC accuracy, floor-calibration
  error, detector mAP and verifier behaviour are measured and reproducible (§7). Cloud
  latency, VizWiz accuracy and the user study are marked **`[TO MEASURE]`** below. They will
  be filled in from the deployed stack before submission, and we have not invented them.

---

## Table of contents

1. [Problem and impact](#1-problem-and-impact)
2. [Users and beneficiaries](#2-users-and-beneficiaries)
3. [System overview](#3-system-overview)
4. [OpenCV 5 implementation (Hemisphere 1)](#4-opencv-5-implementation-hemisphere-1)
5. [AWS deployment (Hemisphere 2)](#5-aws-deployment-hemisphere-2)
6. [Agentic Vision: the perception, decision and action loop](#6-agentic-vision-the-perception-decision-and-action-loop)
7. [Evaluation](#7-evaluation)
8. [Failure cases and limitations](#8-failure-cases-and-limitations)
9. [Responsible use, privacy and security](#9-responsible-use-privacy-and-security)
10. [User experience and accessibility](#10-user-experience-and-accessibility)
11. [Reproducibility: build, deploy, test](#11-reproducibility-build-deploy-test)
12. [Demo and video](#12-demo-and-video)
13. [Roadmap](#13-roadmap)
14. [Team](#14-team)
15. [Appendix: criteria-to-evidence map](#appendix-a-judging-criteria-to-evidence-map)

---

## 1. Problem and impact

### 1.1 The gap

The WHO estimates that at least **2.2 billion people** have a near or distance vision
impairment, and tens of millions of them are blind. `[cite the current WHO fact sheet]`. The
tools most of them use for independent travel each leave a specific gap:

| Tool | What it misses |
|---|---|
| **White cane** | Anything above waist height (branches, signs, open truck doors). Anything that moves fast and quietly (e-scooters, cyclists, EVs). |
| **GPS navigation apps** | They work in 2D and get you to the right block. They do not get you to the door handle, and they cannot see temporary hazards such as construction barriers or parked scooters. |
| **Phone "describe this photo" apps** | You have to stop, take out the phone, aim and wait. They describe a picture; they do not warn you about anything. |

### 1.2 The micro-journey we solve

| Moment | Pain | UnCane-y AI |
|---|---|---|
| Stepping off the bus | "Which way am I facing? What does that sign say?" | The user asks out loud and hears the sign text and the street context. |
| A barrier appears that wasn't there yesterday, and a scooter approaches silently | No warning until contact | Beeps rise in rate and pitch, then **"Stop. Barrier ahead."** / **"Bicycle coming, right!"** |
| Arriving at the address | "Where is the door handle?" | **"Turn 15 degrees left, two steps. The handle is at waist height."** |

### 1.3 Why it matters

- **Safety.** Head-height collisions and collisions with moving objects are exactly the
  injuries a cane cannot prevent. Our reflex tier targets those two categories specifically.
- **Independence.** Last-yard precision ("where is the door, the button, the empty seat?")
  is where people most often have to ask a stranger for help.
- **Works with no signal.** The safety tier has no network dependency. That matters in
  subways, basements and rural areas, which are the places where a cloud-only product fails.

**Evidence of benefit** (§7.5) comes from a within-subject study protocol that compares cane
alone with cane plus UnCane. Results: `[TO MEASURE: median task time and % reduction, collisions, n]`.

---

## 2. Users and beneficiaries

| Group | Role | What they get |
|---|---|---|
| **Blind and low-vision pedestrians** | Primary users | Head-height and approaching-object warnings, object finding, sign reading, scene description |
| **Orientation & Mobility (O&M) instructors** | Trainers and influencers | A training aid used alongside cane instruction, plus session telemetry to review together |
| **Vision-rehabilitation and vocational agencies** | Typical buyers of assistive technology | A per-seat program with privacy-by-default data handling |
| **Transit agencies and cities** | Future partners | Accessible stations, and in future anonymised maps of temporary hazards |
| **Wearable and smart-glasses makers** | Licensees | A drop-in safety plus reasoning layer through the same JSON contract |

**Co-design commitment.** "Nothing about us without us." Development so far used blindfolded
sighted volunteers. Testing with blind participants happens only with informed consent and an
O&M professional present (§9.4). `[Add any partner organisation or tester feedback here.]`

---

## 3. System overview

### 3.1 Architecture diagram

> `[Embed docs/img/architecture.png here: the same diagram as below, drawn cleanly with the
> OpenCV 5 components in one colour and the AWS components in another.]`

```
                         ┌───────────────────── WEARABLE (edge) ─────────────────────┐
 chest camera ──frames──►│ Hemisphere 1: REFLEXES (OpenCV 5)   every frame, p50 47 ms │
                         │                                                           │
                         │  cv2.dnn (ENGINE_NEW) MiDaS depth ─► floor-plane calib.   │
                         │        ─► 3D corridor hazards (head/torso/ground × L/C/R) │
                         │  calcOpticalFlowPyrLK + RANSAC ego-motion ─► looming/TTC  │  (runs in parallel)
                         │  YOLOX (OpenCV Zoo, 2 Hz thread) ─► names ("chair")      │
                         │  Arbiter (2-of-3 debounce) ─► spatial beeps · "Stop."     │──► stereo audio / haptics
                         │  Laplacian sharpest-frame buffer · SSIM keyframe filter   │
                         │  Edge SafetyVerifier (re-check vs. *current* hazards)     │──► speech
                         └───────────────┬───────────────────────────▲───────────────┘
                     question + sharpest │ JPEG + hazards + threats  │ verified answer
                     + 8×6 metric depth  │ + location   (HTTPS, API key)
                                         ▼                           │
 ┌──────────────────────────────── AWS (us-east-1) ────────────────────────────────────┐
 │ API Gateway REST (API key, usage plan 5 rps / burst 10 / 5k per day)                 │
 │   └─► Lambda  Python 3.13 · arm64 (Graviton) · 1 GB · X-Ray                          │
 │         ├─ DynamoDB SessionsTable: last 4 Q&A turns, text only, TTL 12 h             │
 │         ├─ Claude in Amazon Bedrock (Messages API, bedrock-mantle) agent loop ≤4 turns│
 │         │    tools: get_location_context (Amazon Location Service)                   │
 │         │           get_transit_arrivals (GTFS-rt adapter, never invents times)      │
 │         │           final_answer (structured: speech, bbox, claims_path_clear, …)    │
 │         ├─ Grounding: bbox + camera HFOV + edge depth grid ─► bearing, metres, steps │
 │         ├─ Cloud SafetyVerifier (vs. hazards captured with the photo)                │
 │         ├─ S3 telemetry/dt=YYYY-MM-DD (30-day expiry, SSE, TLS-only, no images)      │
 │         └─ CloudWatch EMF metrics + dashboard + alarms (errors, p90 > 3 s)           │
 └──────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why we split the work this way

| | Hemisphere 1: Reflexes (edge) | Hemisphere 2: Reasoning (AWS) |
|---|---|---|
| Runs on | The wearable's computer (laptop, Jetson Orin, Raspberry Pi 5) | API Gateway → Lambda (arm64) → Claude in Bedrock |
| Latency | Every frame, **p50 47 ms** (measured) | On request, target < 1.5 s (`[TO MEASURE]`) |
| Does | Depth, 3D corridor, approaching objects, beeps, urgent speech | Reading, finding objects, describing scenes, location and transit, multi-turn memory |
| Needs network | **No** | Yes |
| Trust | Authoritative about obstacles | Must pass the safety verifier, twice |

**Design rule:** real-time obstacle avoidance is never delegated to an LLM. The language
model is allowed to *help* the user. It is never allowed to *clear* the user's path.

---

## 4. OpenCV 5 implementation (Hemisphere 1)

All code is in `src/uncane/edge/`. `EdgeApp.process_frame` in `app.py` runs once per frame.

### 4.1 Monocular depth on the OpenCV 5 DNN new graph engine (`depth_engine.py`)

- **Model.** MiDaS v2.1 small (ONNX), run through `cv2.dnn` with **`cv2.dnn.ENGINE_NEW`**.
- **Why OpenCV 5 specifically.** The classic engine only runs MiDaS at its 256 px export
  size. **The OpenCV 5 graph engine accepts any input that is a multiple of 32.** That let us
  drop to 128 px and reach a 32 ms depth pass on CPU. GPU targets fall back to the classic
  engine at 256 px.
- **Preprocessing.** Resizing, RGB conversion and ImageNet mean/std normalisation are done
  explicitly in NumPy. `blobFromImage` cannot divide by a per-channel std, and the common
  `scalefactor=1/255, mean=(0.485,…)` recipe is quietly wrong. We found and fixed this bug.
- **Alternative model.** Depth-Anything-V2 small (126 px) is supported for GPU devices.

### 4.2 From relative depth to metres without a depth sensor (`geometry.py`)

A chest camera at height *h* and downward pitch *φ* sees the floor in the lower part of the
image. For each image row the true floor depth is

  `Z_floor(row) = h / (tan θ_row · cos φ + sin φ)`

Monocular networks output inverse depth that is roughly affine, `1/Z ≈ a·r + b`. On every
frame, `GroundPlaneCalibrator` does the following:

1. It fits *(a, b)* to the floor pixels by **trimmed least squares**. It drops the worst
   40 % of residuals, which are mostly obstacles standing on the floor.
2. It rejects sudden jumps in the fit, then smooths with an exponential moving average.
3. Until 3 good frames have been seen, it falls back to a relative sector analyser.

**Result:** metric depth from a single RGB camera, with **< 5 % median error on floor pixels**
in synthetic scenes (`tests/test_geometry_hazards.py`).

### 4.3 A 3D walking corridor in body space (`hazards.py`)

Every depth pixel is back-projected to (lateral X, height above floor, forward distance).
Hazards are cells of a 3 × 3 grid **in body space, not image space**:

| | LEFT (−1.5…−0.4 m) | CENTER (±0.4 m) | RIGHT (0.4…1.5 m) |
|---|---|---|---|
| **HEAD** (1.3 m … user height) | | **critical < 1.6 m** | |
| **TORSO** (0.5–1.3 m) | warning < 1.2 m | **critical < 1.2 m**, warning < 2.2 m | warning < 1.2 m |
| **GROUND** (0.15–0.5 m) | | curb, box, bag | |

- Anything below 0.15 m counts as floor. This is what stops the floor itself from causing
  false alarms.
- Head height gets a longer warning distance, because the cane sweeps underneath it.
- Each cell's distance is the 10th percentile of forward distance in that cell. A cell must
  cover at least 0.4 % of the image, which removes speckle.
- The analyser also produces an **8 × 6 metric depth grid**, which goes to the cloud with
  each keyframe so the cloud can ground distances (§6).

### 4.4 Approaching objects: ego-motion-compensated looming (`optical_flow.py`)

Raw flow magnitude fires all the time on a walking body. Our tracker does this instead:

1. Spreads corners over an 8 × 6 grid and tracks them with `cv2.calcOpticalFlowPyrLK`,
   with a **forward–backward consistency check**.
2. Fits the camera's own motion as a similarity transform with **RANSAC**
   (`cv2.estimateAffinePartial2D`).
3. Measures **looming** by least squares in overlapping windows: how much faster points
   spread apart than the camera model predicts.
4. Subtracts the scene-wide median, because walking forward makes everything expand. It then
   smooths per zone and requires 2 consecutive frames.

The divergence rate *e* gives **time-to-contact ≈ 1/e**. Below 3 s is a WARNING; below 1.5 s
is CRITICAL. Flow runs **concurrently** with depth inference on a one-thread pool, since both
OpenCV calls release the GIL.

### 4.5 Naming hazards (`detector.py`)

YOLOX-S (COCO-80, from the **OpenCV Model Zoo**) runs through `cv2.dnn` in its own 2 Hz
thread at 416 px. It never blocks the reflex loop. It only attaches labels ("chair",
"bicycle") to hazards and threats. When there is no network it also provides the
**offline answer**: *"Cloud assistant is offline. I can see a chair at 11 o'clock, a table at 12 o'clock."*

### 4.6 Keyframe quality (`frame_filter.py`)

- **SharpFrameBuffer.** When the user asks a question, the device sends the least-blurred of
  the last 8 frames, scored by `cv2.Laplacian` variance. About 30 % of VizWiz photos taken by
  blind users are unanswerable, mostly from blur and bad framing, so this cheap step fixes
  most of the failures before they reach the model.
- **KeyframeFilter (SSIM).** In optional ambient mode, a frame goes to the cloud only when
  the scene has changed structurally and at least 3 s have passed.

### 4.7 Calm alerts (`arbiter.py`, `audio_feedback.py`)

- **Debounce.** A hazard is reported only if it appears in 2 of the last 3 frames. It is
  released after 4 frames without it. Only the single most urgent hazard is sonified, and
  each kind of spoken warning is limited to once every 4 s.
- **Spatial audio encoding:**
  - stereo pan gives **direction**;
  - pitch gives **height** (440 Hz ground, 660 Hz body, 1100 Hz head);
  - beep interval gives **distance** (0.1 s at contact, 0.8 s far away).
- **Approaching threats** override everything: a 1500 Hz rapid beep plus a short phrase.
- **Speech priority queue.** Safety speech flushes any queued cloud chatter. Cloud answers
  older than 8 s are dropped.
- **Haptics.** An optional serial belt receives `L,C,R` intensities.

### 4.8 OpenCV 5 APIs used

| Area | APIs |
|---|---|
| DNN | `cv2.dnn.readNet`, `ENGINE_NEW`, `setPreferableBackend/Target`, `forward` |
| Motion | `goodFeaturesToTrack`, `calcOpticalFlowPyrLK`, `estimateAffinePartial2D` (RANSAC) |
| Image quality | `Laplacian`, SSIM built from `GaussianBlur` |
| Pre/post-processing | `resize`, `cvtColor`, `imencode` (JPEG), NMS for YOLOX |
| UI and debug | `imshow`, drawing primitives, depth colormap |

`[Verify this table against the code before submitting.]`

---

## 5. AWS deployment (Hemisphere 2)

The whole cloud tier is one **AWS SAM** template (`template.yaml`) and deploys with
`sam build && sam deploy`.

| Resource | Service | Configuration |
|---|---|---|
| `NavigatorApi` | API Gateway REST, stage `prod` | `POST /v1/navigate`, `GET /v1/health`, API key required |
| `NavigatorApiUsagePlan` | API Gateway usage plan | 5 req/s, burst 10, 5 000 per day per key, which caps the Bedrock bill |
| `NavigatorFunction` | **AWS Lambda, arm64 (Graviton)** | Python 3.13, 1 GB, 28 s timeout, X-Ray on |
| Reasoning | **Claude in Amazon Bedrock** | `anthropic.claude-sonnet-5` by default (Haiku 4.5 as a one-parameter latency lever), effort `low`, cached system prompt |
| `SessionsTable` | DynamoDB on-demand | Q&A text only, TTL 12 h, never images |
| `DataBucket` | S3 | SSE-S3, public access blocked, TLS required, 30-day expiry on telemetry |
| Location | Amazon Location Service (optional) | `SearchPlaceIndexForPosition` on one index |
| Observability | CloudWatch EMF, dashboard, alarms, SNS | `ModelLatencyMs`, `TotalLatencyMs`, `SafetyOverride`, `NeedsBetterView`; alarms on errors and p90 > 3 s |
| Fine-tuning (optional) | SageMaker | Qwen2-VL-2B + LoRA on VizWiz. Scripts are complete; not run in this build (§8). |

**Least-privilege IAM.** The Lambda can call `bedrock-mantle:CreateInference`, do CRUD on its
own table and `PutObject` on its own bucket. It gets `geo:` and `sagemaker:InvokeEndpoint`
only on the single configured resource, and only when that resource is configured.

**Deployment record.** Fill this in from the live stack:

| Item | Value |
|---|---|
| Region | `[us-east-1]` |
| Stack | `uncane-ai` |
| ApiUrl | `[…/prod]` |
| Model ID | `[anthropic.claude-sonnet-5]` |
| SAM CLI version | `[1.166]` |
| Deploy date and commit | `[date] / [git sha]` |

**Cost per question** is about **$0.01** (roughly 2 k input and 300 output tokens at
illustrative Sonnet-class pricing). Lambda, API Gateway, DynamoDB and S3 together cost
under $0.0001. The serverless stack costs nothing when idle. `[Replace with figures from
Cost Explorer after the evaluation runs.]`

---

## 6. Agentic Vision: the perception, decision and action loop

The competition requires that "image or video results must influence a subsequent plan, tool
call, action, or request for human approval." UnCane meets this in four places. In each one,
**OpenCV 5 output changes what the system does next.**

### 6.1 Agent workflow diagram

```
 PERCEIVE (OpenCV 5, edge)            DECIDE (Claude agent, Bedrock)            ACT
 ─────────────────────────            ──────────────────────────────            ───
 depth ─► corridor hazards ─┐
 flow  ─► looming threats  ─┼─► authoritative sensor block ─►  tool loop (≤ 4 turns)
 YOLOX ─► labels           ─┤   + sharpest frame (Laplacian)    ├─ get_location_context ─► Amazon Location
 8×6 metric depth grid     ─┘   + 4 turns of DynamoDB memory    ├─ get_transit_arrivals ─► GTFS-rt
                                                                └─ final_answer{speech, bbox,
                                                                     claims_path_clear,
                                                                     needs_better_view, confidence}
                                                                              │
          ┌───────────────────────────────────────────────────────────────────┘
          ▼
 GROUND (OpenCV geometry): bbox + HFOV + depth grid ─► bearing°, metres, steps
          ▼
 VERIFY #1 (Lambda): answer vs. hazards captured with the photo ─► rewrite / block / prefix
          ▼
 VERIFY #2 (edge, at speak time): answer vs. CURRENT hazards ─► rewrite / block / drop if > 8 s old
          ▼
 ACT: speak  ·  spatial beeps  ·  ask the user to re-aim  ·  offline answer if the cloud is down
```

### 6.2 Four places where vision changes the next action

| # | Visual evidence | What changes next |
|---|---|---|
| 1 | **Hazards from the depth corridor and flow** go into the prompt as authoritative readings. | The agent's plan changes. For example, it sends the user around the chair rather than straight ahead. |
| 2 | **The agent's bbox plus the OpenCV depth grid** | The spoken action is computed, not generated: `bearing = atan((cx − 0.5)·2·tan(HFOV/2))`, distance comes from the depth grid under the box, and steps = distance ÷ 0.7 m. |
| 3 | **The safety verifier** compares the agent's output with the depth map, twice. | The action is overridden (rules below). |
| 4 | **Sharpness score and `needs_better_view`** | The system asks the user to re-aim ("turn a little to your right and ask again") rather than giving a confident wrong answer. This is a request for human action. |
| 5 | **`assess_image_quality`**, an OpenCV tool the agent calls mid-loop | A blurry, dark or glared photo ends the loop with a re-aim request instead of a guess; a sharp one lets the agent go on to read. |
| 6 | **`zoom_region`, `analyze_region_color`, `measure_region_depth`**, also OpenCV tools the agent calls | The zoomed crop goes back to the model, so its next answer is based on a new image it asked for; a colour mismatch lowers its confidence and changes its wording; the depth reading replaces its own distance guess. See §6.5. |

**The verifier's rules** (`src/uncane/common/verifier.py`):

| Rule | Trigger | Action |
|---|---|---|
| 1 | The model says or flags that the path is clear, but the edge sees a CRITICAL hazard in the CENTER cell | Replace the answer with "Stop. Chair directly ahead at body height, about 0.9 meters." |
| 2 | The model steers the user toward a zone with a CRITICAL hazard | Prefix "Careful: …" |
| 3 | The target lies behind a CRITICAL hazard | Prefix "Careful: … before you reach it." |
| 4 | The model says "safe to cross" | Reword to describe the signal and add "Listen for traffic and confirm before you cross." |
| 5 | An approaching threat has TTC ≤ 3 s | Prefix "Caution, bicycle approaching from your right." |

Clear-path detection combines the model's structured `claims_path_clear` flag with a regex
bank that deliberately ignores negations ("the path is *not* clear").

### 6.3 Example trace: OpenCV output changes the action

`[Replace with a real trace from S3 telemetry. The structure below follows the actual contract in schemas.py.]`

```json
// 1. PERCEIVE: edge request
{"query": "Where is the red water bottle?",
 "hazards": [{"zone":"CENTER","level":"TORSO","urgency":"CRITICAL","distance_m":1.1,"label":"chair"}],
 "depth_grid": {"rows":6,"cols":8,"kind":"metric_m","values":[…]}}

// 2. DECIDE: the agent's final_answer tool call
{"speech":"The red water bottle is on the table.", "target_label":"red water bottle",
 "target_bbox":[820,700,950,950], "claims_path_clear":false, "needs_better_view":false}

// 3. GROUND + VERIFY: the Lambda response
{"speech":"Careful: chair directly ahead at body height, about 1.1 meters before you reach it. Turn 30 degrees right, walk three steps. The red water bottle is on the table.",
 "target":{"bearing_deg":28.3,"distance_m":2.1,"steps":3},
 "safety_override":true, "override_reasons":["target_path_blocked_center"]}
```

**Without OpenCV**, the user would hear "The red water bottle is on the table." and walk into
the chair. **With OpenCV**, the depth corridor changed the spoken action, and the depth grid
supplied the turn angle and step count.

### 6.4 Evaluation of the agentic part

| Dimension | How we evaluate it | Result |
|---|---|---|
| Task success | Object-finding success rate in the demo room: the correct object is reached within the instructed bearing ±15° | `[TO MEASURE: x / N]` |
| Safety failure handling | Adversarial prompts ("Is it safe to walk straight ahead?") with an injected CRITICAL hazard: `scripts/smoke_test_api.py`, `tests/test_verifier.py`, `tests/test_cloud_handler.py` | Path never declared clear: **PASS** in offline tests; live `[TO MEASURE]` |
| Override rate in the field | CloudWatch `SafetyOverride` metric | `[TO MEASURE]`% of answers |
| Re-aim behaviour | `needs_better_view` compared with VizWiz "unanswerable" labels (precision / recall) | `[TO MEASURE]` |
| Observability | X-Ray trace per request, per-request JSON in S3, EMF metrics, dashboard | In place |

### 6.5 The agent calls OpenCV 5 tools

The cloud agent has five OpenCV-backed tools (`src/uncane/cloud/vision_tools.py`), offered alongside
`final_answer`. They run on the photo the device sent, and their output returns to the model in the
same loop (the zoomed crop comes back as an image), so the model's next step depends on what OpenCV
measured.

| Tool | OpenCV operation | How the result changes the next step |
|---|---|---|
| `assess_image_quality` | Laplacian variance, brightness, contrast, glare | Unusable photo: the agent stops and asks the user to re-aim. Usable: it continues to read |
| `find_text_regions` | Morphological gradient, Otsu threshold, contour filtering | Picks the box to zoom into |
| `zoom_region` | Crop, cubic upscale, CLAHE, unsharp mask | The model reads from the new, larger image |
| `analyze_region_color` | HSV histogram of the box | A colour mismatch makes the agent doubt or drop its candidate |
| `measure_region_depth` | Edge depth grid under the box | Replaces the model's own distance estimate, and flags a critical obstacle |

Every call is recorded (arguments, result, image returned or not, milliseconds) in the response's
`agent_trace` and in the S3 telemetry record. The loop is capped at five model turns and four tool
calls per request, and failures come back as tool errors rather than crashes. Human control: unusable
evidence produces a request for the person to act, not a guess, and the verifier still has the last word
on safety.

**Reproduce:** `python eval/eval_agent_tools.py` runs six labelled cases through the Lambda handler
and writes the traces to `eval/agent_traces.json`. `pytest tests/test_vision_tools.py` covers each tool
on real rendered images, the Claude and OpenAI-style loops (including the image fed back), the tool
budget and the trace.

**What this does and does not show.** The offline run uses the mock backend, which follows a fixed
policy, so it proves the tools, the data flow and the logging, not model judgement. How often a real
model chooses the right tool is `[TO MEASURE]` with `--backend mantle` or `bedrock` and must be reported
separately. `find_text_regions` is a candidate finder: on a cluttered photo it returns boxes that are not
text, which is why the agent confirms by zooming. These tools are not deployed in the live stack yet.
| Human control | The user starts every cloud query. The device never moves the user; it only advises. Crossings are never certified. Safety speech always takes priority over AI speech. | By design |

**Why not COOL.** Our heavy vision workload runs on the wearable, not in the cloud, because
safety must not depend on the network. Our cloud tier does run on Graviton (arm64 Lambda),
but its core workload is Bedrock reasoning, not OpenCV. Running COOL on a Graviton batch
re-processor for recorded walks is listed as future work (§13). We are not claiming the COOL
award.

---

## 7. Evaluation

Every number below can be reproduced with the command shown next to it.

### 7.1 Edge latency

Laptop with an 8-thread CPU, no GPU, OpenCV 5.0.0. Reproduce with `pytest -m latency -s`.

| Component | Time |
|---|---|
| MiDaS small at 128 / 160 / 256 px | 32 / 48 / ~130 ms |
| Hazard analysis | ~4 ms |
| Optical flow (runs in parallel with depth) | ~8 ms |
| **Reflex loop total at 128 px** | **p50 47 ms, p95 53 ms** (meets the < 50 ms target) |
| Reflex loop with the detector thread (416 px, 2 Hz) sharing the CPU | p50 ~60 ms |
| YOLOX at 640 / 416 / 320 px (own thread) | 341 / 132 / 88 ms |

### 7.2 Time-to-contact accuracy

Synthetic looming sequences with known ground truth. Reproduce with `pytest tests/test_optical_flow.py`.

| True TTC | 0.68 s | 1.13 s | 1.68 s | 4.18 s |
|---|---|---|---|---|
| Detected | 0.68 s | 1.18 s | 2.14 s | not flagged (correct, since it is above 3 s) |

Camera pans, static objects and a whole-scene zoom (walking forward) produce **zero** false threats.

### 7.3 Depth calibration and corridor

`tests/test_geometry_hazards.py` renders synthetic scenes (floor, box ahead, branch at head
height, obstacle to the side) from the camera model:

- **floor calibration error below 5 % median**;
- correct zone, level and urgency for every scripted obstacle.

### 7.4 Detector: COCO navigation subset

Reproduce with `python eval/eval_detector_coco.py --input-size 416`.

| Metric | Value |
|---|---|
| mAP@[.5:.95] | **0.349** |
| mAP@.5 | 0.467 |
| mAP small / medium / large | 0.152 / 0.420 / 0.576 |
| Latency p50 | 96 ms |

This used 60 images and 570 boxes covering 18 navigation classes. `[Rerun with --max-images 500 for the final number.]`
Small-object AP is low at 416 px. That is acceptable because the detector only *names*
hazards; the depth pipeline is what *finds* them.

### 7.5 Cloud reasoning: VizWiz-VQA (real photos taken by blind users)

Reproduce with `python eval/eval_vizwiz_vlm.py --limit 200 --seed 0`.

| Metric | Result |
|---|---|
| Lenient VQA accuracy, answerable questions | `[TO MEASURE]` |
| Same, navigation-relevant subset (`--nav-only`) | `[TO MEASURE]` |
| Unanswerable precision / recall (`needs_better_view`) | `[TO MEASURE]` / `[TO MEASURE]` |
| End-to-end latency p50 / p90 (target < 1.5 s) | `[TO MEASURE]` / `[TO MEASURE]` |
| Sonnet 5 compared with Haiku 4.5 (accuracy vs. latency) | `[TO MEASURE]` |

VizWiz-VQA has no bounding boxes, so mAP is not defined for it. We report VQA accuracy and
answerability instead, and mAP on COCO in §7.4. Our accuracy metric checks whether the
annotator answers appear in our spoken sentence, so it is **not comparable to the leaderboard**.

### 7.6 User study (task time, cane only vs. cane plus UnCane)

Protocol (`docs/DEMO_RUNBOOK.md` §5): a within-subject design with randomised order and
randomised object placement. Tasks: (a) find a named object, (b) find a door handle,
(c) walk 6 m past 2 obstacles.

| Task | Cane only (median) | Cane + UnCane (median) | Change | Collisions (cane only → with UnCane) |
|---|---|---|---|---|
| Find object | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Find door handle | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 6 m obstacle path | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

Participants: `[n, blindfolded sighted / blind, consent process]`.

### 7.7 False alarms in a real walk

Method: record 10 minutes of walking, then count spoken "Stop" warnings when no obstacle was
within 1.5 m. Result: `[TO MEASURE: false stops per minute]`, with arbiter settings
`confirm_frames=2/3`.

---

## 8. Failure cases and limitations

We include these on purpose. Judges asked for them, and they are what we would want to know
before trusting the device.

| Area | Limitation or failure case | Mitigation, current or planned |
|---|---|---|
| **Monocular depth** | It needs visible floor to calibrate. Standing right in front of a wall freezes the last good calibration. Very reflective or very dark floors degrade the fit. | Fall back to the relative sector analyser. Planned: support a stereo or ToF camera. |
| **Camera mount** | If the strap shifts and the pitch changes, distances drift and false "Stop" warnings can follow. | Press `c` to recalibrate. Planned: online pitch estimation from the horizon or IMU. |
| **Night, rain, glare** | Not yet tested. MiDaS and YOLOX both degrade in low light. | Test before making any claims. Planned: low-light mode and an IR camera option. |
| **Small or thin obstacles** | Poles, wires and glass doors are difficult for monocular depth. Small-object AP is 0.152. | The min-area filter trades this off against speckle. The cane still covers the ground. |
| **Drop-offs and stairs going down** | The corridor model detects things that stick up from the floor, not holes or drops below floor level. | **Known gap.** Planned: negative-obstacle detection (floor-plane discontinuity). |
| **Detector load on CPU** | Running the detector thread raises the reflex p50 to ~60 ms. | Run it at 1–2 Hz, use a Jetson with CUDA, or turn it off. |
| **Cloud latency** | Not yet measured on the deployed stack. Lambda cold starts add 1–2 s. | Haiku 4.5 is a one-parameter switch, plus `Effort=low` and the nearest region. |
| **LLM errors** | The model can misread text or name the wrong object. | The verifier blocks *unsafe* errors only. Non-safety errors (a wrong sign reading) can still reach the user. |
| **Re-aim loop** | `needs_better_view` asks the user to re-aim, but the device does not yet re-capture and re-ask automatically. | Planned: an automatic active-perception retry. |
| **Fine-tuning** | The SageMaker VizWiz LoRA scripts are complete but **were not executed** (no GPU quota). | Claude with sensor-grounded prompting is the shipped path. |
| **Authentication** | API keys identify devices and enforce quotas; they are not user authentication. | Planned: Cognito or IoT Core device certificates, plus WAF. |
| **Transit data** | No live GTFS-rt adapter has been deployed. | The tool returns an error and the agent says the data is unavailable. It never invents arrival times. |

---

## 9. Responsible use, privacy and security

### 9.1 Safety-first design

- The LLM **cannot override the sensors**. The verifier runs in Lambda and again on the device.
- The system **never certifies a street crossing**. The prompt forbids it and the verifier
  enforces it regardless.
- **Graceful degradation.** If the network fails, obstacle alerts continue unchanged and the
  user hears once: "Connection lost. Obstacle alerts are still active."
- Late cloud answers (older than 8 s) are dropped, because they describe a scene the user
  has already walked past.
- **Messaging.** We never describe the device as replacing the cane, a guide dog or O&M
  training, or as "preventing injury". That wording carries medical-device regulatory
  implications (FDA, EU MDR).

### 9.2 Privacy

| Data | Default handling |
|---|---|
| Camera frames | Processed on the device. Only the single sharpest frame is sent, and only when the user asks a question. **Not stored** unless `SaveKeyframes=true`, which requires explicit consent. |
| Bystanders | Images are not retained by default. Telemetry contains no images. |
| Conversation memory | Text only, in DynamoDB, deleted automatically after 12 hours |
| Telemetry | JSON in S3, expires after 30 days, SSE-S3 encrypted, public access blocked, TLS required |
| Location | Sent only if the user enables it, and used only for reverse geocoding |

### 9.3 Security and cost controls

- API key per device; usage-plan throttle and daily quota; key rotation documented.
- Least-privilege Lambda role; Bedrock model-invocation logging recommended.
- AWS Budgets alarm set before the first deploy; the serverless stack costs nothing when idle.

### 9.4 Human-subjects ethics

Development testing uses blindfolded sighted volunteers. Any session with blind participants
requires informed consent and an O&M professional present. Participants keep their cane at
all times, and the session stops on any request.

---

## 10. User experience and accessibility

- **Hands-free and eyes-free.** A chest camera, bone-conduction headphones (which keep the
  ears open to traffic) and push-to-talk. Voice recognition is offline (Vosk), and keyboard
  presets are available for loud environments.
- **Audio that carries information.** Direction, height and distance are each encoded in a
  separate property of the beep (pan, pitch, rate), and only one hazard is sonified at a
  time. Every alert also appears in the debug window for sighted helpers.
- **Directions in body terms.** "Turn 30 degrees right, walk three steps", or clock positions
  ("chair at 11 o'clock"). Never "over there".
- **Honest uncertainty.** "I can't tell from this photo. Turn a little to your right and ask again."
- **Optional haptic belt** for users who are deaf-blind or who don't want audio.
- **Remaining accessibility work.** Setup and calibration currently need a sighted helper.
  `[Planned: a voice-guided calibration routine.]`

---

## 11. Reproducibility: build, deploy, test

### 11.1 Pinned environment

| Component | Version |
|---|---|
| Python | 3.13 (edge supports 3.10–3.13) |
| OpenCV | **5.0.0** (`opencv-python`) |
| NumPy | `[pin]` |
| anthropic[bedrock] | `[pin]` |
| AWS SAM CLI | 1.166 |
| Models | MiDaS v2.1 small (MIT), YOLOX-S 2022nov from the OpenCV Zoo (Apache-2.0), Depth-Anything-V2 small (Apache-2.0), Vosk small en-US (Apache-2.0) |

> `[Before submitting: generate requirements-lock.txt with "pip freeze" and include it, so
> that every ">=" in the requirements files has an exact pinned version.]`

### 11.2 Ten-minute local run (no AWS needed)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt && pip install -e .
python scripts/download_models.py                      # MiDaS + YOLOX (~100 MB)
pytest                                                 # 42 offline tests, ~2 s
pytest -m latency -s                                   # asserts reflex p50 < 50 ms

VLM_BACKEND=mock python -m uncane.cloud.local_server   # terminal 1: mock cloud tier
UNCANE_API_URL=http://127.0.0.1:8787 UNCANE_API_KEY=local python -m uncane.edge.app   # terminal 2
```

### 11.3 Deploy to AWS

```bash
sam build && sam deploy --guided           # about 3 min; creates every resource in §5
aws apigateway get-api-key --api-key <ApiKeyId> --include-value --query value --output text
python scripts/smoke_test_api.py --image tests/data/sample_indoor.jpg --query "What is in front of me?"
# Run it twice (the first call is a cold start). The anti-hallucination check must print PASS.
```

Teardown is `sam delete --stack-name uncane-ai` after emptying the bucket. Full details are in
`docs/SETUP.md` and `docs/AWS_SETUP.md`.

### 11.4 Test suite

There are 42 offline tests plus 1 opt-in latency benchmark, covering the verifier, geometry,
flow, arbiter, Lambda handler and agent loop:

| File | Covers |
|---|---|
| `test_verifier.py` | All 5 safety rules and negation handling |
| `test_geometry_hazards.py` | Floor calibration and corridor zones and levels |
| `test_optical_flow.py` | TTC accuracy; no false threats from pans or forward walking |
| `test_cloud_handler.py` | Lambda end to end with a mock VLM, including the override path |
| `test_spatial_schemas.py` | Bearing, steps and clock grounding; contract round-trips |
| `test_edge_units.py` | Arbiter debounce, frame filters, audio encoding |
| `test_edge_latency.py` | Reflex p50 < 50 ms on the real model |

---

## 12. Demo and video

### 12.1 Live demo, three beats (about 3 minutes, also the core of the video)

1. **Setup.** The presenter is blindfolded and wears the chest camera. The stage has a chair,
   a table, boxes and a red water bottle.
2. **Reflexes.** The presenter walks toward the chair. Beeps speed up and pan toward it, then
   **"Stop. Chair ahead."** This takes under 50 ms with no cloud call.
3. **Reasoning.** "Where is the red water bottle?" → **"Turn 30 degrees right, walk three
   steps. It's on the table."** The angle and step count come from OpenCV geometry, not the LLM.
4. **Graceful degradation.** Wi-Fi is switched off. The beeps continue, and the device says
   "Connection lost. Obstacle alerts are still active."
5. **Verifier.** The presenter asks "Is it safe to walk straight ahead?" with the chair in
   front. The answer is overridden. The CloudWatch dashboard shows the `SafetyOverride` count
   and latency against the 1.5 s line.

### 12.2 Video outline (≤ 5:00)

| Time | Content |
|---|---|
| 0:00–0:25 | The team on camera; the problem in one sentence; clip of a cane missing a head-height branch |
| 0:25–1:00 | Architecture diagram: the two hemispheres, OpenCV 5 components, AWS components |
| 1:00–2:30 | Live blindfolded walk: chair "Stop", water bottle found, sign read (split screen with the debug window) |
| 2:30–3:15 | Agentic trace: the request, the agent's tool call, the verifier override (real JSON from S3), the X-Ray trace |
| 3:15–3:45 | Network off; safety continues |
| 3:45–4:30 | Results: latency, TTC, mAP, VizWiz, user study; one failure case shown honestly |
| 4:30–5:00 | Responsible use, roadmap, thanks |

---

## 13. Roadmap

1. **Active-perception retry.** When `needs_better_view` is true, the device guides the user's
   aim with audio, re-captures automatically and asks again. This closes the loop without the
   user having to do anything.
2. **Negative obstacles.** Detect drop-offs, stairs going down and curbs from discontinuities
   in the floor plane.
3. **Wearable hardware.** Jetson Orin Nano with OpenCV 5 built with CUDA (depth at 256 px),
   plus battery-life testing.
4. **COOL on Graviton for fleet analytics.** Re-process consented recorded walks with COOL on
   Graviton to mine false alarms and tune thresholds, benchmarked against x86 OpenCV.
5. **Fleet hazard map.** Users anonymously share temporary hazards (construction, scooters)
   onto a city map, in partnership with transit agencies.
6. **Production authentication.** Cognito or IoT Core certificates, and WAF.
7. **Pilot** with an O&M instructor or rehab centre, following the §7.6 protocol at scale.

---

## 14. Team

| Name | Role | Background and competitions |
|---|---|---|
| `[ ]` | `[Edge CV / OpenCV]` | `[ ]` |
| `[ ]` | `[Cloud / AWS]` | `[ ]` |
| `[ ]` | `[Product / UX / user study]` | `[ ]` |

Contact: `[email]` · `#OpenCVComp26`

---

## Appendix A: judging criteria to evidence map

| Criterion (weight) | Where the evidence is |
|---|---|
| **Technical execution (30 %)** | §4: OpenCV 5 `ENGINE_NEW` depth, floor-plane metric calibration, 3D body-space corridor, RANSAC ego-motion looming and TTC, YOLOX from the Model Zoo, Laplacian and SSIM frame selection. §7.1–7.4: measured latency, TTC, calibration and mAP. 42 tests. |
| **Innovation (20 %)** | Split intelligence (edge reflexes and cloud reasoning). The LLM returns a bbox and **OpenCV geometry computes the action**. A verifier checks the LLM against the depth map **twice**. Pitch encodes height in the audio. |
| **Real-world impact (20 %)** | §1–2: the cane's head-height and moving-object gap; works with no signal. §7.6 user study. §13 pilot plan. |
| **User experience (10 %)** | §10: hands-free, one hazard at a time, directions in body terms, honest uncertainty, haptics, offline answers. |
| **Documentation and presentation (10 %)** | This report; `docs/ARCHITECTURE.md`, `SETUP.md`, `AWS_SETUP.md`, `DATA_AND_TRAINING.md`, `DEMO_RUNBOOK.md`; the 5-minute video. |
| **Cloud delivery, reproducibility, responsible operation (10 %)** | §5: one-command SAM deploy, Graviton Lambda, least-privilege IAM, usage plan, X-Ray, EMF dashboard, alarms. §9: privacy by default, TTLs, encryption. §11: pinned environment and tests. |
| **Agentic Vision award** | §6: workflow diagram, four places where vision changes the next action, a real trace, and an evaluation of task success, failure handling, observability and human control. |

## Appendix B: submission checklist

- [ ] Technical report (this file), with every `[TO MEASURE]` and `[ ]` filled in or removed
- [ ] Repository access granted to the judges
- [x] `requirements-lock.txt` with exact versions (`uncane/requirements-lock.txt`, OpenCV 5.0.0.93)
- [ ] Architecture diagram image in `docs/img/` (OpenCV 5 and AWS components clearly marked)
- [ ] Agent workflow diagram image (perceive, decide, act)
- [ ] Stack deployed and `ApiUrl/v1/health` returns 200; live screen-share slot arranged
- [ ] Video ≤ 5:00, public or unlisted, showing the team, the app working, the architecture and the results
- [ ] VizWiz eval run (`--limit 200`), latency p50/p90 captured from CloudWatch
- [ ] One real verifier-override trace exported from S3
- [ ] User study (even a small n) or an explicit statement that it is pending
- [ ] Failure cases shown in the report and the video
- [ ] Remove references to "Use Case 3" and "the brief" from the other docs (they come from an earlier spec, not this competition)
- [ ] Submitted before **26 Oct 2026, 11:59 p.m. PT**
