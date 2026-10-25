# Changes from the Earlier OpenCV Project

Comparison date: 25 October 2026

## Scope and method

This compares the current project at `Desktop/uncane_final` with the earlier project at `Desktop/opencv/uncane-ai`. Neither directory contains a `.git` repository, so this is a file-content comparison, not a commit-history or authorship report. SHA-256 hashes were compared by relative path. Virtual environments, caches, generated AWS SAM output, bytecode, installed-package metadata, logs, and `.env` files were excluded.

The comparison found **12 added files, 12 modified files, and no removed project files**. Added-file counts include supporting materials as well as source and tests.

## What stayed the same

The main split architecture remains in place: the edge device runs depth estimation, geometric hazard analysis, optical-flow threat detection, object detection, and reflex alerts; AWS Lambda and Bedrock handle questions that need visual reasoning. The shared request/response schemas and cloud API contract were not changed. The existing cloud safety-verification flow remains, including a second check against current edge hazards before a cloud answer is spoken.

## Major additions

### On-device commands and user state

- [`src/uncane/edge/commands.py`](../src/uncane/edge/commands.py) recognizes local commands before forwarding unmatched utterances to the assistant. This keeps controls such as repeat, stop talking, quiet mode, indoor/outdoor mode, speech controls, reading, notes, status, and recalibration available offline.
- [`src/uncane/edge/user_profile.py`](../src/uncane/edge/user_profile.py) persists mode, quiet setting, speech rate and volume, personal notes, and feedback count in a local JSON file. Writes use a temporary file and atomic replacement; corrupt or missing profiles fall back to defaults.
- [`src/uncane/edge/object_memory.py`](../src/uncane/edge/object_memory.py) remembers recent detector sightings during the current session, with clock direction and metric distance when available. It supports “where did you last see my phone?” and aliases common object names to detector classes. This sighting history is in memory only and is not saved to disk.

### Guidance after an object is found

- [`src/uncane/edge/guidance.py`](../src/uncane/edge/guidance.py) starts tracking a cloud-identified target locally. It uses detector matches for known COCO classes and template matching for other targets, then supplies updated direction, distance/step cues, a homing tone, arrival, lost-target, and timeout behavior.
- The edge app can automatically start guidance after a usable cloud answer identifies a target, or the user can start and stop it with local commands. Tracking and guidance continue on the device rather than requiring another cloud request for every cue.

### Hazard and safety behavior

- [`src/uncane/edge/hazards.py`](../src/uncane/edge/hazards.py) merges nearby detections across adjacent body-height levels so one person or object is less likely to create multiple alerts. A merged alert keeps the more urgent severity and nearest distance; a head-height-only obstacle can still produce the duck warning.
- The geometric analyzer adds negative-obstacle detection for drop-offs. It estimates distance to the floor-plane edge, distinguishes warning from critical distance, and leaves drop-offs separate from ordinary obstacles.
- [`src/uncane/edge/arbiter.py`](../src/uncane/edge/arbiter.py) adds drop-off-specific speech/earcons and a longer alert hold, quiet-mode filtering for non-critical tones, and crowd behavior that suppresses people alerts unless people are very close. Critical spoken safety warnings remain enabled in quiet mode.
- [`src/uncane/edge/audio_feedback.py`](../src/uncane/edge/audio_feedback.py) expands audio handling with distinct earcons, interruptible prioritized speech, pre-rendered safety phrases for faster playback, and adjustable speech/beep volume and rate. Speech falls back to logging when audio is unavailable.

### Device self-monitoring and review data

- [`src/uncane/edge/monitors.py`](../src/uncane/edge/monitors.py) checks frame brightness, contrast, glare, camera recovery, floor-calibration state, and stalled frame processing. It rate-limits user-facing health messages and runs the watchdog separately from the frame loop.
- [`src/uncane/edge/blackbox.py`](../src/uncane/edge/blackbox.py) keeps a short in-memory ring buffer of hazard, threat, tone, timing, and event summaries. It writes a JSON record when requested or when feedback/watchdog events occur; saving a camera snapshot is optional and disabled by default.
- The camera loop in [`src/uncane/edge/app.py`](../src/uncane/edge/app.py) retries reopening a live camera after repeated read failures. Slower monitor, memory, crowd, and guidance work runs after the core reflex processing for a frame.

## Modified files and their effect

- [`src/uncane/edge/app.py`](../src/uncane/edge/app.py): wires in local commands, saved preferences, monitoring, object memory, crowd detection, guidance, black-box events, reading mode, automatic retry for unusable photos, and better offline answers. Existing cloud answers are still checked against current edge hazards.
- [`src/uncane/edge/arbiter.py`](../src/uncane/edge/arbiter.py), [`src/uncane/edge/audio_feedback.py`](../src/uncane/edge/audio_feedback.py), and [`src/uncane/edge/hazards.py`](../src/uncane/edge/hazards.py): implement the new alert, audio, and negative-obstacle behavior described above.
- [`src/uncane/edge/config.py`](../src/uncane/edge/config.py) and [`config/edge_config.yaml`](../config/edge_config.yaml): add settings for drop-offs, level merging, indoor/outdoor thresholds, crowd mode, guidance, monitoring, black-box retention, user profile storage, TTS backend, and read/retry behavior. Relative paths for the profile and black-box output are anchored to the project root.
- [`src/uncane/edge/detector.py`](../src/uncane/edge/detector.py): does not try to assign an object label to a drop-off.
- [`src/uncane/cloud/tools.py`](../src/uncane/cloud/tools.py) and [`src/uncane/common/prompts.py`](../src/uncane/common/prompts.py): permit a longer spoken answer when the request is to read text, while retaining the existing structured answer contract.
- [`requirements-edge.txt`](../requirements-edge.txt): adds edge-side dependencies for the new runtime features.
- [`README.md`](../README.md): updates the overview, run instructions, project layout, and documentation index for the expanded feature set.

## Tests and supporting documentation

- [`tests/test_new_features.py`](../tests/test_new_features.py) adds focused coverage for merged hazards, drop-offs, quiet/crowd behavior, command parsing, object memory, guidance, monitors, profile persistence, black-box output, and audio helpers.
- [`tests/test_app_integration.py`](../tests/test_app_integration.py) exercises the real edge app wiring with downloaded models and a simulated cloud: local commands, target-to-guidance flow, automatic re-ask, reading image size, offline answers, and feedback recording. These integration tests skip when model files are unavailable.
- [`docs/NEW_FEATURES_GUIDE.md`](NEW_FEATURES_GUIDE.md) explains how to run and try the added features. [`docs/PRODUCT_ROADMAP_DAILY_WEARABLE.md`](PRODUCT_ROADMAP_DAILY_WEARABLE.md) records the broader daily-wearable roadmap.
- `developer-guide.md` and the competition PDF are additional materials in the newer project root; they are not runtime code.

## Practical compatibility notes

- The new edge features do not require an AWS redeployment. They use the existing navigation request/response format; the cloud prompt change only affects reading responses.
- Profile and black-box data are local files. Object sightings are session-only. Black-box snapshots can contain camera imagery if explicitly enabled, so keep that setting off unless snapshots are needed and handle generated records as sensitive data.
- The current documentation reports inconsistent reflex latency measurements: the README gives p50 47 ms / p95 53 ms, while the feature guide gives p50 24.8 ms / p95 30.5 ms in its summary and 30.7 ms in its test instructions. Run `pytest -m latency -s` on the target machine before publishing a single current figure.
