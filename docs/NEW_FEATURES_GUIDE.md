# Edge Feature Guide

The edge app now handles common controls locally, so they continue to work without a cloud connection. Start the app as usual from `uncane/`:

```powershell
python -m uncane.edge.app
```

## Local commands

| Say or type | Result |
|---|---|
| `repeat` / `say that again` | Repeat the latest answer or speech line |
| `stop talking` | Stop queued and currently playing speech |
| `quiet mode on` / `quiet mode off` | Suppress non-critical tones; critical spoken safety warnings remain active |
| `indoor mode` / `outdoor mode` | Apply the configured distance thresholds |
| `speak faster` / `speak slower` | Adjust and persist speech rate |
| `volume up` / `volume down` | Adjust and persist speech and beep volume |
| `read the sign aloud` | Ask the cloud to transcribe visible text verbatim |
| `add note ...`, `show notes`, `clear notes` | Manage local personal notes |
| `where did you last see my phone?` | Search recent detector sightings held in this session |
| `status`, `recalibrate`, `stop guidance`, `feedback`, `save event log` | Read device status, recalibrate, stop homing, record feedback, or export recent context |

Unmatched phrases remain ordinary assistant questions.

## Guidance and monitoring

When a cloud answer identifies a visible target, the edge app can track it locally using detector matches and template matching. It gives periodic direction and distance cues, announces arrival or loss, and stops after the configured timeout. Guidance does not replace the reflex safety system.

The monitor checks image brightness, contrast, glare, floor calibration, and frame-loop stalls. Camera sources are reopened after repeated live-frame read failures. Health messages are rate-limited so they do not crowd out safety speech.

The geometric analyzer can identify a floor region below the calibrated plane as a separate drop-off hazard. Drop-offs use their own double-pulse audio cue and longer release period. Thresholds are configurable in `config/edge_config.yaml`; verify them on the intended camera and surface before relying on them.

## Local data and privacy

The user profile and notes are written to `data/user_profile.json` using atomic replacement. Recent object sightings are memory-only and are cleared when the process exits. Event summaries remain in an in-memory ring buffer and are appended to `data/blackbox.jsonl` for feedback or watchdog events.

Camera snapshots are disabled by default. Enabling `blackbox.save_snapshots` writes an image alongside exported records; treat those images and event files as sensitive data and remove them when no longer needed.

## Configuration and checks

The `environment`, `crowd`, `guidance`, `monitor`, `blackbox`, and `user` sections in `config/edge_config.yaml` control these features. Run the offline suite with:

```powershell
pytest
```

Latency varies by camera, model, and device. Run `pytest -m latency -s` on the target hardware before publishing a latency figure. This software is a prototype, not a substitute for a cane, orientation and mobility training, or validated assistive equipment.
