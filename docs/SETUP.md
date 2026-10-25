# Setup guide: development machine and wearable

This guide takes you from a clean machine to a running edge device, first fully offline and
then connected to AWS. The AWS side is covered in [AWS_SETUP.md](AWS_SETUP.md).

**Time:** about 20 minutes for sections 1–5.

---

## 1. Prerequisites

| Need | Notes |
|---|---|
| Python 3.10 – 3.13 | `python --version`. On Windows use the python.org installer and tick "Add to PATH". |
| Git | To clone the repo |
| A camera | Laptop webcam for development. On the body: a USB webcam on a chest strap (GoPro-style mount), or a phone streaming video (§8). |
| Headphones | Stereo is required, because the beeps are panned left and right. For real use, **bone-conduction headphones** keep the ears open to traffic. |
| ~400 MB disk | For the models |
| Optional GPU | NVIDIA Jetson Orin or a CUDA laptop, for larger depth models |

## 2. Install

```bash
git clone <your-repo-url> uncane-ai
cd uncane-ai/uncane
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

`requirements-dev.txt` includes the edge runtime, the cloud tier (for running Lambda code
locally), and the test tools. A production wearable only needs `requirements-edge.txt`.

Linux notes:
- `sounddevice` needs PortAudio: `sudo apt install libportaudio2`.
- `pyttsx3` needs eSpeak: `sudo apt install espeak-ng`.
- On a board without a display, replace `opencv-python` with `opencv-python-headless` and
  run with `--headless`.

## 3. Download the models

```bash
python scripts/download_models.py          # required: MiDaS small (67 MB) + YOLOX-S (36 MB)
python scripts/download_models.py --all    # + Depth-Anything-V2 small (99 MB) + Vosk speech (41 MB)
```

| Model | File | Purpose | License |
|---|---|---|---|
| MiDaS v2.1 small | `models/midas_v21_small_256.onnx` | Monocular depth (default) | MIT |
| YOLOX-S (OpenCV Zoo) | `models/object_detection_yolox_2022nov.onnx` | COCO object names | Apache-2.0 |
| Depth-Anything-V2 small | `models/depth_anything_v2_small.onnx` | Sharper depth, for GPU devices | Apache-2.0 |
| Vosk small en-US | `models/vosk-model-small-en-us-0.15/` | Offline voice questions | Apache-2.0 |

To use Depth-Anything instead of MiDaS, set these in `config/edge_config.yaml`:

```yaml
depth:
  model_path: models/depth_anything_v2_small.onnx
  input_size: 126   # a multiple of 14 for Depth-Anything
```

## 4. Run the tests

```bash
pytest                    # 42 offline tests, ~2 s: verifier, geometry, flow, arbiter, Lambda handler, agent loop
pytest -m latency -s      # reflex-loop benchmark on the real model; asserts p50 < 50 ms
```

To change the latency budget, set `UNCANE_LATENCY_BUDGET_MS=80 pytest -m latency -s`.
All tests should pass before you move on.

## 5. First run: edge only (no AWS)

```bash
python -m uncane.edge.app --set cloud.enabled=false
```

Two windows open:

- **UnCane-y AI – edge**: the camera feed, showing reflex latency (p50/p95), depth status
  (`calibrating` → `metric`), cloud status, active hazards and threats, detector boxes, and
  red dots on points that are looming.
- **depth**: the depth map, with near areas bright.

Keys (click the edge window first):

| Key | Action |
|---|---|
| `1`–`4` | Preset questions (these need the cloud; offline you get the on-device answer) |
| `v` | Push-to-talk (Vosk backend) |
| `c` | Restart floor calibration |
| `q` / `Esc` | Quit |

You can also type a question into the terminal and press Enter.

Try walking slowly toward a chair or wall with the camera at chest height. The beeps should
speed up, and at about 1.2 m you should hear "Stop. Chair ahead."

Other ways to run it:

```bash
python -m uncane.edge.app --source path/to/walk.mp4          # a recorded video (loops)
python -m uncane.edge.app --headless --frames 300            # no windows; prints latency at the end
python -m uncane.edge.app --set depth.input_size=192 --set detector.hz=1
```

## 6. Calibrate for your body and camera (important)

Metric distances come from floor-plane geometry, so these three numbers must be right.

1. **`camera.hfov_deg`**: the camera's horizontal field of view. Check the spec sheet, or
   measure it: stand exactly *D* meters from a wall and mark the left and right edges of the
   image on the wall, *W* meters apart. Then `hfov = 2·atan(W / (2·D))` in degrees. Typical
   webcams are 60–78°; GoPro-style wide mode is about 120° (use its linear or undistorted
   mode).
2. **`mount.height_m`**: the camera lens height above the floor when the user stands
   (chest strap ≈ 1.2–1.4 m).
3. **`mount.pitch_deg`**: how far the camera tilts down. 10–15° is ideal: the floor
   2–4 m ahead should fill the bottom quarter of the image. Check with a phone inclinometer
   app held against the camera.

Also set **`mount.user_height_m`**, the top of the user's head; this bounds head-level hazards.

**Check it.** Stand in an open corridor, wait for the status to read `depth metric`, then
place a box 1.5 m ahead. It should appear as `WARNING box directly ahead at ground level,
about 1.5 meters`, give or take 20 %. If distances are consistently long or short, adjust
`pitch_deg` first, then `height_m`.

The calibrator needs visible floor in the lower image, which is normal when walking. Standing
right in front of a wall (no floor visible) keeps the last good calibration.

## 7. Add the cloud tier locally (still no deployment)

Run the Lambda code on your machine. There are two backends:

```bash
# A) mock VLM: no AWS account needed, deterministic answers
VLM_BACKEND=mock python -m uncane.cloud.local_server
# PowerShell:  $env:VLM_BACKEND="mock"; python -m uncane.cloud.local_server

# B) real Claude in Amazon Bedrock with your AWS credentials (see AWS_SETUP.md §2–3)
AWS_REGION=us-east-1 MODEL_ID=anthropic.claude-sonnet-5 python -m uncane.cloud.local_server
```

In a second terminal:

```bash
# bash
export UNCANE_API_URL=http://127.0.0.1:8787 UNCANE_API_KEY=local
# PowerShell
$env:UNCANE_API_URL="http://127.0.0.1:8787"; $env:UNCANE_API_KEY="local"

python -m uncane.edge.app
```

Press `1` ("Where is the red water bottle?"). You should hear "Looking.", then an answer
that starts with a deterministic turn instruction, for example
"Turn 15 degrees right, walk four steps. …".

To check the API without the camera:

```bash
python scripts/smoke_test_api.py --image tests/data/sample_indoor.jpg --query "What is in front of me?"
```

The smoke test also checks the anti-hallucination rule: it sends a fake critical obstacle and
confirms the reply is not "the path is clear".

## 8. Connect to the deployed AWS stack

After [AWS_SETUP.md](AWS_SETUP.md) §5–6 you have an `ApiUrl` and an API key. Then:

```bash
export UNCANE_API_URL=https://abc123.execute-api.us-east-1.amazonaws.com/prod
export UNCANE_API_KEY=<value from aws apigateway get-api-key --include-value>
python -m uncane.edge.app
```

Or put them in `config/edge_config.yaml` under `cloud:`, but don't commit the key.

## 9. Camera and wearable options

| Setup | How |
|---|---|
| Laptop in a backpack + USB chest camera | The simplest robust option. `camera.source: 0` (or 1 if the laptop has a built-in camera). |
| Phone as the camera | Install an IP-camera app (e.g. "IP Webcam" on Android, or any RTSP app), put phone and laptop on the same Wi-Fi or hotspot, and set `camera.source: "http://<phone-ip>:8080/video"` or `rtsp://…`. Adds 50–150 ms of video latency; fine for demos, not ideal for reflexes. |
| NVIDIA Jetson Orin Nano | JetPack 6. Build OpenCV 5 with CUDA (the pip wheels are CPU-only), then set `depth.opencv_target: cuda` and `depth.input_size: 256`. |
| Raspberry Pi 5 | `opencv-python-headless`, `--headless`, `depth.input_size: 128`, `detector.enabled: false` (or 1 Hz). Expect about 100–150 ms per frame. The `onnxruntime` backend can help. |

## 10. Voice input and haptics

**Voice (Vosk, offline):**

```bash
pip install vosk
python scripts/download_models.py --only vosk
python -m uncane.edge.app --set voice.backend=vosk
```

Press `v` and speak for 4 seconds (`voice.record_seconds`). For a physical button, call
`app.voice.push_to_talk()` from a GPIO callback.

**Haptic belt or vest:** any microcontroller that reads lines like `L,C,R\n` (0–255 each,
at 115200 baud) and drives three vibration motors. Set `audio.haptics_serial_port: COM5`
(or `/dev/ttyUSB0`) and `pip install pyserial`.

## 11. Tuning reference

| Key | Default | Effect |
|---|---|---|
| `depth.input_size` | 128 | Bigger is sharper and slower (~32 ms at 128, ~48 at 160 on a laptop CPU) |
| `hazards.critical_m` / `critical_head_m` | 1.2 / 1.6 | Distance at which "Stop" fires |
| `hazards.corridor_half_width_m` | 0.40 | Body half-width plus margin |
| `hazards.min_pixel_fraction` | 0.004 | Raise if you get speckle false alarms |
| `flow.ttc_warning_s` / `ttc_critical_s` | 3.0 / 1.5 | Approaching-object thresholds |
| `arbiter.confirm_frames` / `window` | 2 / 3 | Debounce: raise for fewer false alarms, lower for faster alerts |
| `arbiter.speech_cooldown_s` | 4 | Minimum gap between repeated spoken warnings |
| `detector.hz` / `input_size` | 2 / 416 | Lower both if the reflex loop exceeds 50 ms on CPU |
| `keyframes.max_side_px` / `jpeg_quality` | 768 / 80 | Upload size, which drives token cost and latency |
| `cloud.ambient_mode` | false | Proactively describe signs and doors when the scene changes |

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot open camera source 0` | Try `--source 1`; close other apps using the camera; on macOS grant Terminal camera permission. |
| Status stuck on `calibrating` | Floor not visible: tilt the camera down (`mount.pitch_deg`) and check `mount.height_m`. |
| Distances off by a constant factor | Wrong `hfov_deg` or `pitch_deg` (§6). |
| Constant beeping in open space | Floor misclassified: pitch or height is wrong, or raise `hazards.floor_tolerance_m` to 0.2. |
| Reflex p50 > 50 ms | Lower `depth.input_size`, set `detector.hz: 1`, or disable the detector; close other heavy apps. |
| `Targets are not supported by the new graph engine` warning | Harmless on CPU. For CUDA, the code switches to the classic engine automatically (then use input 256). |
| No sound | `python -c "import sounddevice; print(sounddevice.query_devices())"`; beeps and speech are logged when audio is unavailable. |
| Cloud always `OFFLINE` | Check `UNCANE_API_URL` (must include the `/prod` stage) and `UNCANE_API_KEY`; run `scripts/smoke_test_api.py`. |
| HTTP 403 from the API | Missing or wrong API key (AWS_SETUP.md §6). |
