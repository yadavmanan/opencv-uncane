# Live demo runbook: the "wow moment"

The brief's three-beat demo:

1. **Setup.** The presenter is blindfolded, wearing a chest camera, on a stage cluttered
   with a chair, a table, boxes and a red water bottle.
2. **Reflexes (edge).** They walk toward the chair. OpenCV maps it instantly, and fast
   spatial beeps plus "Stop. Chair ahead." prevent a collision.
3. **Reasoning (cloud).** They ask "Where is the red water bottle?" Bedrock answers:
   "Turn 30 degrees right, walk three steps. It is on the table."

## 1. Hardware checklist

- [ ] Laptop (charged plus charger) in a backpack, or on a table with a long USB cable
- [ ] USB webcam on a chest strap, tilted down 10–15°. Measure `hfov_deg`, `height_m` and `pitch_deg` (SETUP.md §6).
- [ ] Stereo or bone-conduction headphones, plus a **Bluetooth speaker or audio splitter** so the audience hears the beeps
- [ ] A phone hotspot as backup Wi-Fi. Venue Wi-Fi is the most common failure.
- [ ] Props: a chair, a table, a *red* water bottle, 2–3 boxes, and a tape line on the floor for the walking path
- [ ] A second screen facing the judges showing the debug window (latency, hazards, answers)

## 2. Configuration for the stage

`config/edge_config.yaml` overrides for the demo (or pass them with `--set`):

```yaml
camera: {source: 0, hfov_deg: <measured>}
mount: {height_m: <measured>, pitch_deg: <measured>, user_height_m: <presenter>}
hazards: {critical_m: 1.3}            # a little extra margin on stage
detector: {hz: 2, input_size: 416}
voice:
  backend: keyboard                    # presets are more reliable than a microphone on a loud stage
  presets:
    "1": "Where is the red water bottle?"
    "2": "What does the sign in front of me say?"
    "3": "Where is the door handle?"
    "4": "Describe what is around me."
cloud: {max_answer_age_s: 8.0}
```

Put a printed sign on the table (e.g. "EXIT →") for preset 2.

## 3. Rehearsal (the day before, and again 30 minutes before)

1. `sam deploy` is done and `python scripts/smoke_test_api.py` passes. Run it twice so the
   Lambda is warm. **Note the second-run latency.**
2. Walk the path 5 times with eyes open:
   - "Stop" should fire at 1.0–1.5 m from the chair every time;
   - no false "Stop" in the open part of the path;
   - if it misfires, adjust `pitch_deg` first (SETUP.md §12).
3. Place the bottle about 30° right and about 2 m away. Press `1` from the start mark, and
   check the answer names a turn, steps and "table". If the bearing is off, fix `hfov_deg`.
4. Pull the network cable or turn off Wi-Fi and walk again. Beeps and "Stop" must still
   work. Pressing `1` should give "Cloud assistant is offline. I can see …". This is the
   graceful-degradation beat; consider showing it on purpose.
5. Keep the CloudWatch dashboard (`DashboardUrl`) open on a second tab.

## 4. On-stage script (about 3 minutes)

| Time | Presenter | System | Narrator's line |
|---|---|---|---|
| 0:00 | Puts on the blindfold at the start mark | "UnCane ready." | "Everything you'll hear is computed live: the reflexes on this laptop, the reasoning on AWS." |
| 0:20 | Walks toward the chair | Beeps speed up and pan to the chair's side → "Stop. Chair ahead." | "That warning took under 50 milliseconds. There was no cloud call; it would still work with zero signal." |
| 0:45 | Steps around it (beeps pan away and stop) | | "Pitch tells height. A head-height branch sounds different from a box on the floor." |
| 1:00 | Asks "Where is the red water bottle?" (key `1`) | "Looking." … "Turn 30 degrees right, walk three steps. The red water bottle is on the table." | "Claude on Bedrock found the bottle. The turn angle and step count come from our own camera geometry and depth, not from the language model." |
| 1:30 | Walks to it and picks it up | | |
| 1:50 | (optional) Operator holds up a sign, presses `2` | Reads the sign | |
| 2:10 | (optional) Operator disables Wi-Fi; presenter walks back | Beeps continue; "Connection lost. Obstacle alerts are still active." | "Graceful degradation: safety never depends on the network." |
| 2:40 | (optional) Show the dashboard | Latency vs. the 1.5 s line, safety overrides | "Every answer is checked against the depth map. The model can't tell you the path is clear when it isn't." |

## 5. Measuring the brief's user-impact target (40 % faster precision tasks)

Protocol for a small within-subject study (blindfolded sighted volunteers for development;
blind participants only with proper consent and supervision):

1. Tasks: (a) find a named object on a table in a room, (b) find a door handle, (c) walk a
   6 m path with 2 obstacles.
2. Conditions: white cane only, versus cane plus UnCane. Randomise the order and the object
   placement.
3. Record completion time, collisions and assistance requests. Report median time per task
   and condition, and the % reduction.
4. Log the session's `request_id`s so cloud latency and override rates can be joined from
   the S3 telemetry.

## 6. If something goes wrong on stage

| Problem | Recovery |
|---|---|
| Cloud slow or failing | Keep going: the reflex demo is independent. Say "and this is graceful degradation". Switch to the phone hotspot. |
| Answer is wrong | Press `1` again after turning slightly; the sharpest-frame buffer and a new view usually fix it. |
| False "Stop" | The prop is too close to the path edge, or the pitch changed when the strap moved. Re-seat the camera and press `c` to recalibrate. |
| No sound | Check the OS output device; the debug window still shows every alert and spoken line. |
