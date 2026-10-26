# UnCane-y AI: demo script (what to say and do)

This is the speaking script. For hardware, props, rehearsal and recovery steps, use
[DEMO_RUNBOOK.md](docs/DEMO_RUNBOOK.md). Lines marked **SAY** are for the presenter or narrator,
**DO** is an action, and **HEAR** is what the device says or plays back.

Roles: a **presenter** (blindfolded, wearing the chest camera) and a **narrator** (talks to the
audience and drives the keyboard). Two people make this much smoother than one.

---

## 1. Before you go on (5 minutes)

```powershell
cd uncane
.\.venv\Scripts\Activate.ps1

# Terminal 1: cloud tier (local mock, or use your deployed API instead)
$env:VLM_BACKEND = "mock"; python -m uncane.cloud.local_server

# Terminal 2: the wearable runtime
$env:UNCANE_API_URL = "http://127.0.0.1:8787"; $env:UNCANE_API_KEY = "local"
python -m uncane.edge.app
```

- Spoken commands need the offline speech backend (`--set voice.backend=vosk`, models from
  `python scripts/download_models.py --only vosk`, then press `v` to talk). With the default
  keyboard backend, the narrator types the same phrases in the terminal. Note text is saved in
  lower case.
- Using a deployed AWS or Nebius endpoint behind a corporate proxy and getting a certificate
  error? Add `--set cloud.verify_ssl=false` for the demo only, and say nothing about it on stage.
- Windows speech has about a second of start-up delay. **Leave a pause after every spoken
  line** before the next action, or it will talk over itself.
- Quick checks: the debug window shows `depth metric` (not `calibrating`), the cloud shows
  `online`, and pressing `1` gets an answer. If `calibrating` persists, tilt the camera down
  until the floor is visible, or say "recalibrate".
- **Show the OpenCV version.** The first log lines print `OpenCV 5.0.0`. Leave the terminal visible
  for a moment, because the competition requires OpenCV 5.
- **The agent tools run in the cloud tier.** The local server (`python -m uncane.cloud.local_server`)
  runs them today. The mock backend follows a fixed script, so say "scripted" if asked. To show a
  real model choosing its own tools, start the server with `VLM_BACKEND=mantle` and your key, and
  rehearse that path first. The live AWS stack does not have the tools yet.
- Print a second copy of the sign and smear it (or hold it out of focus) for the blur beat. Keep the
  sharp copy next to it. The device retries a blurry answer once on its own, so expect two short
  messages in that beat.
- Run `python eval/eval_agent_tools.py` beforehand and keep `eval/agent_traces.json` open in an
  editor. It is the audit trail you show at the end.
- Put the red bottle on the table about 30 degrees to the right and 2 m from the start mark. Put
  your phone on the chair for the memory beat. Put a printed "EXIT" sign on the table.

---

## 2. The 30-second pitch (use as an intro, or the whole thing if time is tight)

**SAY:**

> "A white cane finds what is at your feet. It misses a branch at head height, a drop you
> haven't reached yet, and a scooter coming up silently from the side. Phone apps can describe a
> photo, but only if you stop, aim, and wait. UnCane is a chest-worn co-pilot with two brains.
> The first reacts in under fifty milliseconds, on the device, with no internet. The second is a
> cloud AI agent that finds your door handle and reads signs. It uses OpenCV 5 tools to check
> and zoom into what it sees, and it is never allowed to contradict the safety sensors."

---

## 3. The four-minute live demo

The competition video limit is five minutes, so this leaves room for the team introduction.

| Time | DO | HEAR | SAY (narrator) |
|---|---|---|---|
| 0:00 | Presenter puts on the blindfold at the start mark | "UnCane ready." | "Everything you will hear is computed live. The reflexes run on this laptop. The reasoning runs in the cloud." |
| 0:15 | Presenter walks toward the chair | Beeps speed up and pan towards the chair, then "Stop. Chair ahead." | "That warning was computed on the device in under fifty milliseconds. There was no network call. Direction is left and right in the headphones, pitch is height, and speed is distance." |
| 0:40 | Presenter steps around the chair | Beeps pan away and stop | "A branch at head height sounds higher than a box on the floor, and it gets its own spoken warning: duck or step aside." |
| 1:00 | Narrator presses `1` (or presenter says "where is the red water bottle") | "Looking." then "Turn 30 degrees right, walk three steps. The red water bottle is on the table." | "The model found the bottle. The turn angle and step count are not from the language model. They come from our own camera geometry and the depth sensor." |
| 1:25 | Presenter walks towards it | Every few seconds: "The red water bottle is at 2 o'clock, slightly right, about 1.4 meters, walk two steps." with a steady homing tone | "Now the device is guiding on its own. No more cloud calls. It tracks the bottle locally and tells the presenter when to stop." |
| 1:50 | Presenter reaches the table | "You have reached the red water bottle." | "Arrival is detected on the device." |
| 2:00 | Narrator holds the **smeared** sign in front of the camera and asks "read the sign aloud" | "I could not get a clear view. Trying one more recent frame." then, after a pause, "The photo is too blurry to read. Hold still, aim at the sign and ask again." The log shows `agent OpenCV tools: assess_image_quality` | "Watch the agent. Its first tool is an OpenCV 5 sharpness check, and the number says the photo is unreadable. So it did not guess. It asked the user to re-aim. The OpenCV result changed what it did next." |
| 2:20 | Narrator swaps to the **sharp** sign and asks again | "The sign says: Exit, stairs to the left." The log shows `assess_image_quality -> find_text_regions -> zoom_region` | "Same question, but now the sharpness check passes. OpenCV finds the text region, zooms and sharpens it, and the agent reads from that zoomed image. That is a multi-step loop where each step depends on the last OpenCV result." |
| 2:45 | Presenter says "where did you last see my phone" | "I last saw the cell phone at 3 o'clock, about 1.2 meters away, just now." | "Recent sightings are remembered on the device for the current session. Nothing is uploaded and nothing is saved to disk." |
| 3:00 | Narrator turns Wi-Fi off (or stops the local server). Presenter walks back along the line | Beeps and "Stop" still work. If asked a question: "Cloud assistant is offline. I can see a chair at 12 o'clock." | "Safety never depends on the network. The device tells you the cloud is gone, then keeps protecting you." |
| 3:25 | Narrator shows the debug window, the trace file, then the architecture slide | | "Every cloud answer is checked against the depth map twice, once on the server and once again on the device. Every tool call is logged with its result, so you can audit exactly why the agent answered as it did." |
| 3:50 | | | "Reflexes on the device. Reasoning in the cloud. Every answer verified before it is spoken." |

> If the cloud is slow, do not wait for it. Keep the reflex segment going and say "this is graceful
> degradation" (see section 6).

---

## 4. Optional segments (add if you have 5 minutes or more)

Say these phrases aloud, or type them in the terminal. They run on the device, so they work
without the cloud.

| Segment | DO / SAY | HEAR | Why it matters |
|---|---|---|---|
| **Read text** | Say or type "read the sign aloud" (preset `2` gives a normal short answer, not the long read-aloud mode) | The sign text, spoken in full | "When you ask it to read, the answer is allowed to be as long as the text. Normal answers stay to one or two short sentences." |
| **Quiet mode** | "quiet mode on" | "Quiet mode on." | "Useful in a crowd or a quiet room. Ordinary beeps are silenced. Critical warnings are never silenced." |
| **Indoor mode** | "indoor mode" | "Indoor mode." | "Warning distances shrink for tight spaces. It is remembered next time the device starts." |
| **Speech controls** | "speak slower", then "volume up" | "Speech rate slower." / "Volume up." | "Rate and volume are saved in the user's profile." |
| **Notes** | "add note meet Alex at four", then "show notes" | "Note saved." then the note | "Personal notes stay on the device." |
| **Stop talking** | While a long answer plays: "stop talking" | Speech cuts off | "The user can always interrupt. A safety warning can also cut in on an answer, and the answer is replayed afterwards so it is not lost." |
| **Status** | "status" | "Device ready. outdoor mode. Cloud online. 0 active hazards." | "A quick self-check for the user." |
| **Feedback** | "feedback" | "Feedback recorded." | "Flags a bad moment. The device saves a short event log you can review. Camera snapshots are off unless you turn them on." |
| **Drop-off** | Presenter walks towards a clear step-down | "Drop-off ahead. Stop and check your footing." with a distinct double-pulse tone | "A cane only finds a drop when it is already there. See the caution below." |

> **Caution on drop-offs.** This detector is the least validated feature. Only show it on stage if
> you have rehearsed it on that exact floor with that exact camera mount, and do not claim it works
> everywhere. If it fires wrongly, say "recalibrate" and carry on.

---

## 5. Questions you will be asked

**How fast is it?**
"The safety loop ran at about 47 milliseconds median on a laptop CPU in our benchmark. It varies
by machine, so we measure it on the target hardware. We have not yet measured the cloud round trip
on a deployed stack."

**What if the model gets it wrong?**
"For safety, it cannot. The cloud answer is checked against our own depth sensing, on the server
and again on the device. If the model says the path is clear and the sensor sees an obstacle, the
user hears a stop warning instead. For other answers the agent checks with OpenCV tools first. A
blurry photo gets a re-aim request, and a colour that does not match gets a hedged answer, and the
device will automatically retry once with a fresh frame."

**Is this really agentic, or just a chatbot?**
"The agent has OpenCV 5 tools and its next step depends on what they return. A failed sharpness
check ends the loop with a re-aim request. A found text region leads to a zoom, and it reads the
zoomed image. A colour mismatch changes the answer. Each call is logged with its arguments and result.
Our offline test uses a scripted stand-in for the model, which proves the tools and the data flow. How
often a real model picks the right tool is something we measure separately."

**Which OpenCV version, and where is it used?**
"OpenCV 5.0.0. On the device it runs the depth and object-detection networks through its DNN module,
the optical flow for approaching objects, and the template tracking for guidance. In the cloud it
runs the agent's image tools. The app refuses to start on an older version."

**Is it deployed on AWS?**
"The reasoning tier runs on AWS: API Gateway, Lambda on Graviton, Bedrock, DynamoDB and CloudWatch. The
new OpenCV agent tools are tested locally and not in the deployed stack yet." (Only say this if it is
still true on the day. Update it if you have redeployed.)

**Does it work without internet?**
"Obstacle warnings, the beeps, drop-offs, object memory, notes and every spoken command work with
no connection. Only the questions that need a vision-language model, such as reading a sign or
finding an object, need the cloud."

**What about privacy?**
"Photos are sent only when the user asks a question. Conversation memory is text only. The
profile, notes and object sightings stay on the device, and sightings are never written to disk.
Event-log snapshots are off by default."

**Has it been tested with blind users?**
"Not yet. This is a research prototype and it does not replace a cane, a guide dog or mobility
training. A pilot with orientation and mobility instructors is our next step."

**What does it run on?**
"A laptop and a USB camera today. The same code is aimed at a Jetson Orin Nano for a dedicated
wearable."

**Why two parts?**
"Reflexes cannot wait for a network. Understanding needs a large model. So each does the job it is
best at, and the reflexes never depend on the cloud."

---

## 6. Recovery lines (stay calm and keep talking)

| What goes wrong | SAY | DO |
|---|---|---|
| Cloud is slow or fails | "And this is graceful degradation. Safety keeps working while the cloud catches up." | Carry on with the walk. Switch to the phone hotspot. |
| Wrong answer | "It is honest about uncertainty. Let us give it a better view." | Turn slightly and press `1` again. |
| Agent tools do not fire | "The tools only run on the cloud tier, so let me check that connection." | Check the local server is running with the right backend. With the mock backend, ask "read the sign aloud" or "where is the red water bottle"; other questions use no tools. |
| The blur beat reads the sign anyway | "The sharpness check passed, so the agent went ahead." | Smear it more or move it farther out of focus. The threshold is a sharpness score of about 40. |
| False "Stop" | "The camera shifted on the strap." | Say "recalibrate", or press `c`. |
| Long silence from the speaker | | Windows speech can take a second. Wait, do not repeat the command. |
| No sound at all | "The debug window shows every alert and every spoken line." | Check the output device. Point the audience at the window. |
| Phone not remembered | "It only remembers what the detector has seen this session." | Hold the phone in view for a second, then ask again. |

---

## 7. Closing line

**SAY:**

> "The white cane tells you what is at your feet. UnCane tells you what is coming, what is above
> you, and where things are, and it will never tell you the way is clear when it is not."
