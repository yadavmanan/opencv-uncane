# UnCane-y AI: product overview and go-to-market plan

> **Status.** Working prototype (competition build). The edge safety loop, the cloud reasoning
> service and the safety verifier are implemented and tested. The cloud tier has not yet been
> deployed or tested with real users. Every price, cost and market figure below is a planning
> assumption to validate, not a measured result.

---

## 1. The product in one paragraph

UnCane-y AI is a wearable AI co-pilot for blind and low-vision people. A small chest-mounted
camera, connected to a pocket computer, **warns about dangers a white cane cannot detect**:

- a branch or open truck door at head height;
- a silent e-scooter or electric car closing in;
- a construction barrier that wasn't there yesterday.

It warns with directional beeps within about 50 ms, fully on the device and with no internet
needed. When the user wants to know something, they simply ask: *"Where is the door
handle?"*, *"What does this sign say?"*, *"Where's my red water bottle?"*. The answer is
spoken, with precise directions: *"Turn 30 degrees right, walk three steps. It's on the
table."* Every answer from the AI is checked against the device's own depth sensing before it
is spoken, so the assistant can never tell the user a path is clear when it isn't.

**Positioning line:** *The white cane tells you what's at your feet. UnCane tells you what's
coming, what's above you, and where things are.*

## 2. The problem

- The competition brief cites **285 million people with visual impairment** worldwide. Use
  the latest WHO figures in any external material; WHO also reports far larger numbers for
  all levels of vision impairment.
- **The white cane is passive and ground-level.** It misses anything above waist height, and
  anything moving fast.
- **GPS apps are 2D.** They get you to the block, not to the door handle. They don't see
  temporary hazards: construction, parked scooters, open doors.
- **Phone "describe this photo" apps are slow and hands-busy.** You stop, hold up the phone,
  aim, wait. They are not safety tools.
- The result, from the brief's user journey: anxiety, injuries, and restricted independence.
  Many people avoid unfamiliar routes altogether.

### The micro-journey we solve

| Moment | Pain | UnCane |
|---|---|---|
| **0:00 Getting off the bus** | "Which way am I facing? Where's the crosswalk?" | Ask; hear a spoken description of signs, crossings and street context. |
| **0:02 The dynamic threat** | Unmapped barrier on the sidewalk; a silent scooter approaching | Beeps rise in speed and pitch; "Stop. Barrier ahead." / "Scooter coming, right!" |
| **0:05 The last yard** | "I'm at the address, but where is the door handle?" | "Turn 15 degrees left, two steps. The handle is at waist height." |

## 3. Capabilities

### 3.1 Instant safety: "the reflexes"

Runs on the device, works with zero signal.

| Capability | What the user experiences | Technical basis |
|---|---|---|
| **Obstacle warning in a 3D walking corridor** | Beeps whose **direction** (left/right), **pitch** (ground, body, head height) and **speed** (distance) encode the hazard; "Stop. Chair ahead." when close | Monocular depth AI (OpenCV 5) converted to real distances using the camera's height and tilt; hazards located in body space |
| **Head-height hazard detection** | A distinct high tone plus "Head height obstacle ahead. Duck or step aside." | Height-above-floor classification; extra warning distance at head level |
| **Approaching-object alerts** | "Bicycle coming, right!" with an urgent tone | Motion analysis that ignores the user's own walking and estimates time-to-impact |
| **Knows what things are** | "Chair ahead" rather than "obstacle ahead" | On-device object recognition (80 everyday object types) |
| **Calm by design** | Few false alarms; one sound at a time; warnings not repeated every second | Alerts must persist across frames; only the most urgent hazard is voiced |
| **Offline fallback** | "Cloud assistant is offline. I can see a chair at 12 o'clock, a table at 1 o'clock." | Local object recognition answers when there's no signal |

Measured on a laptop CPU: **47 ms** median reaction time for the safety loop.

### 3.2 Understanding: "the reasoning"

Runs in the cloud on AWS.

| Capability | Example |
|---|---|
| **Find objects** with precise directions | "Where's my water bottle?" → "Turn 30 degrees right, walk three steps. It's on the table." |
| **Read text** | Signs, menus, door numbers, labels, bus-stop plates |
| **Describe scenes** | "Describe what's around me." → "You're in a lobby. Elevators on your right, reception desk ahead." |
| **Last-yard precision** | Door handles, elevator buttons, empty seats, shop entrances |
| **Conversation memory** | "What about the other door?" works as a follow-up |
| **Location and transit** (with integrations) | "Where am I?" → nearest address; "When's the next 12 bus?" from live transit feeds |
| **Honest uncertainty** | "I can't tell from this photo. Turn a little to your right and ask again." |
| **Picks the sharpest frame** | Automatically sends the least-blurred recent frame, which matters because blind users' photos are often blurred |
| **Optional ambient mode** | Proactively mentions signs, doors and crossings when the scene changes |

### 3.3 Trust and safety: the differentiator

- **The AI cannot override the sensors.** Before any cloud answer is spoken, it is checked
  twice against live depth data. If it says "the path is clear" while an obstacle is ahead,
  the user hears "Stop. Obstacle ahead" instead.
- **It never certifies a street crossing.** It may describe a pedestrian signal, but always
  adds "Listen for traffic and confirm before you cross."
- **It degrades gracefully.** If the network drops, safety alerts continue unchanged.
- **It is private by default.** Photos are processed but not stored. Conversation memory is
  text-only and auto-deletes after 12 hours.

### 3.4 Hardware options

| Configuration | Parts | Rough parts cost (verify) |
|---|---|---|
| **Starter / pilot kit** | Chest-strap USB camera + bone-conduction headphones + a user's existing laptop or mini-PC in a bag | ~$150–250 |
| **Dedicated wearable** | Camera + NVIDIA Jetson Orin Nano (or similar AI board) + battery + bone-conduction headphones | ~$400–600 |
| **Low-cost** | Raspberry Pi 5 build (slower safety loop; detector reduced) | ~$200–300 |
| **Add-on** | Haptic belt or vest: vibrates left, centre or right | ~$50–150 DIY |
| **Future** | Smart glasses or phone-only app; fully offline vision-language model on next-gen chips | from the brief's roadmap |

## 4. Who buys it

The **user** is almost never the only **buyer**. Assistive technology is often funded by
agencies, employers or programs. Plan for both.

| Segment | Who pays | Why they buy | Notes |
|---|---|---|---|
| **Individual users** (B2C) | The user or their family | Independence; safer commuting and travel | Price-sensitive; needs financing or subsidy; word of mouth in blind communities is powerful |
| **Rehabilitation and vocational programs** | Government vocational-rehabilitation agencies, veterans' blind-rehabilitation services, national sight-loss charities | Clients get and keep jobs, travel independently | Often fund assistive tech for employment; long procurement but high-trust |
| **Orientation & Mobility (O&M) specialists** | Their agencies | A training tool alongside cane training | Key influencers; win them first |
| **Employers and universities** | HR / disability services (reasonable accommodations) | Accessibility compliance, retention | Accommodation budgets exist |
| **Cities and transit agencies** (B2G) | Municipal accessibility and transit budgets | Accessible stations, smart crosswalks, compliance | Matches the brief's "smart city" vision; slow, but large contracts |
| **Device makers** (B2B licensing) | Smart-glasses, smart-cane and wearable companies | Adds a safety layer plus AI reasoning to their hardware | License the edge engine + cloud API per device |

## 5. Competitive landscape

Research each before pitching; these products change fast.

| Category | Examples | Their strength | Our angle |
|---|---|---|---|
| Phone "describe my surroundings" apps | Be My Eyes / Be My AI, Microsoft Seeing AI | Free or cheap, great OCR and description | Hands-free, continuous, **real-time safety**; they don't warn you about a scooter |
| AI smart glasses for the blind | Envision Glasses, OrCam MyEye | Polished hardware, reading | We add **3D obstacle and threat detection** and verified answers |
| Smart canes / obstacle sensors | WeWALK and ultrasonic canes | Detect obstacles, some above waist | We add **what** it is, moving-object prediction, and conversational help |
| Human-assisted services | Volunteer or paid video agents | Human judgement | Instant, private, always available; can hand off to a human later |

**Why we win, in three bullets:**

1. The only design that combines **millisecond on-device safety** with **cloud AI reasoning**.
2. **Verified answers**: the AI is mathematically constrained by the depth sensor, which
   builds trust.
3. **Works offline** for safety; the cloud is a bonus, not a dependency.

## 6. Business model options

The cloud is the main running cost. Each question costs roughly **$0.005–0.02** in AI model
fees, depending on the model. This is an estimate; see [AWS_SETUP.md §11](AWS_SETUP.md#11-cost-estimate).
A heavy user asking 30 questions a day costs about **$5–15 per month**. The safety features
cost nothing to run, because they are on the device.

| Model | How it works | Example pricing (hypothesis) | Pros / cons |
|---|---|---|---|
| **Hardware + subscription** | Sell the kit; monthly plan for AI questions | Kit $499; $19/month with a fair-use cap (e.g. 1 000 questions) | Recurring revenue; the cap protects margin |
| **Freemium** | Safety features free forever; AI questions paid | Free safety + 10 questions/day; $9.99/month unlimited | Great for adoption and the mission; needs volume |
| **Agency / program license** | Annual per-seat price including hardware, training and support | $900–1 500 per user per year | Fits how rehab agencies buy; slower sales |
| **B2B licensing** | License the engine to device makers | Per-device royalty + per-question API fee | Scales without owning hardware |
| **B2G / smart city** | Deploy with transit agencies; integrate crosswalk and transit data | Project contract + annual service fee | Large, strategic; long cycles |

**Recommended starting point:** freemium for individuals, where safety is always free and is
the mission story, plus a program license for rehab agencies, which brings revenue and
credibility. Add device-maker licensing once the engine is proven.

Cost levers already built in:
- switching to a faster, cheaper model (`anthropic.claude-haiku-4-5`) is one configuration change;
- per-device daily quotas;
- only the sharpest frame is sent, and only on request.

## 7. How to sell it: a step-by-step plan

### Phase 0: now (competition → proof)
1. **Win or place in the competition.** Film the stage demo (blindfolded walk, chair
   avoided, water bottle found). This video is your best sales asset.
2. **Deploy the cloud tier** and publish real numbers: cloud latency, safety override rate,
   VizWiz accuracy (`eval/`).
3. **Co-design with blind users.** Recruit 5–10 testers through a local blindness
   organisation. Nothing replaces their feedback. "Nothing about us without us" is taken
   seriously in this community.

### Phase 1: pilot (3–6 months)
4. **Partner with 1–2 Orientation & Mobility instructors or a rehab centre.** Run the
   user-study protocol ([DEMO_RUNBOOK.md §5](DEMO_RUNBOOK.md)). The target from the brief:
   **40 % faster** precision tasks such as finding a door handle, versus a cane alone.
5. **Collect testimonials and data**: collisions avoided, time saved, confidence scores.
6. **Hardware:** move from laptop-in-backpack to a dedicated wearable (Jetson or similar) and
   test battery life.

### Phase 2: first revenue
7. **Sell to programs first.** Prepare an "agency pack": the outcome data, price per seat,
   training plan, a support SLA, and a privacy and data-processing summary.
8. **Launch the free safety tier** through blindness organisations, accessibility podcasts,
   and community forums. Demos at accessibility conferences are high-impact.
9. **Apply for grants and accelerators**: accessibility and assistive-technology grants,
   AWS / cloud credits for startups, disability-innovation challenges.

### Phase 3: scale
10. **Licensing** to smart-glasses and smart-cane makers.
11. **Smart-city pilots** with a transit agency: live bus times, accessible station
    guidance, and "fleet learning", where users anonymously share temporary hazards on a
    city map (the brief's future vision).

### The pitch (60 seconds)
> "A white cane can't see a branch at head height or a silent scooter. Phone apps can
> describe a photo, but only if you stop and aim. UnCane is a wearable co-pilot with two
> brains. One reacts in 50 milliseconds, on the device, to keep you from getting hurt, even
> with no signal. The other is a cloud AI that reads signs and finds your door handle, and
> it's never allowed to contradict the safety sensors. We're piloting with [partner] and
> measured [X %] faster task completion. We sell program licenses to rehab agencies, and
> safety is free for everyone."

## 8. Risks, and what you must get right before selling

| Risk | What to do |
|---|---|
| **Safety liability** | Never market it as replacing the cane, a guide dog or O&M training. Keep the safety notice, get product-liability insurance, and log incidents. |
| **Regulation** | Check whether your claims make it a **medical device** (FDA in the US, MDR in the EU, and equivalents). Wording like "prevents injury" can trigger regulation, so get advice early. |
| **Privacy** | Cameras capture bystanders. Keep "no image storage" as the default; comply with GDPR and local laws; publish a clear privacy policy. |
| **Accuracy in the real world** | Rain, night, crowds, glare. Test broadly before claims; publish limitations honestly. |
| **Battery and comfort** | Users won't wear something heavy or hot. Treat hardware ergonomics as a product feature. |
| **Cloud cost per user** | Enforce fair-use caps, use the cheaper model where quality holds, and price with margin. |
| **Accessibility of the product itself** | Setup, updates, billing and support must be fully usable with a screen reader or by voice. |

## 9. Key metrics to track

- **Safety:** collisions per hour of walking (pilot), false-alarm rate, safety-override rate
  (on the CloudWatch dashboard).
- **Speed:** edge reaction time (< 50 ms), cloud answer time (< 1.5 s).
- **Usefulness:** questions per day, "needs better view" rate, task-completion time vs. cane only.
- **Business:** cost per active user per month, conversion from free to paid, agency renewals.

---

*Related docs:* [ARCHITECTURE.md](ARCHITECTURE.md) (how it works) ·
[AWS_SETUP.md](AWS_SETUP.md) (deployment and cost) · [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md)
(demo and user-study protocol).
