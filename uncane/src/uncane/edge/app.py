"""UnCane-y AI edge runtime.

    python -m uncane.edge.app --config config/edge_config.yaml

Reflex loop (every frame, target < 50 ms): depth -> 3D hazards -> optical-flow
threats -> arbiter -> spatial beeps / urgent speech. Everything else (object
detection, cloud reasoning, speech synthesis, voice capture) runs on other
threads and can never delay a collision warning.

Keys in the debug window: 1-9 preset questions, v push-to-talk, c recalibrate,
q / Esc quit. Questions can also be typed into the terminal.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import base64
import logging
import statistics
import threading
import time
import uuid
from typing import List, Optional, Union

import cv2
import numpy as np

from ..common.schemas import CameraInfo, HazardItem, ModelAnswer, NavigateRequest, ThreatItem
from ..common.verifier import SafetyVerifier, claims_clear_path
from .arbiter import ReflexArbiter, ReflexOutput, ToneCue
from .audio_feedback import PRIORITY_CLOUD, PRIORITY_INFO, PRIORITY_SAFETY, Haptics, Speaker, SpatialBeeper
from .blackbox import BlackBox
from .cloud_client import CloudClient, CloudResult
from .commands import parse_command
from .config import load_config
from .depth_engine import DepthEngine, colorize
from .detector import DetectorThread, YoloXDetector, label_hazards, label_threats
from .frame_filter import KeyframeFilter, SharpFrameBuffer, encode_jpeg
from .guidance import GuidanceTracker
from .hazards import HazardFrame, HazardPipeline
from .monitors import FrameMonitor, FrameWatchdog
from .object_memory import ObjectMemory
from .optical_flow import ThreatTracker
from .user_profile import UserProfile
from .voice_input import make_voice

log = logging.getLogger("uncane.edge")


class Camera:
    def __init__(self, source: Union[int, str], width: int, height: int, loop: bool = True,
                 reopen_after_failures: int = 5, reopen_attempts: int = 3):
        self.source, self.loop = source, loop
        self.is_file = isinstance(source, str) and not source.lower().startswith(("rtsp://", "http://", "https://"))
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera source {source!r}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.size = (width, height)
        self._t0 = time.monotonic()
        self.reopen_after_failures = max(1, reopen_after_failures)
        self.reopen_attempts = max(0, reopen_attempts)
        self._failed_reads = 0
        self._reopens = 0
        self.reopen_total = 0
        self.exhausted = False

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.is_file and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            if not self.is_file:
                self._failed_reads += 1
                if self._failed_reads >= self.reopen_after_failures:
                    if self._reopens >= self.reopen_attempts:
                        self.exhausted = True
                    else:
                        log.warning("camera read failed repeatedly; reopening source")
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.source)
                        if self.cap.isOpened():
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                        self._reopens += 1
                        self.reopen_total += 1
                        self._failed_reads = 0
            return None, 0.0
        self._failed_reads = 0
        self._reopens = 0
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        # Files: use media time so optical-flow dt is right even if we run slower than real time.
        t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if self.is_file else time.monotonic() - self._t0
        return frame, t

    def release(self) -> None:
        self.cap.release()


class EdgeApp:
    def __init__(self, cfg: dict, headless: bool = False):
        self.cfg = cfg
        cam = cfg["camera"]
        self.camera_info = CameraInfo(width=cam["width"], height=cam["height"], hfov_deg=cam["hfov_deg"])
        self.session_id = f"{cfg['device_id']}-{uuid.uuid4().hex[:8]}"
        self.show = cfg["debug"]["show_windows"] and not headless
        self.profile = UserProfile(cfg["user"]["profile_path"], {
            "environment": cfg["environment"]["mode"],
            "speech_rate": cfg["audio"]["tts_rate"],
            "speech_volume": cfg["audio"]["tts_volume"],
        })
        self._apply_environment(self.profile.data["environment"])

        d = cfg["depth"]
        self.depth = DepthEngine(d["model_path"], d["input_size"], d["output"], d["backend"],
                                 d["opencv_target"], d["input_name"])
        self.hazards = HazardPipeline(cfg, d["output"], cam["width"], cam["height"])
        f = cfg["flow"]
        self.tracker = ThreatTracker(cam["hfov_deg"], f["proc_width"], f["max_corners"], f["redetect_every"],
                                     f["ttc_warning_s"], f["ttc_critical_s"], f["min_cluster_points"],
                                     f["persist_frames"]) if f["enabled"] else None
        det = cfg["detector"]
        self.detector: Optional[DetectorThread] = None
        if det["enabled"]:
            try:
                self.detector = DetectorThread(
                    YoloXDetector(det["model_path"], det["conf_threshold"], det["nms_threshold"],
                                  det["input_size"]), det["hz"]).start()
            except cv2.error as exc:
                log.warning("object detector disabled: %s", exc)
        a = cfg["arbiter"]
        crowd = cfg["crowd"]
        self.arbiter = ReflexArbiter(
            a["confirm_frames"], a["window"], a["release_frames"], a["speech_cooldown_s"],
            quiet_mode=self.profile.data["quiet_mode"], crowd_mode=crowd["enabled"],
            crowd_alert_distance_m=crowd["person_alert_distance_m"],
            dropoff_confirm_frames=a["dropoff_confirm_frames"], dropoff_window=a["dropoff_window"],
        )
        au = cfg["audio"]
        self.beeper = SpatialBeeper(au["volume"] * self.profile.data["speech_volume"], au["enabled"])
        self.speaker = Speaker(self.profile.data["speech_rate"], au["tts_voice"], au["enabled"],
                               self.profile.data["speech_volume"])
        self.haptics = Haptics(au["haptics_serial_port"])
        g = cfg["guidance"]
        self.guidance = GuidanceTracker(g["timeout_s"], g["lost_after_s"], g["arrival_distance_m"])
        self.guidance_interval_s = g["update_interval_s"]
        m = cfg["monitor"]
        self.monitor = FrameMonitor(m["message_cooldown_s"]) if m["enabled"] else None
        b = cfg["blackbox"]
        self.blackbox = BlackBox(b["path"], b["retention_frames"], b["save_snapshots"]) if b["enabled"] else None
        self.watchdog = None
        self.object_memory = ObjectMemory()
        self.last_camera_frame: Optional[np.ndarray] = None
        self._last_guidance_update = 0.0
        self._guidance_bearing: Optional[float] = None
        self._retry_queries = {}
        k = cfg["keyframes"]
        self.keyframes = KeyframeFilter(k["ssim_threshold"], k["min_interval_s"])
        self.sharp = SharpFrameBuffer(k["buffer_frames"])
        self.verifier = SafetyVerifier(threat_ttc_limit_s=f["ttc_warning_s"])

        c = cfg["cloud"]
        self.cloud: Optional[CloudClient] = None
        if c["enabled"] and c["api_url"]:
            self.cloud = CloudClient(c["api_url"], c["api_key"], self._on_cloud_result, c["timeout_s"],
                                     c["offline_after_failures"], c["verify_ssl"])
        elif c["enabled"]:
            log.warning("cloud.api_url is empty - running edge-only (set UNCANE_API_URL)")
        self._announced_offline = False
        self._answered_offline: set = set()

        v = cfg["voice"]
        self.voice = make_voice(v["backend"], self.handle_utterance, v["vosk_model_path"], v["record_seconds"]) \
            if v["backend"] != "none" else None
        self.presets = {str(key): q for key, q in (v.get("presets") or {}).items()}

        # Optical flow runs concurrently with depth inference (both release the GIL).
        self._flow_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="flow")
        self._lock = threading.Lock()
        self.current_hazards: List[HazardItem] = []
        self.current_threats: List[ThreatItem] = []
        self.last_frame: Optional[HazardFrame] = None
        self.reflex_ms: List[float] = []
        self.last_answer = ""
        self.last_cloud_ms = 0.0

    def _apply_environment(self, mode: str) -> None:
        mode = "indoor" if mode == "indoor" else "outdoor"
        env = self.cfg["environment"]
        self.cfg["hazards"]["critical_m"] = env[f"{mode}_critical_m"]
        self.cfg["hazards"]["warning_m"] = env[f"{mode}_warning_m"]

    def handle_utterance(self, text: str) -> None:
        command = parse_command(text)
        if command is None:
            self.ask(text)
            return
        value = command.value
        if command.name == "repeat":
            self.speaker.say(self.last_answer or self.speaker.last_spoken or "Nothing to repeat.", PRIORITY_INFO)
        elif command.name == "stop_speech":
            self.speaker.stop()
        elif command.name == "quiet":
            enabled = not self.profile.data["quiet_mode"] if value == "toggle" else value == "on"
            self.profile.update(quiet_mode=enabled)
            self.arbiter.set_modes(quiet_mode=enabled)
            self.speaker.say("Quiet mode on." if enabled else "Quiet mode off.", PRIORITY_INFO)
        elif command.name == "environment":
            self.profile.update(environment=value)
            self._apply_environment(value)
            self.speaker.say(f"{value.capitalize()} mode.", PRIORITY_INFO)
        elif command.name == "speech_rate":
            rate = self.profile.data["speech_rate"] + (20 if value == "faster" else -20)
            self.profile.update(speech_rate=max(80, min(300, rate)))
            self.speaker.set_controls(rate=self.profile.data["speech_rate"])
            self.speaker.say(f"Speech rate {value}.", PRIORITY_INFO)
        elif command.name == "speech_volume":
            volume = self.profile.data["speech_volume"] + (0.1 if value == "up" else -0.1)
            self.profile.update(speech_volume=max(0.0, min(1.0, volume)))
            self.speaker.set_controls(volume=self.profile.data["speech_volume"])
            self.beeper.set_volume(self.cfg["audio"]["volume"] * self.profile.data["speech_volume"])
            self.speaker.say(f"Volume {value}.", PRIORITY_INFO)
        elif command.name == "read":
            query = "Read aloud: " + (value or "Read the visible text verbatim.")
            self.ask(query)
        elif command.name == "add_note":
            self.profile.add_note(value)
            self.speaker.say("Note saved.", PRIORITY_INFO)
        elif command.name == "show_notes":
            notes = self.profile.data["notes"]
            self.speaker.say(". ".join(notes) if notes else "You have no saved notes.", PRIORITY_INFO)
        elif command.name == "clear_notes":
            self.profile.clear_notes()
            self.speaker.say("Notes cleared.", PRIORITY_INFO)
        elif command.name == "status":
            mode = "quiet" if self.profile.data["quiet_mode"] else self.profile.data["environment"]
            cloud = "online" if self.cloud and self.cloud.online else "offline"
            self.speaker.say(f"Device ready. {mode} mode. Cloud {cloud}. "
                             f"{len(self.current_hazards)} active hazards.", PRIORITY_INFO)
        elif command.name == "recalibrate":
            self.hazards.calibrator.frames_ok = 0
            self.hazards.calibrator.a = None
            self.hazards.calibrator.b = None
            self.speaker.say("Floor calibration restarted.", PRIORITY_INFO)
        elif command.name == "stop_guidance":
            self.guidance.stop()
            self._guidance_bearing = None
            self.speaker.say("Target guidance stopped.", PRIORITY_INFO)
        elif command.name == "find_object":
            sighting = self.object_memory.find(value)
            self.speaker.say(self.object_memory.describe(sighting), PRIORITY_INFO)
        elif command.name == "feedback":
            self.profile.update(feedback_count=self.profile.data["feedback_count"] + 1)
            if self.blackbox:
                self.blackbox.record("feedback", {"count": self.profile.data["feedback_count"]},
                                     self.last_camera_frame, export=True)
            self.speaker.say("Feedback recorded.", PRIORITY_INFO)
        elif command.name == "export_blackbox":
            if self.blackbox:
                path = self.blackbox.export("manual", self.last_camera_frame)
                self.speaker.say(f"Event log saved to {path.name}.", PRIORITY_INFO)
            else:
                self.speaker.say("Event logging is disabled.", PRIORITY_INFO)

    def _on_watchdog(self) -> None:
        log.error("frame processing stalled")
        if self.blackbox:
            self.blackbox.record("watchdog", export=True, frame=self.last_camera_frame)
        self.speaker.say("Camera processing stalled. Obstacle alerts may be delayed.", PRIORITY_SAFETY)

    # ---------------------------------------------------------------- reflex
    def process_frame(self, frame: np.ndarray, t: float) -> ReflexOutput:
        t0 = time.perf_counter()
        flow = self._flow_pool.submit(self.tracker.update, frame, t) if self.tracker else None
        depth = self.depth.infer(frame)
        hf = self.hazards.process(depth)
        threats = flow.result() if flow else []
        if self.detector is not None and time.monotonic() - self.detector.latest_t < 1.0:
            label_hazards(hf.hazards, self.detector.latest, self.camera_info.hfov_deg)
            label_threats(threats, self.detector.latest, self.camera_info.hfov_deg)
        out = self.arbiter.update(hf.hazards, threats)
        tone = out.tone
        if tone is None and self.guidance.active and self._guidance_bearing is not None:
            tone = ToneCue(max(-1.0, min(1.0, self._guidance_bearing / 45.0)),
                           880.0, 0.35, "INFO", "guidance")
        self.beeper.set_cue(tone)
        self.haptics.update(out.tone)
        for text in out.speech:
            self.speaker.say(text, PRIORITY_SAFETY)
        self.reflex_ms.append((time.perf_counter() - t0) * 1000)
        self.reflex_ms = self.reflex_ms[-300:]
        with self._lock:
            self.current_hazards, self.current_threats = out.active_hazards, out.active_threats
            self.last_frame = hf
        hf.debug["depth"] = depth
        if self.blackbox:
            self.blackbox.record("reflex", {
                "hazards": [h.__dict__.copy() for h in out.active_hazards],
                "threats": [t.__dict__.copy() for t in out.active_threats],
                "tone": out.tone.__dict__.copy() if out.tone else None,
                "reflex_ms": round(self.reflex_ms[-1], 2),
            })
        return out

    # ----------------------------------------------------------------- cloud
    def ask(self, query: str, mode: str = "query", _retry: bool = False) -> None:
        """Send the sharpest recent frame + edge context to the cloud."""
        if mode == "query" and not _retry:
            self._retry_queries[query] = 0
        best = self.sharp.best()
        if best is None:
            return
        answered_offline = False
        if self.cloud is None or not self.cloud.online:
            self._offline_answer(query)
            answered_offline = True
            if self.cloud is None:
                return
        frame, _ = best
        k = self.cfg["keyframes"]
        jpeg = encode_jpeg(frame, k["jpeg_quality"], k["max_side_px"])
        with self._lock:
            hazards, threats = list(self.current_hazards), list(self.current_threats)
            grid = self.last_frame.depth_grid if self.last_frame else None
        req = NavigateRequest(
            device_id=self.cfg["device_id"], session_id=self.session_id, timestamp=time.time(),
            query=query, image_jpeg_b64=base64.b64encode(jpeg).decode("ascii"), camera=self.camera_info,
            hazards=hazards, threats=threats, depth_grid=grid, mode=mode,
            location=self.cfg["cloud"].get("static_location"),
        )
        if answered_offline:
            self._answered_offline.add(req.timestamp)   # still send: it doubles as a reconnect probe
        elif mode == "query":
            self.speaker.say("Looking.", PRIORITY_INFO)
            log.info("asking cloud: %r (%d KB image)", query, len(jpeg) // 1024)
        self.cloud.submit(req)

    def _offline_answer(self, query: str) -> None:
        """Graceful degradation: answer from the on-device detector alone."""
        dets = self.detector.latest if self.detector else []
        if dets:
            from ..common.spatial import bearing_from_x, clock_from_bearing

            top = sorted(dets, key=lambda d: -d.area)[:3]
            parts = [f"{d.label} at {clock_from_bearing(bearing_from_x(d.cx, self.camera_info.hfov_deg))} o'clock"
                     for d in top]
            msg = "Cloud assistant is offline. I can see " + ", ".join(parts) + "."
        else:
            msg = "Cloud assistant is offline. Obstacle alerts are still active."
        self.speaker.say(msg, PRIORITY_CLOUD)

    def _on_cloud_result(self, res: CloudResult) -> None:
        if res.error:
            if self.cloud and not self.cloud.online and not self._announced_offline:
                self._announced_offline = True
                self.speaker.say("Connection lost. Obstacle alerts are still active.", PRIORITY_CLOUD)
            if res.request.mode == "query" and res.request.timestamp not in self._answered_offline:
                self._offline_answer(res.request.query)
            self._answered_offline.discard(res.request.timestamp)
            return
        self._answered_offline.discard(res.request.timestamp)
        self._announced_offline = False
        resp = res.response
        self.last_cloud_ms = res.round_trip_ms
        age = time.monotonic() - res.sent_at
        if age > self.cfg["cloud"]["max_answer_age_s"]:
            log.info("dropping stale answer (%.1fs old)", age)
            self._retry_queries.pop(res.request.query, None)
            return
        if resp.needs_better_view and res.request.mode == "query":
            attempts = self._retry_queries.get(res.request.query, 0)
            if attempts < 1:
                self._retry_queries[res.request.query] = attempts + 1
                self.speaker.say("I could not get a clear view. Trying one more recent frame.", PRIORITY_INFO)
                self.ask(res.request.query, _retry=True)
                return
        self._retry_queries.pop(res.request.query, None)
        if not resp.speech.strip():
            return
        # Second verification pass against what the sensors see *now*.
        with self._lock:
            hazards, threats = list(self.current_hazards), list(self.current_threats)
        answer = ModelAnswer(speech=resp.speech, claims_path_clear=claims_clear_path(resp.speech))
        checked = self.verifier.verify(answer, hazards, threats, resp.target)
        if checked.overridden:
            log.info("edge re-verification changed answer: %s", checked.reasons)
        self.last_answer = checked.speech
        log.info("cloud answer in %.0f ms: %s", res.round_trip_ms, checked.speech)
        self.speaker.say(checked.speech, PRIORITY_CLOUD)
        guidance_cfg = self.cfg["guidance"]
        if guidance_cfg["enabled"] and resp.target and resp.target.bbox:
            self.guidance.start(resp.target.label, resp.target.bbox, self.last_camera_frame)
            self._last_guidance_update = 0.0
            if self.blackbox:
                self.blackbox.record("guidance_started", {"label": resp.target.label})

    # ------------------------------------------------------------------ main
    def run(self, max_frames: Optional[int] = None, startup_query: Optional[str] = None) -> None:
        cam_cfg = self.cfg["camera"]
        camera = Camera(cam_cfg["source"], cam_cfg["width"], cam_cfg["height"], cam_cfg["loop_video"],
                        cam_cfg["reopen_after_failures"], cam_cfg["reopen_attempts"])
        if self.cfg["monitor"]["enabled"]:
            self.watchdog = FrameWatchdog(self.cfg["monitor"]["frame_stall_s"], self._on_watchdog).start()
        self.speaker.say("UnCane ready.", PRIORITY_INFO)
        n = 0
        last_reopen_total = camera.reopen_total
        try:
            while max_frames is None or n < max_frames:
                frame, t = camera.read()
                if frame is None:
                    if camera.is_file or camera.exhausted:
                        break
                    continue
                if camera.reopen_total != last_reopen_total:
                    last_reopen_total = camera.reopen_total
                    self.speaker.say("Camera connection restored.", PRIORITY_INFO)
                    if self.blackbox:
                        self.blackbox.record("camera_reopened", {"attempt": last_reopen_total},
                                             frame, export=True)
                n += 1
                self.last_camera_frame = frame
                out = self.process_frame(frame, t)
                if self.watchdog:
                    self.watchdog.beat()
                self.sharp.push(frame, t)
                if self.detector is not None:
                    self.detector.submit(frame)
                    if self.detector.latest_t > 0 and self.detector.latest_t != getattr(self, "_last_memory_update_t", 0.0):
                        metric_depth = self.last_frame.metric_depth if self.last_frame else None
                        self.object_memory.update(self.detector.latest, self.camera_info.hfov_deg, metric_depth)
                        self._last_memory_update_t = self.detector.latest_t
                if self.monitor:
                    calibrated = self.last_frame.calibrated if self.last_frame else False
                    for message in self.monitor.inspect(frame, calibrated):
                        self.speaker.say(message, PRIORITY_INFO)
                if self.guidance.active and time.monotonic() - self._last_guidance_update >= self.guidance_interval_s:
                    self._last_guidance_update = time.monotonic()
                    metric_depth = self.last_frame.metric_depth if self.last_frame else None
                    detections = self.detector.latest if self.detector else []
                    cue = self.guidance.update(frame, detections, self.camera_info.hfov_deg, metric_depth)
                    if cue:
                        self._guidance_bearing = cue.bearing_deg
                        self.speaker.say(cue.message, PRIORITY_INFO)
                        if self.blackbox:
                            self.blackbox.record("guidance", {"state": cue.state, "message": cue.message})
                if startup_query and n == 10:
                    self.ask(startup_query)
                if (self.cloud and self.cfg["cloud"]["ambient_mode"] and not self.cloud.in_flight
                        and self.keyframes.should_send(frame)):
                    self.ask("", mode="ambient")
                if self.show and not self._draw(frame, out):
                    break
                if n % 150 == 0:
                    log.info("reflex p50 %.1f ms / p95 %.1f ms", *self.latency_stats())
        finally:
            camera.release()
            if self.watchdog:
                self.watchdog.stop()
            self.beeper.close()
            self.speaker.close()
            if self.detector:
                self.detector.stop()
            if self.show:
                cv2.destroyAllWindows()

    def latency_stats(self):
        xs = sorted(self.reflex_ms) or [0.0]
        return statistics.median(xs), xs[min(len(xs) - 1, int(0.95 * len(xs)))]

    # ----------------------------------------------------------------- debug
    def _draw(self, frame: np.ndarray, out: ReflexOutput) -> bool:
        vis = frame.copy()
        h, w = vis.shape[:2]
        for x in (w // 3, 2 * w // 3):
            cv2.line(vis, (x, 0), (x, h), (80, 80, 80), 1)
        if self.detector:
            for d in self.detector.latest:
                x0, y0, x1, y1 = (int(d.box[0] * w), int(d.box[1] * h), int(d.box[2] * w), int(d.box[3] * h))
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 1)
                cv2.putText(vis, f"{d.label} {d.score:.2f}", (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        if self.tracker is not None:
            s = w / self.tracker.proc_width
            for p in self.tracker.debug.movers:
                cv2.circle(vis, (int(p[0] * s), int(p[1] * s)), 3, (0, 0, 255), -1)
        y = 24
        p50, p95 = self.latency_stats()
        calib = "metric" if (self.last_frame and self.last_frame.calibrated) else "calibrating"
        cloud = "edge-only" if not self.cloud else ("online" if self.cloud.online else "OFFLINE")
        cv2.putText(vis, f"reflex {p50:.0f}/{p95:.0f} ms p50/p95 | depth {calib} | cloud {cloud}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for hz in out.active_hazards:
            y += 20
            color = (0, 0, 255) if hz.urgency == "CRITICAL" else (0, 200, 255)
            cv2.putText(vis, f"{hz.urgency} {hz.describe()}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for th in out.active_threats:
            y += 20
            cv2.putText(vis, f"THREAT {th.label or ''} {th.zone} ttc {th.ttc_s:.1f}s", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if self.last_answer:
            cv2.putText(vis, self.last_answer[:90], (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("UnCane-y AI - edge", vis)
        if self.last_frame is not None and "depth" in self.last_frame.debug:
            cv2.imshow("depth", colorize(self.last_frame.debug["depth"], self.hazards.metric_model, (w // 2, h // 2)))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if chr(key) in self.presets:
            self.ask(self.presets[chr(key)])
        elif key == ord("v") and self.voice is not None:
            self.voice.push_to_talk()
        elif key == ord("c"):
            self.hazards.calibrator.frames_ok = 0
            self.hazards.calibrator.a = None
            self.hazards.calibrator.b = None
        return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="UnCane-y AI edge runtime")
    ap.add_argument("--config", default="config/edge_config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--source", help="camera index, video file, or stream URL")
    ap.add_argument("--headless", action="store_true", help="no debug windows")
    ap.add_argument("--frames", type=int, help="stop after N frames")
    ap.add_argument("--ask", help="send this question to the cloud after start-up")
    args = ap.parse_args(argv)
    overrides = list(args.set) + ([f"camera.source={args.source}"] if args.source else [])
    cfg = load_config(args.config, overrides)
    logging.basicConfig(level=cfg["debug"]["log_level"], format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = EdgeApp(cfg, headless=args.headless)
    app.run(max_frames=args.frames, startup_query=args.ask)
    p50, p95 = app.latency_stats()
    print(f"reflex latency: p50 {p50:.1f} ms, p95 {p95:.1f} ms over {len(app.reflex_ms)} frames")


if __name__ == "__main__":
    main()
