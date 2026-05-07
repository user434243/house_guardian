"""
agent/guardian.py  (v2)
The brain of House Guardian — now with 9 tools:
  move_camera, send_alert, start_recording, log_event, patrol_sweep,
  query_history, assess_package, estimate_group_intent, recognize_vehicle
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

log = logging.getLogger("agent")


# ══════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS
# ══════════════════════════════════════════════════════════════════════
TOOLS = [
    # ── Original tools ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "move_camera",
            "description": (
                "Pan and/or tilt the surveillance camera to a specific angle. "
                "pan=0 is straight ahead, negative=left, positive=right. "
                "tilt=0 is level, negative=down, positive=up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pan_degrees":  {"type": "number", "description": "Pan angle -90 to 90"},
                    "tilt_degrees": {"type": "number", "description": "Tilt angle -30 to 60"},
                    "reason":       {"type": "string"},
                },
                "required": ["pan_degrees", "tilt_degrees", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_alert",
            "description": (
                "Send an email alert to the owner. Only for genuine security concerns. "
                "A cooldown is enforced automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject":          {"type": "string"},
                    "body":             {"type": "string"},
                    "severity":         {"type": "string", "enum": ["low", "medium", "high"]},
                    "attach_snapshot":  {"type": "boolean", "default": True},
                },
                "required": ["subject", "body", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_recording",
            "description": "Start recording a video clip. Auto-uploads to Google Drive when done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason":       {"type": "string"},
                    "duration_sec": {"type": "integer", "default": 30},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_event",
            "description": "Log a security observation to the local database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": ["person_seen", "vehicle_seen", "face_identified",
                                 "plate_read", "suspicious_behavior", "patrol_complete",
                                 "package_detected", "audio_event", "system_note"],
                    },
                    "description": {"type": "string"},
                    "metadata":    {"type": "object"},
                },
                "required": ["event_type", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patrol_sweep",
            "description": "Command camera to sweep the full 180° field of view. Use during quiet periods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {"type": "string", "enum": ["slow", "normal", "fast"], "default": "normal"},
                },
                "required": [],
            },
        },
    },

    # ── NEW: query_history ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "query_history",
            "description": (
                "Query the local database to check if a face, vehicle, plate, or sound "
                "has been seen before. Use this BEFORE deciding on threat level — "
                "a repeat unknown visitor is far more dangerous than a first-time one. "
                "Examples: check if a plate appeared before, if an unknown face is a repeat visitor, "
                "if glass-break sounds were heard recently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["face", "vehicle", "plate", "audio", "group", "summary"],
                        "description": "What kind of history to query",
                    },
                    "identifier": {
                        "type": "string",
                        "description": "Face identity, plate text, vehicle make/model, or sound class to look up",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "How many hours back to search (default 168 = 7 days)",
                        "default": 168,
                    },
                },
                "required": ["query_type"],
            },
        },
    },

    # ── NEW: assess_package ───────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "assess_package",
            "description": (
                "Register a detected package on the porch and start monitoring it for theft. "
                "Call this when YOLO detects a backpack/box/package left on the porch after "
                "a delivery person leaves. The system will monitor and alert if the package "
                "disappears without a known person collecting it (porch piracy detection)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["detected", "collected", "check_status"],
                        "description": (
                            "detected = new package seen; "
                            "collected = known person picked it up; "
                            "check_status = how many packages are being monitored"
                        ),
                    },
                    "zone":        {"type": "string", "default": "front_door"},
                    "package_id":  {"type": "integer", "description": "Required for 'collected' action"},
                    "collector":   {"type": "string", "description": "Identity of person who collected it"},
                },
                "required": ["action"],
            },
        },
    },

    # ── NEW: estimate_group_intent ────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "estimate_group_intent",
            "description": (
                "Analyze a group of 2+ people and estimate their collective intent. "
                "Considers: group size, dwell time, time of day, zone, proximity to doors/vehicles, "
                "movement patterns. Returns a threat score and intent label. "
                "Call this whenever 2+ people are detected together."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_count":    {"type": "integer"},
                    "dwell_seconds":   {"type": "number", "description": "How long they've been in the scene"},
                    "zone":            {"type": "string"},
                    "near_door":       {"type": "boolean", "default": False},
                    "near_vehicle":    {"type": "boolean", "default": False},
                    "any_unknown_face":{"type": "boolean", "default": True},
                    "behaviors":       {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of per-person behaviors: loitering, running, walking, standing",
                    },
                },
                "required": ["person_count", "zone"],
            },
        },
    },

    # ── NEW: recognize_vehicle ────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "recognize_vehicle",
            "description": (
                "Deeply analyze a detected vehicle: extract its color, class, and model fingerprint, "
                "then check if this same vehicle (not just plate) has appeared before. "
                "A silver Toyota is the same car even if the plate changes. "
                "Call this for every car/truck/motorcycle detection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_class": {
                        "type": "string",
                        "enum": ["car", "truck", "motorcycle", "bus", "van"],
                    },
                    "plate_text":  {"type": "string", "description": "Plate text if readable, else empty string"},
                    "trusted":     {"type": "boolean", "default": False},
                    "zone":        {"type": "string"},
                },
                "required": ["vehicle_class"],
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════
def build_system_prompt(settings, current_pan: float, current_tilt: float) -> str:
    kb = settings.knowledge
    zones_text = "\n".join(
        f"  - {name}: pan={z['pan']}°, tilt={z['tilt']}° ({z['description']})"
        for name, z in kb.camera_zones.items()
    )
    suspicious_text = "\n".join(f"  - {b}" for b in kb.suspicious_behaviors)

    return f"""You are House Guardian, an autonomous AI security agent protecting {kb.owner_name}'s home.
Camera is currently at pan={current_pan:.0f}°, tilt={current_tilt:.0f}°.

YOUR TOOLS AND WHEN TO USE THEM:
- move_camera: Point the camera at something interesting. You choose the exact degrees.
- send_alert: Email the owner. Only for real threats. A cooldown prevents spam.
- start_recording: Record a clip and upload to Drive. Use when a threat is in progress.
- log_event: Log any notable observation.
- patrol_sweep: Automated 180° sweep. Use when scene is quiet.
- query_history: CHECK THE DATABASE before judging threat level. A repeat unknown is far
  more dangerous than a first-time visitor. Always query before sending a high alert.
- assess_package: Detect package delivery and monitor for porch piracy.
- estimate_group_intent: Analyze groups of 2+ people for collective threat.
- recognize_vehicle: Identify vehicle color/model fingerprint and match to prior visits.

CAMERA ZONES:
{zones_text}

SUSPICIOUS BEHAVIORS:
{suspicious_text}

NORMAL HOURS: {kb.normal_hours}
TRUSTED PLATES: {', '.join(kb.trusted_vehicle_plates) or 'none set'}
OWNER: {kb.owner_name}

DECISION RULES:
1. Quiet scene → patrol_sweep or move to a zone.
2. Person detected → move_camera to track, cross-check with query_history("face").
3. Vehicle detected → recognize_vehicle, then query_history("plate" or "vehicle").
4. 2+ people together → estimate_group_intent.
5. Package left on porch after delivery person leaves → assess_package("detected").
6. Person picks up package → assess_package("collected").
7. anomaly_score > 0.5 → query_history first, then send_alert + start_recording.
8. Repeat unknown face (visit_count >= 2) → always send_alert at HIGH severity.
9. Always log_event for anything notable.
10. You can call multiple tools per cycle — chain them logically.

REASONING STYLE:
Think step by step. First observe the scene. Then check history if needed.
Then decide actions. Be decisive — missing a real threat is worse than a false positive.
You are the owner's eyes, memory, and judgment."""


# ══════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════
class GuardianAgent:
    def __init__(self, settings, vision_pipeline, servo_controller):
        self.settings = settings
        self.vision   = vision_pipeline
        self.servo    = servo_controller
        self._current_pan  = 0.0
        self._current_tilt = 0.0
        self._cycle_count  = 0
        self._alert_cooldowns: Dict[str, float] = {}

        self._client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

        # Import all tools
        from tools.alerter               import Alerter
        from tools.recorder              import VideoRecorder
        from tools.drive_uploader        import DriveUploader
        from tools.event_logger          import EventLogger
        from tools.query_history         import HistoryQuery
        from tools.assess_package        import PackageMonitor
        from tools.cross_reference_faces import FaceCrossReference
        from tools.estimate_group_intent import GroupIntentEstimator, GroupObservation
        from tools.recognize_vehicle_model import VehicleRecognizer

        self.alerter        = Alerter(settings)
        self.recorder       = VideoRecorder(settings, vision_pipeline)
        self.uploader       = DriveUploader(settings)
        self.event_logger   = EventLogger()
        self.history_query  = HistoryQuery(self.event_logger)
        self.pkg_monitor    = PackageMonitor(settings, self.event_logger, vision_pipeline, self.alerter)
        self.face_crossref  = FaceCrossReference(self.event_logger)
        self.group_estimator = GroupIntentEstimator(self.event_logger)
        self.vehicle_rec    = VehicleRecognizer(self.event_logger)

        # Keep a reference to these classes for tool execution
        self._GroupObservation = GroupObservation

        log.info("Guardian agent v2 initialized with 9 tools.")

    async def run(self, stop_event: asyncio.Event):
        log.info("Guardian agent loop starting.")
        interval = self.settings.llm.reasoning_interval_sec
        while not stop_event.is_set():
            try:
                await self._reason_cycle()
            except Exception as e:
                log.error(f"Agent cycle error: {e}", exc_info=True)
            self._cycle_count += 1
            await asyncio.sleep(interval)

    # ── Reasoning cycle ───────────────────────────────────────────────
    async def _reason_cycle(self):
        scene = self.vision.get_scene()

        # Auto-run face cross-reference and vehicle recognition
        # so history is populated even before the LLM calls tools
        await self._auto_background_analysis(scene)

        scene_ctx = scene.to_agent_context()
        system_prompt = build_system_prompt(self.settings, self._current_pan, self._current_tilt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Current scene:\n\n{scene_ctx}\n\nAnalyze and decide."},
        ]

        log.debug(f"Cycle #{self._cycle_count} | anomaly={scene.anomaly_score:.2f} | "
                  f"objs={len(scene.detections)} | faces={len(scene.faces)}")

        response = await self._client.chat.completions.create(
            model=self.settings.llm.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=self.settings.llm.max_tokens,
            temperature=self.settings.llm.temperature,
        )

        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            log.debug("Agent: no action this cycle.")
            return

        for tc in tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                log.warning(f"Bad tool args for {fn}: {tc.function.arguments}")
                continue
            log.info(f"Agent → {fn}({args})")
            await self._execute_tool(fn, args, scene)

    # ── Background analysis (runs every cycle automatically) ──────────
    async def _auto_background_analysis(self, scene):
        """
        Run face cross-reference and vehicle recognition automatically
        so the DB is always populated before the LLM reasons.
        """
        if scene.raw_frame is None:
            return

        # Cross-reference faces
        for face in scene.faces:
            if scene.raw_frame is not None:
                h, w = scene.raw_frame.shape[:2]
                x1, y1, x2, y2 = face.bbox
                face_crop = scene.raw_frame[
                    int(y1*h):int(y2*h), int(x1*w):int(x2*w)
                ]
                if face_crop.size > 0:
                    await self.face_crossref.process(
                        face_img=face_crop,
                        identity=face.identity,
                        zone=self._current_zone_name(),
                        snapshot_path=scene.snapshot_path or "",
                    )

        # Vehicle recognition (if vehicles present, run less frequently)
        if self._cycle_count % 3 == 0:
            for det in scene.detections:
                if det.label in ("car", "truck", "motorcycle", "bus", "van"):
                    plate = ""
                    trusted = False
                    for p in scene.plates:
                        plate = p.plate_text
                        trusted = p.trusted
                        break
                    await self.vehicle_rec.process(
                        frame=scene.raw_frame,
                        detection=det,
                        plate_text=plate,
                        trusted=trusted,
                        zone=self._current_zone_name(),
                        snapshot_path=scene.snapshot_path or "",
                    )

    def _current_zone_name(self) -> str:
        """Find the closest named zone to current camera position."""
        zones = self.settings.knowledge.camera_zones
        best, best_dist = "unknown", float("inf")
        for name, z in zones.items():
            dist = abs(z["pan"] - self._current_pan) + abs(z["tilt"] - self._current_tilt)
            if dist < best_dist:
                best, best_dist = name, dist
        return best

    # ── Tool execution ────────────────────────────────────────────────
    async def _execute_tool(self, name: str, args: Dict[str, Any], scene):

        if name == "move_camera":
            pan  = float(args.get("pan_degrees", 0))
            tilt = float(args.get("tilt_degrees", 0))
            await self.servo.move(pan, tilt)
            self._current_pan  = pan
            self._current_tilt = tilt
            log.info(f"Camera → pan={pan}° tilt={tilt}° | {args.get('reason','')}")

        elif name == "send_alert":
            subject  = args["subject"]
            body     = args["body"]
            severity = args.get("severity", "medium")
            snapshot = scene.snapshot_path if args.get("attach_snapshot", True) else None
            cooldown_key = subject[:40]
            now = time.time()
            if now - self._alert_cooldowns.get(cooldown_key, 0) < self.settings.alert.cooldown_sec:
                log.info(f"Alert suppressed (cooldown): {subject}")
                return
            success = await self.alerter.send(subject, body, severity, snapshot)
            if success:
                self._alert_cooldowns[cooldown_key] = now

        elif name == "start_recording":
            reason   = args.get("reason", "agent triggered")
            duration = int(args.get("duration_sec", self.settings.clip_duration_sec))
            clip_path = await self.recorder.record(duration)
            if clip_path:
                asyncio.create_task(self.uploader.upload(clip_path, reason, scene))

        elif name == "log_event":
            await self.event_logger.log(
                event_type=args["event_type"],
                description=args["description"],
                metadata=args.get("metadata", {}),
                pan=self._current_pan,
                tilt=self._current_tilt,
            )

        elif name == "patrol_sweep":
            asyncio.create_task(self._do_patrol(args.get("speed", "normal")))

        elif name == "query_history":
            result = await self.history_query.query(
                query_type=args["query_type"],
                identifier=args.get("identifier", ""),
                hours=int(args.get("hours", 168)),
            )
            # Feed result back into next cycle context via log
            log.info(f"History query result: {result.get('assessment', result)}")
            # Store for potential use in alert body (best effort)
            self._last_history_result = result

        elif name == "assess_package":
            action = args.get("action", "detected")
            if action == "detected":
                pkg_id = await self.pkg_monitor.package_detected(
                    zone=args.get("zone", "front_door"),
                    snapshot_path=scene.snapshot_path or "",
                )
                log.info(f"Package #{pkg_id} registered for monitoring.")
            elif action == "collected":
                pkg_id = args.get("package_id")
                if pkg_id:
                    await self.pkg_monitor.package_collected(
                        package_id=pkg_id,
                        collector_identity=args.get("collector", "unknown"),
                    )
            elif action == "check_status":
                count = await self.pkg_monitor.get_active_count()
                log.info(f"Active packages being monitored: {count}")

        elif name == "estimate_group_intent":
            person_count = int(args.get("person_count", 2))
            behaviors    = args.get("behaviors", [])
            zone         = args.get("zone", self._current_zone_name())

            obs = self._GroupObservation(
                person_count=person_count,
                dwell_seconds=float(args.get("dwell_seconds", 0)),
                zone=zone,
                behaviors=behaviors,
                time_of_day=self._time_of_day(),
                near_door=bool(args.get("near_door", False)),
                near_vehicle=bool(args.get("near_vehicle", False)),
                any_unknown_face=bool(args.get("any_unknown_face", True)),
            )
            track_ids = frozenset(
                d.track_id for d in scene.detections
                if d.label == "person" and d.track_id is not None
            )
            estimate = await self.group_estimator.estimate(
                obs, track_ids, snapshot_path=scene.snapshot_path or ""
            )
            log.info(f"Group intent: {estimate.reasoning}")

            # If high threat, automatically trigger recording
            if estimate.should_record and not estimate.should_alert:
                asyncio.create_task(self.recorder.record(30))
            if estimate.should_alert:
                await self.alerter.send(
                    subject=f"⚠ Suspicious Group Activity — {estimate.intent_label.title()}",
                    body=estimate.reasoning,
                    severity="high" if estimate.threat_score > 0.7 else "medium",
                    snapshot_path=scene.snapshot_path,
                )

        elif name == "recognize_vehicle":
            vclass  = args.get("vehicle_class", "car")
            plate   = args.get("plate_text", "")
            trusted = bool(args.get("trusted", False))
            zone    = args.get("zone", self._current_zone_name())

            # Find the matching detection for this vehicle
            det = next(
                (d for d in scene.detections if d.label == vclass), None
            )
            if det and scene.raw_frame is not None:
                result = await self.vehicle_rec.process(
                    frame=scene.raw_frame,
                    detection=det,
                    plate_text=plate,
                    trusted=trusted,
                    zone=zone,
                    snapshot_path=scene.snapshot_path or "",
                )
                log.info(f"Vehicle recognition: {result['assessment']}")
                # If repeat unknown vehicle, feed info back
                if result["prior_model_appearances"] >= 3 and not trusted:
                    await self.alerter.send(
                        subject=f"⚠ Repeat Unknown Vehicle: {result['fingerprint']}",
                        body=result["assessment"],
                        severity="medium",
                        snapshot_path=scene.snapshot_path,
                    )

        else:
            log.warning(f"Unknown tool: {name}")

    def _time_of_day(self) -> str:
        import datetime
        h = datetime.datetime.now().hour
        if h < 6 or h >= 22:
            return "night"
        elif h >= 19:
            return "evening"
        return "day"

    async def _do_patrol(self, speed: str):
        delays = {"slow": 2.0, "normal": 1.2, "fast": 0.5}
        delay  = delays.get(speed, 1.2)
        step   = self.settings.servo.sweep_step
        pan_min = self.settings.servo.pan_min
        pan_max = self.settings.servo.pan_max
        log.info(f"Patrol sweep starting ({speed})")
        pan = pan_min
        while pan <= pan_max:
            await self.servo.move(pan, 0)
            self._current_pan = pan
            await asyncio.sleep(delay)
            pan += step
        await self.servo.move(0, 0)
        self._current_pan  = 0
        self._current_tilt = 0
        log.info("Patrol sweep complete.")
