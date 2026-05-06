"""
agent/guardian.py
The brain of House Guardian.
Every N seconds it reads the latest scene state, builds a prompt,
calls the local LLM, parses its tool calls, and executes them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI   # openai SDK works against Ollama's /v1 endpoint too

log = logging.getLogger("agent")


# ---------------------------------------------------------------------------
# Tool schemas — sent to the LLM as JSON schema
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_camera",
            "description": (
                "Pan and/or tilt the surveillance camera to point at a specific angle. "
                "Use this when you want to investigate an area, track a subject, or patrol. "
                "pan=0 is straight ahead, negative=left, positive=right. "
                "tilt=0 is level, negative=down, positive=up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pan_degrees": {"type": "number", "description": "Pan angle in degrees (-90 to 90)"},
                    "tilt_degrees": {"type": "number", "description": "Tilt angle in degrees (-30 to 60)"},
                    "reason": {"type": "string", "description": "Brief reason for moving the camera"},
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
                "Send an email alert to the owner with a description of the security event. "
                "Only call this when there is a genuine security concern — unknown person, "
                "suspicious behaviour, untrusted vehicle, or activity outside normal hours. "
                "Do NOT spam; there is a cooldown enforced automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Short alert subject line"},
                    "body": {"type": "string", "description": "Detailed description of the event"},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How urgent is this alert?",
                    },
                    "attach_snapshot": {
                        "type": "boolean",
                        "description": "Whether to attach the latest camera snapshot",
                        "default": True,
                    },
                },
                "required": ["subject", "body", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_recording",
            "description": (
                "Start recording a video clip right now. The clip will be automatically "
                "uploaded to Google Drive with event metadata when it finishes. "
                "Call this when a security event is in progress."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why recording was triggered"},
                    "duration_sec": {
                        "type": "integer",
                        "description": "How many seconds to record (default 30)",
                        "default": 30,
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_event",
            "description": (
                "Log a security observation to the local database. "
                "Use this for anything worth recording — people arriving, vehicles seen, "
                "nothing suspicious found during patrol, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": ["person_seen", "vehicle_seen", "face_identified", "plate_read",
                                 "suspicious_behavior", "patrol_complete", "system_note"],
                    },
                    "description": {"type": "string"},
                    "metadata": {"type": "object", "description": "Any extra structured data"},
                },
                "required": ["event_type", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patrol_sweep",
            "description": (
                "Command the camera to do a slow sweep across the full 180-degree field of view. "
                "Use this during quiet periods when there is nothing to focus on, "
                "to proactively monitor all zones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "string",
                        "enum": ["slow", "normal", "fast"],
                        "description": "Speed of the sweep",
                        "default": "normal",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
def build_system_prompt(settings, current_pan: float, current_tilt: float) -> str:
    kb = settings.knowledge
    zones_text = "\n".join(
        f"  - {name}: pan={z['pan']}°, tilt={z['tilt']}° ({z['description']})"
        for name, z in kb.camera_zones.items()
    )
    suspicious_text = "\n".join(f"  - {b}" for b in kb.suspicious_behaviors)

    return f"""You are House Guardian, an autonomous AI security agent protecting {kb.owner_name}'s home.
You have a camera mounted on a servo that can pan -90° to +90° and tilt -30° to +60°.
The camera is currently at pan={current_pan:.0f}°, tilt={current_tilt:.0f}°.

YOUR RESPONSIBILITIES:
1. Continuously monitor for security threats using the scene data provided.
2. Decide where to point the camera — focus on areas of interest, track suspects, patrol when quiet.
3. Identify people (known vs unknown), vehicles, license plates, and behaviors.
4. Analyze behaviors for suspicious patterns and use your judgment.
5. Alert the owner immediately if there is a genuine security concern.
6. Record video clips of security events and upload to Google Drive.
7. Log all significant observations to the database.

CAMERA ZONES:
{zones_text}

SUSPICIOUS BEHAVIORS TO WATCH FOR:
{suspicious_text}

NORMAL HOURS: {kb.normal_hours}
Any person detected outside these hours is automatically suspicious.

TRUSTED PLATES: {', '.join(kb.trusted_vehicle_plates) or 'none configured'}

DECISION GUIDELINES:
- If the scene is quiet and nothing is happening → call patrol_sweep or move_camera to check zones.
- If a person is detected → move_camera to track them and assess their behavior.
- If anomaly_score > 0.5 → always send_alert AND start_recording.
- If anomaly_score 0.3–0.5 → start_recording and consider alert.
- If an unknown face is detected → log it and send_alert at medium severity.
- If an untrusted vehicle is idling → log it and send_alert.
- Always call log_event for meaningful observations.
- You can call multiple tools in one cycle if needed.
- Be decisive. False negatives (missing threats) are worse than false positives.

You are the owner's eyes. Think carefully, act swiftly."""


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------
class GuardianAgent:
    def __init__(self, settings, vision_pipeline, servo_controller):
        self.settings = settings
        self.vision = vision_pipeline
        self.servo = servo_controller
        self._alert_cooldowns: Dict[str, float] = {}
        self._current_pan = 0.0
        self._current_tilt = 0.0
        self._cycle_count = 0

        self._client = AsyncOpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

        # Import tools lazily to avoid circular imports
        from tools.alerter import Alerter
        from tools.recorder import VideoRecorder
        from tools.drive_uploader import DriveUploader
        from tools.event_logger import EventLogger

        self.alerter = Alerter(settings)
        self.recorder = VideoRecorder(settings, vision_pipeline)
        self.uploader = DriveUploader(settings)
        self.event_logger = EventLogger()

        log.info("Guardian agent initialized.")

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

    async def _reason_cycle(self):
        scene = self.vision.get_scene()
        scene_ctx = scene.to_agent_context()
        system_prompt = build_system_prompt(
            self.settings, self._current_pan, self._current_tilt
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current scene observation:\n\n{scene_ctx}\n\nAnalyze this scene and decide what to do."},
        ]

        log.debug(f"Cycle #{self._cycle_count} | anomaly={scene.anomaly_score:.2f} | "
                  f"objects={len(scene.detections)} | faces={len(scene.faces)}")

        response = await self._client.chat.completions.create(
            model=self.settings.llm.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=self.settings.llm.max_tokens,
            temperature=self.settings.llm.temperature,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []

        if not tool_calls:
            # Agent decided nothing to do — that's fine
            log.debug("Agent decided no action needed this cycle.")
            return

        for tc in tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                log.warning(f"Could not parse tool args for {fn}: {tc.function.arguments}")
                continue

            log.info(f"Agent calls tool: {fn}({args})")
            await self._execute_tool(fn, args, scene)

    async def _execute_tool(self, name: str, args: Dict[str, Any], scene):
        if name == "move_camera":
            pan = float(args.get("pan_degrees", 0))
            tilt = float(args.get("tilt_degrees", 0))
            reason = args.get("reason", "")
            await self.servo.move(pan, tilt)
            self._current_pan = pan
            self._current_tilt = tilt
            log.info(f"Camera → pan={pan}° tilt={tilt}° | reason: {reason}")

        elif name == "send_alert":
            subject = args["subject"]
            body = args["body"]
            severity = args.get("severity", "medium")
            attach = args.get("attach_snapshot", True)
            snapshot = scene.snapshot_path if attach else None

            cooldown_key = subject[:40]
            now = time.time()
            last = self._alert_cooldowns.get(cooldown_key, 0)
            if now - last < self.settings.alert.cooldown_sec:
                log.info(f"Alert suppressed (cooldown): {subject}")
                return

            success = await self.alerter.send(subject, body, severity, snapshot)
            if success:
                self._alert_cooldowns[cooldown_key] = now
                log.info(f"Alert sent [{severity}]: {subject}")

        elif name == "start_recording":
            reason = args.get("reason", "agent triggered")
            duration = int(args.get("duration_sec", self.settings.clip_duration_sec))
            clip_path = await self.recorder.record(duration)
            if clip_path:
                log.info(f"Recording saved: {clip_path}")
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
            speed = args.get("speed", "normal")
            asyncio.create_task(self._do_patrol(speed))

        else:
            log.warning(f"Unknown tool called by agent: {name}")

    async def _do_patrol(self, speed: str):
        delays = {"slow": 2.0, "normal": 1.2, "fast": 0.5}
        delay = delays.get(speed, 1.2)
        step = self.settings.servo.sweep_step
        pan_min = self.settings.servo.pan_min
        pan_max = self.settings.servo.pan_max

        log.info(f"Starting patrol sweep ({speed})")
        pan = pan_min
        while pan <= pan_max:
            await self.servo.move(pan, 0)
            self._current_pan = pan
            await asyncio.sleep(delay)
            pan += step

        # Return to center
        await self.servo.move(0, 0)
        self._current_pan = 0
        self._current_tilt = 0
        log.info("Patrol sweep complete.")
