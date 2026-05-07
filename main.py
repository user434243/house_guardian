#!/usr/bin/env python3
"""
House Guardian — Autonomous AI Security Agent  (v2)
Entry point. Starts all subsystems in parallel:
  - Vision pipeline (camera + CV models)
  - Audio monitor (USB mic + sound classifier)
  - LLM agent reasoning loop
  - Servo controller
"""

import asyncio
import signal
import logging
from pathlib import Path

from config.settings import Settings
from vision.pipeline import VisionPipeline
from agent.guardian  import GuardianAgent
from tools.servo     import ServoController
from tools.audio_monitor import AudioMonitor
from tools.alerter   import Alerter
from tools.event_logger import EventLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/logs/guardian.log"),
    ],
)
log = logging.getLogger("main")


async def main():
    log.info("🏠 House Guardian v2 starting up...")

    # Create data directories
    for d in ["data/clips", "data/snapshots", "data/logs", "data/known_faces"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    settings     = Settings.load()
    event_logger = EventLogger()
    alerter      = Alerter(settings)

    servo  = ServoController(pan_pin=settings.servo.pan_pin, tilt_pin=settings.servo.tilt_pin)
    await servo.center()

    vision = VisionPipeline(settings)
    agent  = GuardianAgent(settings, vision, servo)
    audio  = AudioMonitor(settings, event_logger, alerter)

    stop = asyncio.Event()

    def _sig(signum, frame):
        log.info("Shutdown signal received.")
        stop.set()

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Start all subsystems as concurrent tasks
    tasks = [
        asyncio.create_task(vision.run(),         name="vision"),
        asyncio.create_task(audio.run(),           name="audio"),
        asyncio.create_task(agent.run(stop),       name="agent"),
    ]

    log.info("All subsystems running. Press Ctrl+C to stop.")
    await stop.wait()

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await servo.park()
    log.info("House Guardian stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
