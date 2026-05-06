#!/usr/bin/env python3
"""
House Guardian — Autonomous AI Security Agent
Entry point. Starts all subsystems and runs the main agent loop.
"""

import asyncio
import signal
import logging
from pathlib import Path
from config.settings import Settings
from vision.pipeline import VisionPipeline
from agent.guardian import GuardianAgent
from tools.servo import ServoController

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
    log.info("🏠 House Guardian starting up...")
    settings = Settings.load()

    servo = ServoController(
        pan_pin=settings.servo.pan_pin,
        tilt_pin=settings.servo.tilt_pin,
    )
    await servo.center()

    vision = VisionPipeline(settings)
    agent = GuardianAgent(settings, vision, servo)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _sig(signum, frame):
        log.info("Shutdown signal received.")
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    vision_task = asyncio.create_task(vision.run())
    agent_task = asyncio.create_task(agent.run(stop))

    await stop.wait()
    vision_task.cancel()
    agent_task.cancel()
    await servo.park()
    log.info("House Guardian stopped.")


if __name__ == "__main__":
    asyncio.run(main())
