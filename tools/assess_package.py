"""
tools/assess_package.py
Monitors the porch/door zone for package delivery and theft.

Flow:
  1. Vision detects a package (large box-like object) left on porch
  2. assess_package() logs it and starts a background monitor
  3. Every CHECK_INTERVAL seconds it re-checks if the package is still visible
  4. If package disappears WITHOUT a known person collecting it → STOLEN alert
  5. If package disappears after a known face → COLLECTED (normal)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

log = logging.getLogger("package")

CHECK_INTERVAL_SEC = 30       # how often to re-check porch
THEFT_WINDOW_SEC   = 600      # if gone within 10 min with no known person → suspicious


class PackageMonitor:
    def __init__(self, settings, event_logger, vision_pipeline, alerter):
        self.settings = settings
        self.db       = event_logger
        self.vision   = vision_pipeline
        self.alerter  = alerter

        # active monitors: package_id → asyncio.Task
        self._monitors: dict[int, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Called by the agent when it first detects a package
    # ------------------------------------------------------------------
    async def package_detected(self, zone: str = "front_door",
                                snapshot_path: str = "") -> int:
        pkg_id = await self.db.log_package_detected(zone=zone, snapshot_path=snapshot_path)
        log.info(f"Package #{pkg_id} detected at {zone}. Starting monitor.")

        task = asyncio.create_task(self._monitor_loop(pkg_id, zone))
        self._monitors[pkg_id] = task
        return pkg_id

    # ------------------------------------------------------------------
    # Called by the agent when it sees a known person pick something up
    # ------------------------------------------------------------------
    async def package_collected(self, package_id: int, collector_identity: str):
        log.info(f"Package #{package_id} collected by {collector_identity}.")
        await self.db.update_package_status(package_id, "collected")
        task = self._monitors.pop(package_id, None)
        if task:
            task.cancel()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------
    async def _monitor_loop(self, pkg_id: int, zone: str):
        detected_at = time.time()
        last_seen_at = time.time()
        consecutive_missing = 0

        try:
            while True:
                await asyncio.sleep(CHECK_INTERVAL_SEC)

                scene = self.vision.get_scene()
                package_still_visible = self._is_package_in_scene(scene)

                if package_still_visible:
                    last_seen_at = time.time()
                    consecutive_missing = 0
                    await self.db.update_package_status(pkg_id, "present")
                    log.debug(f"Package #{pkg_id} still present.")
                else:
                    consecutive_missing += 1
                    log.info(f"Package #{pkg_id} not visible (miss #{consecutive_missing})")

                    # Need 2 consecutive misses to confirm gone (avoids false positives)
                    if consecutive_missing >= 2:
                        time_since_detection = time.time() - detected_at

                        # Check if a known face was present recently
                        known_face_present = self._was_known_face_recent(scene)

                        if known_face_present:
                            log.info(f"Package #{pkg_id} likely collected by known person.")
                            await self.db.update_package_status(pkg_id, "collected")
                        elif time_since_detection < THEFT_WINDOW_SEC:
                            # Disappeared fast with no known face → THEFT
                            await self._trigger_theft_alert(pkg_id, zone, last_seen_at)
                        else:
                            # Disappeared after a long time — could be owner collected it
                            log.info(f"Package #{pkg_id} gone after {time_since_detection/60:.0f} min — marking unknown.")
                            await self.db.update_package_status(pkg_id, "unknown")

                        self._monitors.pop(pkg_id, None)
                        break

        except asyncio.CancelledError:
            log.debug(f"Package #{pkg_id} monitor cancelled.")

    def _is_package_in_scene(self, scene) -> bool:
        """
        Check if any detection looks like a stationary package.
        YOLO labels: 'backpack', 'suitcase', 'handbag' are closest.
        Also look for large stationary bounding boxes that haven't moved.
        """
        package_labels = {"backpack", "suitcase", "handbag", "box", "package"}
        for det in scene.detections:
            if det.label.lower() in package_labels:
                return True
        return False

    def _was_known_face_recent(self, scene) -> bool:
        """Check if a known (non-unknown) face was just seen in the last scene."""
        for face in scene.faces:
            if face.identity != "unknown":
                return True
        return False

    async def _trigger_theft_alert(self, pkg_id: int, zone: str, last_seen_at: float):
        log.warning(f"⚠ PORCH PIRACY — Package #{pkg_id} stolen from {zone}!")
        await self.db.update_package_status(pkg_id, "stolen", alert_sent=True)

        mins_ago = (time.time() - last_seen_at) / 60
        body = (
            f"A package has disappeared from your {zone} within {mins_ago:.0f} minutes "
            f"of being delivered, with no recognized person seen collecting it.\n\n"
            f"Package ID: #{pkg_id}\n"
            f"Zone: {zone}\n"
            f"Last confirmed present: {mins_ago:.0f} minutes ago\n\n"
            f"This may be porch piracy. Check the recorded clip in Google Drive."
        )

        await self.alerter.send(
            subject="⚠ Porch Piracy Detected — Package May Have Been Stolen",
            body=body,
            severity="high",
        )

    async def get_active_count(self) -> int:
        packages = await self.db.get_active_packages()
        return len(packages)
