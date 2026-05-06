"""
tools/recorder.py
Records video clips from the live camera feed.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import cv2

log = logging.getLogger("recorder")


class VideoRecorder:
    def __init__(self, settings, vision_pipeline):
        self.settings = settings
        self.vision = vision_pipeline
        self._recording = False

    async def record(self, duration_sec: int = 30) -> Optional[str]:
        if self._recording:
            log.info("Already recording, skipping.")
            return None

        self._recording = True
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"data/clips/clip_{ts}.mp4"

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._record_sync, output_path, duration_sec
            )
            return result
        except Exception as e:
            log.error(f"Recording error: {e}")
            return None
        finally:
            self._recording = False

    def _record_sync(self, output_path: str, duration_sec: int) -> Optional[str]:
        fps = self.settings.camera.fps
        w, h = self.settings.camera.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        start = time.time()
        frame_count = 0

        log.info(f"Recording {duration_sec}s clip → {output_path}")

        while time.time() - start < duration_sec:
            scene = self.vision.get_scene()
            if scene.raw_frame is not None:
                frame = scene.raw_frame.copy()
                # Overlay HUD
                frame = self._draw_hud(frame, scene)
                writer.write(frame)
                frame_count += 1
            time.sleep(1.0 / fps)

        writer.release()
        log.info(f"Clip saved: {output_path} ({frame_count} frames)")
        return output_path

    def _draw_hud(self, frame, scene):
        """Draw minimal overlay on recorded frames."""
        h, w = frame.shape[:2]
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        # Timestamp
        cv2.putText(frame, ts, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Anomaly indicator
        if scene.anomaly_score > 0.3:
            color = (0, 0, 255) if scene.anomaly_score > 0.6 else (0, 128, 255)
            cv2.putText(frame, f"ALERT {scene.anomaly_score:.0%}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        # Draw bounding boxes
        for det in scene.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame,
                          (int(x1 * w), int(y1 * h)),
                          (int(x2 * w), int(y2 * h)),
                          (0, 255, 0), 1)
            cv2.putText(frame, f"{det.label} {det.confidence:.0%}",
                        (int(x1 * w), int(y1 * h) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return frame
