"""
vision/pipeline.py
Continuous vision pipeline. Runs in its own async task.
Produces a shared SceneState object consumed by the agent.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading

import cv2
import numpy as np

log = logging.getLogger("vision")


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple            # (x1, y1, x2, y2) normalized 0-1
    track_id: Optional[int] = None


@dataclass
class FaceMatch:
    bbox: tuple
    identity: str          # "unknown" or name from known_faces
    confidence: float
    emotion: Optional[str] = None


@dataclass
class PlateDetection:
    plate_text: str
    confidence: float
    bbox: tuple
    trusted: bool = False


@dataclass
class PoseKeypoint:
    track_id: int
    keypoints: List[tuple]   # list of (x, y, confidence)
    action_hint: str = ""    # e.g. "crouching", "running", "standing"


@dataclass
class SceneState:
    """Latest fused snapshot of everything the vision system knows."""
    timestamp: float = field(default_factory=time.time)
    frame_idx: int = 0
    detections: List[Detection] = field(default_factory=list)
    faces: List[FaceMatch] = field(default_factory=list)
    plates: List[PlateDetection] = field(default_factory=list)
    poses: List[PoseKeypoint] = field(default_factory=list)
    raw_frame: Optional[np.ndarray] = None
    snapshot_path: Optional[str] = None   # JPEG written to disk when interesting
    anomaly_score: float = 0.0            # 0.0 = nothing, 1.0 = very suspicious
    anomaly_reasons: List[str] = field(default_factory=list)

    def to_agent_context(self) -> str:
        """Serialize to natural language for the LLM prompt."""
        lines = [
            f"Scene timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"Frame #{self.frame_idx}",
        ]

        if self.detections:
            obj_summary = {}
            for d in self.detections:
                obj_summary[d.label] = obj_summary.get(d.label, 0) + 1
            lines.append("Objects detected: " + ", ".join(f"{v}x {k}" for k, v in obj_summary.items()))
        else:
            lines.append("Objects detected: none")

        if self.faces:
            for f in self.faces:
                emo = f" [{f.emotion}]" if f.emotion else ""
                lines.append(f"Face: {f.identity} (conf={f.confidence:.2f}){emo}")
        else:
            lines.append("Faces: none detected")

        if self.plates:
            for p in self.plates:
                trust = "TRUSTED" if p.trusted else "UNKNOWN"
                lines.append(f"License plate: {p.plate_text} ({trust}, conf={p.confidence:.2f})")
        else:
            lines.append("License plates: none detected")

        if self.poses:
            for pose in self.poses:
                lines.append(f"Person #{pose.track_id} pose: {pose.action_hint or 'standing/unknown'}")

        if self.anomaly_score > 0:
            lines.append(f"Anomaly score: {self.anomaly_score:.2f}")
            for reason in self.anomaly_reasons:
                lines.append(f"  ⚠ {reason}")

        return "\n".join(lines)


class VisionPipeline:
    def __init__(self, settings):
        self.settings = settings
        self.scene: SceneState = SceneState()
        self._lock = threading.Lock()
        self._frame_idx = 0
        self._tracker: Dict[int, List[tuple]] = {}   # track_id -> list of (x,y,ts)

        self._load_models()

    def _load_models(self):
        from ultralytics import YOLO
        log.info("Loading YOLO object detection model...")
        self._yolo = YOLO(self.settings.vision.yolo_model)

        log.info("Loading YOLO pose model...")
        self._pose = YOLO(self.settings.vision.pose_model)

        log.info("Vision models loaded.")

        # plate reader
        self._plate_reader = None
        if self.settings.vision.plate_backend == "easyocr":
            try:
                import easyocr
                self._plate_reader = easyocr.Reader(["en"], gpu=False)
                log.info("EasyOCR plate reader ready.")
            except ImportError:
                log.warning("EasyOCR not installed, plate detection disabled.")

    def get_scene(self) -> SceneState:
        with self._lock:
            return self.scene

    def _save_snapshot(self, frame: np.ndarray) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"data/snapshots/snap_{ts}.jpg"
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path

    def _analyze_trajectory(self, track_id: int, x: float, y: float) -> str:
        history = self._tracker.setdefault(track_id, [])
        history.append((x, y, time.time()))
        # keep last 60 positions
        if len(history) > 60:
            history.pop(0)
        if len(history) < 10:
            return ""
        # compute velocity
        dx = history[-1][0] - history[-5][0]
        dy = history[-1][1] - history[-5][1]
        speed = (dx**2 + dy**2) ** 0.5
        # dwell: has person barely moved in last 20 frames?
        dwell_dx = abs(history[-1][0] - history[-20][0]) if len(history) >= 20 else 1
        dwell_dy = abs(history[-1][1] - history[-20][1]) if len(history) >= 20 else 1
        if dwell_dx < 0.03 and dwell_dy < 0.03:
            return "loitering"
        if speed > 0.12:
            return "running"
        return "walking"

    def _detect_plates(self, frame: np.ndarray) -> List[PlateDetection]:
        if self._plate_reader is None:
            return []
        trusted_plates = [p.upper() for p in self.settings.knowledge.trusted_vehicle_plates]
        results = []
        try:
            detections = self._plate_reader.readtext(frame)
            for (bbox, text, conf) in detections:
                clean = "".join(c for c in text.upper() if c.isalnum())
                if len(clean) >= 4:
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[2]
                    h, w = frame.shape[:2]
                    normalized = (x1/w, y1/h, x2/w, y2/h)
                    results.append(PlateDetection(
                        plate_text=clean,
                        confidence=conf,
                        bbox=normalized,
                        trusted=clean in trusted_plates,
                    ))
        except Exception as e:
            log.debug(f"Plate detection error: {e}")
        return results

    def _detect_faces(self, frame: np.ndarray) -> List[FaceMatch]:
        try:
            from deepface import DeepFace
            known_dir = self.settings.vision.known_faces_dir
            results = []

            # Find faces without recognition first (fast)
            faces_data = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False)
            for face_data in faces_data:
                if face_data["confidence"] < 0.7:
                    continue
                region = face_data["facial_area"]
                bbox = (
                    region["x"] / frame.shape[1],
                    region["y"] / frame.shape[0],
                    (region["x"] + region["w"]) / frame.shape[1],
                    (region["y"] + region["h"]) / frame.shape[0],
                )

                # Try recognition against known faces
                identity = "unknown"
                rec_conf = 0.0
                emotion_label = None
                if Path(known_dir).exists() and any(Path(known_dir).iterdir()):
                    try:
                        recs = DeepFace.find(
                            img_path=frame,
                            db_path=known_dir,
                            model_name=self.settings.vision.face_model,
                            enforce_detection=False,
                            silent=True,
                        )
                        if recs and not recs[0].empty:
                            top = recs[0].iloc[0]
                            identity = Path(top["identity"]).parent.name
                            rec_conf = 1.0 - top.get("distance", 0.5)
                    except Exception:
                        pass

                # Emotion analysis (optional, adds latency)
                try:
                    ana = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)
                    if ana:
                        emotion_label = ana[0].get("dominant_emotion", None)
                except Exception:
                    pass

                results.append(FaceMatch(
                    bbox=bbox, identity=identity,
                    confidence=rec_conf, emotion=emotion_label
                ))
            return results
        except ImportError:
            log.debug("DeepFace not installed, face detection disabled.")
            return []
        except Exception as e:
            log.debug(f"Face detection error: {e}")
            return []

    def _compute_anomaly(self, dets: List[Detection], faces: List[FaceMatch],
                          plates: List[PlateDetection], poses: List[PoseKeypoint]) -> tuple:
        score = 0.0
        reasons = []
        cfg = self.settings.knowledge

        # Time-of-day check
        import datetime
        now = datetime.datetime.now().time()
        start_h, start_m = map(int, cfg.normal_hours.split(" - ")[0].split(":"))
        end_h, end_m = map(int, cfg.normal_hours.split(" - ")[1].split(":"))
        normal_start = datetime.time(start_h, start_m)
        normal_end = datetime.time(end_h, end_m)
        outside_hours = not (normal_start <= now <= normal_end)
        if outside_hours and any(d.label == "person" for d in dets):
            score += 0.4
            reasons.append(f"Person detected outside normal hours ({cfg.normal_hours})")

        # Unknown faces
        for f in faces:
            if f.identity == "unknown":
                score += 0.3
                reasons.append("Unrecognized face detected")
                break

        # Untrusted vehicles
        for p in plates:
            if not p.trusted:
                score += 0.2
                reasons.append(f"Untrusted vehicle plate: {p.plate_text}")

        # Behavior hints from trajectories
        for pose in poses:
            if pose.action_hint == "loitering":
                score += 0.4
                reasons.append(f"Person #{pose.track_id} appears to be loitering")
            elif pose.action_hint == "running":
                score += 0.15
                reasons.append(f"Person #{pose.track_id} is running")

        return min(score, 1.0), reasons

    async def run(self):
        """Main vision loop — runs continuously."""
        log.info("Vision pipeline starting.")
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cfg = cam.create_video_configuration(
                main={"size": self.settings.camera.resolution, "format": "RGB888"}
            )
            cam.configure(cfg)
            cam.start()
            use_picam = True
        except Exception:
            log.warning("Picamera2 not available, falling back to USB webcam (index 0).")
            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            use_picam = False

        skip = self.settings.vision.frame_skip

        try:
            while True:
                if use_picam:
                    frame = cam.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = cam.read()
                    if not ret:
                        await asyncio.sleep(0.1)
                        continue

                self._frame_idx += 1
                if self._frame_idx % skip != 0:
                    await asyncio.sleep(0.01)
                    continue

                # --- YOLO object detection + tracking ---
                yolo_results = self._yolo.track(frame, persist=True, verbose=False,
                                                 conf=self.settings.vision.yolo_confidence)
                detections = []
                for r in yolo_results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        label = r.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxyn[0].tolist()
                        tid = int(box.id[0]) if box.id is not None else None
                        detections.append(Detection(label=label, confidence=conf,
                                                     bbox=(x1, y1, x2, y2), track_id=tid))

                # --- Pose estimation ---
                pose_results = self._pose.track(frame, persist=True, verbose=False)
                poses = []
                for r in pose_results:
                    if r.keypoints is None:
                        continue
                    for i, kp in enumerate(r.keypoints.xyn):
                        tid = int(r.boxes.id[i]) if r.boxes.id is not None else i
                        kp_list = [(float(k[0]), float(k[1]), float(k[2])) for k in kp]
                        # estimate center of mass for trajectory
                        cx = float(r.boxes.xywhn[i][0]) if r.boxes is not None else 0.5
                        cy = float(r.boxes.xywhn[i][1]) if r.boxes is not None else 0.5
                        action = self._analyze_trajectory(tid, cx, cy)
                        poses.append(PoseKeypoint(track_id=tid, keypoints=kp_list, action_hint=action))

                # --- Face detection (run less frequently for performance) ---
                faces = []
                if self._frame_idx % (skip * 5) == 0:
                    faces = await asyncio.get_event_loop().run_in_executor(
                        None, self._detect_faces, frame.copy()
                    )

                # --- Plate detection (run when car is in scene) ---
                plates = []
                has_car = any(d.label in ("car", "truck", "motorcycle") for d in detections)
                if has_car and self._frame_idx % (skip * 3) == 0:
                    plates = await asyncio.get_event_loop().run_in_executor(
                        None, self._detect_plates, frame.copy()
                    )

                # --- Anomaly scoring ---
                anomaly_score, anomaly_reasons = self._compute_anomaly(
                    detections, faces, plates, poses
                )

                # --- Save snapshot if interesting ---
                snap_path = None
                if anomaly_score > 0.3:
                    snap_path = await asyncio.get_event_loop().run_in_executor(
                        None, self._save_snapshot, frame.copy()
                    )

                new_state = SceneState(
                    timestamp=time.time(),
                    frame_idx=self._frame_idx,
                    detections=detections,
                    faces=faces,
                    plates=plates,
                    poses=poses,
                    raw_frame=frame,
                    snapshot_path=snap_path,
                    anomaly_score=anomaly_score,
                    anomaly_reasons=anomaly_reasons,
                )

                with self._lock:
                    self.scene = new_state

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        finally:
            if use_picam:
                cam.stop()
            else:
                cam.release()
            log.info("Vision pipeline stopped.")
