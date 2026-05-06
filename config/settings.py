"""
config/settings.py
All configuration loaded from config.yaml with sensible defaults.
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ServoConfig:
    pan_pin: int = 18        # GPIO BCM pin for pan servo
    tilt_pin: int = 12       # GPIO BCM pin for tilt servo
    pan_min: float = -90.0   # degrees
    pan_max: float = 90.0
    tilt_min: float = -30.0
    tilt_max: float = 60.0
    sweep_step: float = 15.0 # degrees per patrol step


@dataclass
class CameraConfig:
    resolution: tuple = (1920, 1080)
    fps: int = 15
    rotation: int = 0
    hdr: bool = True


@dataclass
class VisionConfig:
    yolo_model: str = "yolov8n.pt"           # nano model for speed; use yolov8s for more accuracy
    yolo_confidence: float = 0.45
    face_model: str = "Facenet512"           # DeepFace backend
    plate_backend: str = "easyocr"           # "easyocr" or "openalpr"
    pose_model: str = "yolov8n-pose.pt"
    frame_skip: int = 3                       # run detection every N frames
    known_faces_dir: str = "data/known_faces"


@dataclass
class LLMConfig:
    provider: str = "ollama"                  # "ollama" | "openai_compatible"
    model: str = "llama3.2:3b"               # fast model for edge; llama3.1:8b for better reasoning
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_tokens: int = 512
    temperature: float = 0.2
    reasoning_interval_sec: float = 8.0      # how often the agent thinks


@dataclass
class AlertConfig:
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    recipient_email: str = ""
    cooldown_sec: int = 120                  # minimum seconds between same-type alerts


@dataclass
class DriveConfig:
    credentials_file: str = "config/drive_credentials.json"
    folder_name: str = "HouseGuardian"
    max_clip_size_mb: int = 50


@dataclass
class KnowledgeBase:
    """Injected into every LLM reasoning cycle as system context."""
    owner_name: str = "Owner"
    home_address_hint: str = "front yard / driveway / backyard"
    trusted_vehicle_plates: List[str] = field(default_factory=list)
    normal_hours: str = "07:00 - 22:00"     # activity outside this window is suspicious
    suspicious_behaviors: List[str] = field(default_factory=lambda: [
        "loitering near doorway > 30 seconds",
        "person repeatedly approaching and retreating",
        "crouching near windows or doors",
        "running away from house",
        "unfamiliar vehicle idling > 2 minutes",
        "person checking door handles",
        "group of 3+ people not moving for > 60 seconds",
    ])
    camera_zones: dict = field(default_factory=lambda: {
        "front_door": {"pan": -30, "tilt": 0, "description": "Front door and porch"},
        "driveway": {"pan": 0, "tilt": -5, "description": "Driveway and street view"},
        "side_gate": {"pan": 45, "tilt": 5, "description": "Side gate and fence"},
        "backyard": {"pan": 80, "tilt": 10, "description": "Backyard left"},
    })


@dataclass
class Settings:
    servo: ServoConfig = field(default_factory=ServoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)
    record_always: bool = False             # set True to always record video
    clip_duration_sec: int = 30

    @classmethod
    def load(cls, path: str = "config/config.yaml") -> "Settings":
        p = Path(path)
        if not p.exists():
            import logging
            logging.getLogger("settings").warning(f"No config at {path}, using defaults.")
            return cls()
        raw = yaml.safe_load(p.read_text())
        s = cls()
        # simple shallow merge per section
        if "servo" in raw:
            s.servo = ServoConfig(**{**vars(s.servo), **raw["servo"]})
        if "camera" in raw:
            s.camera = CameraConfig(**{**vars(s.camera), **raw["camera"]})
        if "vision" in raw:
            s.vision = VisionConfig(**{**vars(s.vision), **raw["vision"]})
        if "llm" in raw:
            s.llm = LLMConfig(**{**vars(s.llm), **raw["llm"]})
        if "alert" in raw:
            s.alert = AlertConfig(**{**vars(s.alert), **raw["alert"]})
        if "drive" in raw:
            s.drive = DriveConfig(**{**vars(s.drive), **raw["drive"]})
        if "knowledge" in raw:
            s.knowledge = KnowledgeBase(**{**vars(s.knowledge), **raw["knowledge"]})
        if "record_always" in raw:
            s.record_always = raw["record_always"]
        if "clip_duration_sec" in raw:
            s.clip_duration_sec = raw["clip_duration_sec"]
        return s
