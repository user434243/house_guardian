"""
tools/recognize_vehicle_model.py
Classifies vehicle make/model and dominant color from a cropped vehicle image.

Why this matters:
  Plates can be changed in 30 seconds. A car's color, shape, and model
  is much harder to change. "The same silver Toyota that appeared Tuesday"
  is a pattern even if the plate changes.

Color detection: dominant color via K-means on the vehicle crop.
Make/model: Uses a fine-tuned YOLO or falls back to heuristic description
            (we use the YOLO class label + color as a fingerprint).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("vehicle_model")

# Color name mapping from HSV ranges
COLOR_MAP = [
    # (name,  H_low, H_high, S_low, V_low)
    ("red",     0,   10,  80, 80),
    ("red",   160,  180,  80, 80),
    ("orange",  10,   25,  80, 80),
    ("yellow",  25,   35,  80, 80),
    ("green",   35,   85,  50, 50),
    ("cyan",    85,  100,  50, 50),
    ("blue",   100,  130,  50, 50),
    ("purple", 130,  160,  50, 50),
    ("white",    0,  180,   0,200),
    ("silver",   0,  180,   0,150),
    ("black",    0,  180,   0, 50),
    ("gray",     0,  180,  20,100),
]


def detect_dominant_color(img: np.ndarray) -> str:
    """
    Detect the dominant color of a vehicle crop using K-means on HSV pixels.
    Returns a color name string.
    """
    if img is None or img.size == 0:
        return "unknown"

    # Resize for speed
    small = cv2.resize(img, (64, 64))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Reshape to pixel list
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # K-means with 3 clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, labels, centers = cv2.kmeans(
            pixels, 3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )
    except Exception:
        return "unknown"

    # Find the most common cluster
    counts = np.bincount(labels.flatten())
    dominant = centers[counts.argmax()]
    h, s, v = dominant

    # Match to color name
    for name, h_lo, h_hi, s_min, v_min in COLOR_MAP:
        if h_lo <= h <= h_hi and s >= s_min and v >= v_min:
            return name

    return "unknown"


def extract_vehicle_crop(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    """Crop vehicle from frame using normalized bbox (x1,y1,x2,y2)."""
    h, w = frame.shape[:2]
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)
    # Add some padding
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


class VehicleRecognizer:
    """
    Processes vehicle detections to extract color + class + a fingerprint
    that can be matched across sessions even if the plate changes.
    """

    def __init__(self, event_logger):
        self.db = event_logger

    async def process(
        self,
        frame: np.ndarray,
        detection,              # Detection object from vision pipeline
        plate_text: str = "",
        trusted: bool = False,
        zone: str = "",
        snapshot_path: str = "",
    ) -> dict:
        """
        Process a vehicle detection. Returns a rich description dict.
        """
        crop = extract_vehicle_crop(frame, detection.bbox)
        color = detect_dominant_color(crop) if crop is not None else "unknown"

        vehicle_class = detection.label   # "car", "truck", "motorcycle", "bus"
        make_model_hint = self._guess_make_model(detection, crop)

        # Build a vehicle fingerprint: class + color (plates can change)
        fingerprint = f"{color} {vehicle_class}"
        if make_model_hint:
            fingerprint = f"{color} {make_model_hint}"

        # Log to DB
        vehicle_id = await self.db.log_vehicle(
            plate_text=plate_text,
            trusted=trusted,
            vehicle_class=vehicle_class,
            color=color,
            make_model=make_model_hint,
            snapshot_path=snapshot_path,
            zone=zone,
        )

        # Check prior appearances of same fingerprint (regardless of plate)
        prior_same_model = self.db.query_vehicle_history(
            make_model=make_model_hint or vehicle_class, hours=168
        ) if make_model_hint else []
        prior_same_plate = self.db.query_vehicle_history(
            plate=plate_text, hours=168
        ) if plate_text else []

        # How many times has this VEHICLE (not just plate) appeared?
        fingerprint_matches = [
            r for r in prior_same_model
            if r["color"] == color and r["vehicle_class"] == vehicle_class
        ]
        total_appearances = len(fingerprint_matches)

        result = {
            "vehicle_id": vehicle_id,
            "vehicle_class": vehicle_class,
            "color": color,
            "make_model": make_model_hint or vehicle_class,
            "fingerprint": fingerprint,
            "plate_text": plate_text,
            "trusted": trusted,
            "prior_plate_appearances": len(prior_same_plate),
            "prior_model_appearances": total_appearances,
        }

        # Build assessment
        if trusted:
            result["assessment"] = f"Trusted vehicle: {fingerprint}, plate {plate_text}."
        elif total_appearances >= 3:
            result["assessment"] = (
                f"⚠ REPEAT VEHICLE — {fingerprint} has appeared {total_appearances} times before. "
                f"Plate: {plate_text or 'unread'}. This vehicle is familiar but NOT trusted."
            )
        elif total_appearances >= 1:
            result["assessment"] = (
                f"Previously seen vehicle — {fingerprint} appeared {total_appearances} time(s) before. "
                f"Plate: {plate_text or 'unread'}. Monitor."
            )
        else:
            result["assessment"] = (
                f"New vehicle: {fingerprint}. Plate: {plate_text or 'unread'}. First appearance."
            )

        log.info(f"Vehicle: {result['assessment']}")
        return result

    def _guess_make_model(self, detection, crop: Optional[np.ndarray]) -> str:
        """
        Basic make/model estimation.
        In production you would swap this with a fine-tuned vehicle classifier
        (e.g. VMMRdb model or Stanford Cars dataset trained model).
        For now we use YOLO's class label + aspect ratio heuristic.
        """
        if crop is None:
            return detection.label

        h, w = crop.shape[:2] if crop is not None else (1, 1)
        aspect = w / h if h > 0 else 1.0

        label = detection.label
        if label == "truck":
            return "pickup_truck" if aspect > 1.8 else "heavy_truck"
        elif label == "car":
            if aspect > 2.2:
                return "sedan"
            elif aspect > 1.6:
                return "SUV"
            else:
                return "hatchback"
        elif label == "motorcycle":
            return "motorcycle"
        elif label == "bus":
            return "bus"
        return label
