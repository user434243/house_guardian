"""
tools/cross_reference_faces.py
Tracks unknown visitors across sessions using perceptual face hashing.

When an unknown face is seen:
  1. Compute a lightweight face hash (not a full embedding — fast on Pi)
  2. Query the DB for any prior hash match
  3. Return: first visit OR repeat visit with full history
  4. Repeat unknowns get a higher threat score automatically

The face hash uses the average hash of a normalized face crop —
fast enough to run every detection cycle on a Pi 5.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("cross_ref")

# How similar two hashes must be to count as "same person"
HASH_SIMILARITY_THRESHOLD = 0.85   # 85% bit match


def _compute_face_hash(face_img: np.ndarray) -> str:
    """
    Compute a perceptual hash string for a face crop.
    Resize to 16x16 grayscale, threshold against mean, encode as hex.
    Fast (~1ms), good enough for re-identification across sessions.
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    bits = (resized > mean_val).flatten()
    # Pack bits into bytes
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def _hash_similarity(h1: str, h2: str) -> float:
    """Return 0.0–1.0 similarity between two hex hashes (hamming distance normalized)."""
    if len(h1) != len(h2) or not h1 or not h2:
        return 0.0
    b1 = bytes.fromhex(h1)
    b2 = bytes.fromhex(h2)
    total_bits = len(b1) * 8
    different = sum(bin(a ^ b).count("1") for a, b in zip(b1, b2))
    return 1.0 - (different / total_bits)


class FaceCrossReference:
    def __init__(self, event_logger):
        self.db = event_logger
        # In-memory cache of recent hashes to avoid DB hit every frame
        # { hash_str: { identity, visit_count, first_seen, last_seen } }
        self._cache: dict[str, dict] = {}
        self._cache_ttl = 3600   # 1 hour

    async def process(
        self,
        face_img: np.ndarray,
        identity: str,          # "unknown" or recognized name from DeepFace
        zone: str = "",
        dwell_seconds: float = 0,
        snapshot_path: str = "",
    ) -> dict:
        """
        Main entry point. Call this for every detected face.
        Returns a result dict the agent uses to decide threat level.
        """
        face_hash = _compute_face_hash(face_img)

        # Check in-memory cache first
        cached = self._find_in_cache(face_hash)
        if cached:
            visit_count = cached["visit_count"] + 1
            cached["visit_count"] = visit_count
            cached["last_seen"] = time.time()
            is_repeat = visit_count >= 2
        else:
            # Check DB for prior sightings
            prior_rows = self.db.query_face_history(
                identity=identity if identity != "unknown" else None,
                hours=168   # 7 days
            )
            # Find hash match in prior rows
            matched_row = None
            for row in prior_rows:
                if row.get("face_hash"):
                    sim = _hash_similarity(face_hash, row["face_hash"])
                    if sim >= HASH_SIMILARITY_THRESHOLD:
                        matched_row = row
                        break

            is_repeat = matched_row is not None
            visit_count = (matched_row["visit_count"] + 1) if matched_row else 1

            # Add to cache
            self._cache[face_hash] = {
                "identity": identity,
                "visit_count": visit_count,
                "first_seen": matched_row["timestamp"] if matched_row else time.time(),
                "last_seen": time.time(),
                "cached_at": time.time(),
            }

        # Compute threat score
        threat_score = self._compute_threat(identity, visit_count, is_repeat, dwell_seconds)

        # Assign a stable anonymous ID for repeat unknowns
        stable_identity = identity
        if identity == "unknown":
            stable_identity = f"unknown_{face_hash[:8]}"

        # Log to DB
        await self.db.log_face(
            identity=stable_identity,
            face_hash=face_hash,
            snapshot_path=snapshot_path,
            zone=zone,
            dwell_seconds=dwell_seconds,
            threat_score=threat_score,
        )

        result = {
            "identity": stable_identity,
            "is_known": identity != "unknown",
            "is_repeat_unknown": identity == "unknown" and is_repeat,
            "visit_count": visit_count,
            "threat_score": threat_score,
            "face_hash": face_hash,
        }

        if identity == "unknown" and is_repeat:
            result["assessment"] = (
                f"⚠ REPEAT UNKNOWN VISITOR (visit #{visit_count}). "
                f"This unidentified person has appeared before. Threat score: {threat_score:.2f}."
            )
        elif identity == "unknown":
            result["assessment"] = f"Unknown visitor, first time seen. Threat score: {threat_score:.2f}."
        else:
            result["assessment"] = f"Known person: {identity}. No threat."

        log.info(f"Face cross-ref: {result['assessment']}")
        return result

    def _compute_threat(self, identity: str, visit_count: int,
                        is_repeat: bool, dwell_seconds: float) -> float:
        if identity != "unknown":
            return 0.0    # known person — no threat

        score = 0.2       # base score for any unknown
        if is_repeat:
            score += 0.3
        if visit_count >= 3:
            score += 0.2
        if dwell_seconds > 60:
            score += 0.15
        if dwell_seconds > 180:
            score += 0.15
        return min(score, 1.0)

    def _find_in_cache(self, face_hash: str) -> Optional[dict]:
        """Find a similar hash in the in-memory cache."""
        now = time.time()
        # Expire old entries
        expired = [k for k, v in self._cache.items()
                   if now - v["cached_at"] > self._cache_ttl]
        for k in expired:
            del self._cache[k]

        for cached_hash, data in self._cache.items():
            sim = _hash_similarity(face_hash, cached_hash)
            if sim >= HASH_SIMILARITY_THRESHOLD:
                return data
        return None
