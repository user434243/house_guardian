"""
tools/event_logger.py  (v2 — extended schema)

Tables:
  events          — general log (backward compatible)
  face_sightings  — every face appearance with re-id hash and visit counter
  vehicle_log     — every vehicle: plate + make/model/color + visit counter
  package_log     — packages detected on porch with status tracking
  group_log       — multi-person group observations
  audio_events    — sound classification events
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("eventlog")
DB_PATH = "data/logs/events.db"


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class EventLogger:
    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with _connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     REAL    NOT NULL,
                    event_type    TEXT    NOT NULL,
                    description   TEXT,
                    metadata      TEXT,
                    pan_degrees   REAL,
                    tilt_degrees  REAL
                );
                CREATE INDEX IF NOT EXISTS idx_ts   ON events (timestamp);
                CREATE INDEX IF NOT EXISTS idx_type ON events (event_type);

                CREATE TABLE IF NOT EXISTS face_sightings (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     REAL    NOT NULL,
                    identity      TEXT    NOT NULL,
                    face_hash     TEXT,
                    snapshot_path TEXT,
                    zone          TEXT,
                    dwell_seconds REAL    DEFAULT 0,
                    visit_count   INTEGER DEFAULT 1,
                    threat_score  REAL    DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_face_id   ON face_sightings (identity);
                CREATE INDEX IF NOT EXISTS idx_face_hash ON face_sightings (face_hash);
                CREATE INDEX IF NOT EXISTS idx_face_ts   ON face_sightings (timestamp);

                CREATE TABLE IF NOT EXISTS vehicle_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     REAL    NOT NULL,
                    plate_text    TEXT,
                    trusted       INTEGER DEFAULT 0,
                    vehicle_class TEXT,
                    color         TEXT,
                    make_model    TEXT,
                    snapshot_path TEXT,
                    zone          TEXT,
                    dwell_seconds REAL    DEFAULT 0,
                    visit_count   INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_veh_plate ON vehicle_log (plate_text);
                CREATE INDEX IF NOT EXISTS idx_veh_ts    ON vehicle_log (timestamp);

                CREATE TABLE IF NOT EXISTS package_log (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at       REAL    NOT NULL,
                    last_seen_at      REAL,
                    confirmed_gone_at REAL,
                    zone              TEXT    DEFAULT 'front_door',
                    snapshot_path     TEXT,
                    status            TEXT    DEFAULT 'present',
                    alert_sent        INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS group_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     REAL    NOT NULL,
                    person_count  INTEGER NOT NULL,
                    dwell_seconds REAL    DEFAULT 0,
                    zone          TEXT,
                    time_of_day   TEXT,
                    behavior_hint TEXT,
                    threat_score  REAL    DEFAULT 0,
                    snapshot_path TEXT
                );

                CREATE TABLE IF NOT EXISTS audio_events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     REAL    NOT NULL,
                    sound_class   TEXT    NOT NULL,
                    confidence    REAL,
                    db_level      REAL,
                    duration_sec  REAL,
                    clip_path     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_audio_ts ON audio_events (timestamp);
            """)
        log.debug(f"Event DB v2 ready: {self._db_path}")

    # ── General ────────────────────────────────────────────────────────
    async def log(self, event_type: str, description: str,
                  metadata: Dict[str, Any] = None,
                  pan: float = 0.0, tilt: float = 0.0):
        await asyncio.get_event_loop().run_in_executor(
            None, self._log_sync, event_type, description, metadata or {}, pan, tilt
        )

    def _log_sync(self, event_type, description, metadata, pan, tilt):
        with _connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO events (timestamp,event_type,description,metadata,pan_degrees,tilt_degrees) "
                "VALUES (?,?,?,?,?,?)",
                (time.time(), event_type, description, json.dumps(metadata), pan, tilt),
            )

    # ── Faces ──────────────────────────────────────────────────────────
    async def log_face(self, identity: str, face_hash: str = "",
                       snapshot_path: str = "", zone: str = "",
                       dwell_seconds: float = 0, threat_score: float = 0) -> int:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._log_face_sync,
            identity, face_hash, snapshot_path, zone, dwell_seconds, threat_score
        )

    def _log_face_sync(self, identity, face_hash, snapshot_path, zone, dwell, threat) -> int:
        with _connect(self._db_path) as conn:
            prior = 0
            if face_hash:
                row = conn.execute(
                    "SELECT visit_count FROM face_sightings WHERE face_hash=? ORDER BY timestamp DESC LIMIT 1",
                    (face_hash,)
                ).fetchone()
                if row:
                    prior = row["visit_count"]
            cur = conn.execute(
                "INSERT INTO face_sightings "
                "(timestamp,identity,face_hash,snapshot_path,zone,dwell_seconds,visit_count,threat_score) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), identity, face_hash, snapshot_path, zone, dwell, prior + 1, threat)
            )
            return cur.lastrowid

    # ── Vehicles ───────────────────────────────────────────────────────
    async def log_vehicle(self, plate_text: str = "", trusted: bool = False,
                          vehicle_class: str = "", color: str = "",
                          make_model: str = "", snapshot_path: str = "",
                          zone: str = "", dwell_seconds: float = 0) -> int:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._log_vehicle_sync,
            plate_text, trusted, vehicle_class, color, make_model, snapshot_path, zone, dwell_seconds
        )

    def _log_vehicle_sync(self, plate, trusted, vclass, color, make_model, snap, zone, dwell) -> int:
        with _connect(self._db_path) as conn:
            prior = 0
            if plate:
                row = conn.execute(
                    "SELECT COUNT(*) as c FROM vehicle_log WHERE plate_text=?", (plate,)
                ).fetchone()
                prior = row["c"]
            cur = conn.execute(
                "INSERT INTO vehicle_log "
                "(timestamp,plate_text,trusted,vehicle_class,color,make_model,snapshot_path,zone,dwell_seconds,visit_count) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (time.time(), plate or None, int(trusted), vclass, color, make_model, snap, zone, dwell, prior + 1)
            )
            return cur.lastrowid

    # ── Packages ───────────────────────────────────────────────────────
    async def log_package_detected(self, zone: str = "front_door", snapshot_path: str = "") -> int:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._log_package_sync, zone, snapshot_path
        )

    def _log_package_sync(self, zone, snapshot_path) -> int:
        with _connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO package_log (detected_at, last_seen_at, zone, snapshot_path, status) VALUES (?,?,?,?,?)",
                (time.time(), time.time(), zone, snapshot_path, "present")
            )
            return cur.lastrowid

    async def update_package_status(self, package_id: int, status: str, alert_sent: bool = False):
        await asyncio.get_event_loop().run_in_executor(
            None, self._update_package_sync, package_id, status, alert_sent
        )

    def _update_package_sync(self, package_id, status, alert_sent):
        ts = time.time()
        with _connect(self._db_path) as conn:
            if status in ("collected", "stolen"):
                conn.execute(
                    "UPDATE package_log SET status=?, confirmed_gone_at=?, alert_sent=? WHERE id=?",
                    (status, ts, int(alert_sent), package_id)
                )
            else:
                conn.execute(
                    "UPDATE package_log SET status=?, last_seen_at=? WHERE id=?",
                    (status, ts, package_id)
                )

    async def get_active_packages(self) -> List[dict]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_active_packages_sync
        )

    def _get_active_packages_sync(self) -> List[dict]:
        with _connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM package_log WHERE status='present' ORDER BY detected_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Groups ─────────────────────────────────────────────────────────
    async def log_group(self, person_count: int, dwell_seconds: float,
                        zone: str, behavior_hint: str, threat_score: float,
                        snapshot_path: str = "") -> int:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._log_group_sync,
            person_count, dwell_seconds, zone, behavior_hint, threat_score, snapshot_path
        )

    def _log_group_sync(self, count, dwell, zone, behavior, threat, snap) -> int:
        import datetime
        hour = datetime.datetime.now().hour
        tod = "night" if (hour < 6 or hour >= 22) else "evening" if hour >= 19 else "day"
        with _connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO group_log "
                "(timestamp,person_count,dwell_seconds,zone,time_of_day,behavior_hint,threat_score,snapshot_path) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), count, dwell, zone, tod, behavior, threat, snap)
            )
            return cur.lastrowid

    # ── Audio ──────────────────────────────────────────────────────────
    async def log_audio(self, sound_class: str, confidence: float,
                        db_level: float, duration_sec: float, clip_path: str = ""):
        await asyncio.get_event_loop().run_in_executor(
            None, self._log_audio_sync, sound_class, confidence, db_level, duration_sec, clip_path
        )

    def _log_audio_sync(self, sound_class, confidence, db_level, duration_sec, clip_path):
        with _connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO audio_events (timestamp,sound_class,confidence,db_level,duration_sec,clip_path) "
                "VALUES (?,?,?,?,?,?)",
                (time.time(), sound_class, confidence, db_level, duration_sec, clip_path)
            )

    # ── Query helpers ──────────────────────────────────────────────────
    def query_recent(self, hours: int = 24, event_type: str = None) -> list:
        since = time.time() - hours * 3600
        with _connect(self._db_path) as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM events WHERE timestamp>? AND event_type=? ORDER BY timestamp DESC",
                    (since, event_type)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events WHERE timestamp>? ORDER BY timestamp DESC",
                    (since,)
                ).fetchall()
        return [dict(r) for r in rows]

    def query_face_history(self, identity: str = None, hours: int = 168) -> list:
        since = time.time() - hours * 3600
        with _connect(self._db_path) as conn:
            if identity:
                rows = conn.execute(
                    "SELECT * FROM face_sightings WHERE identity=? AND timestamp>? ORDER BY timestamp DESC",
                    (identity, since)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM face_sightings WHERE timestamp>? ORDER BY timestamp DESC LIMIT 100",
                    (since,)
                ).fetchall()
        return [dict(r) for r in rows]

    def query_vehicle_history(self, plate: str = None, make_model: str = None, hours: int = 168) -> list:
        since = time.time() - hours * 3600
        with _connect(self._db_path) as conn:
            if plate:
                rows = conn.execute(
                    "SELECT * FROM vehicle_log WHERE plate_text=? AND timestamp>? ORDER BY timestamp DESC",
                    (plate, since)
                ).fetchall()
            elif make_model:
                rows = conn.execute(
                    "SELECT * FROM vehicle_log WHERE make_model LIKE ? AND timestamp>? ORDER BY timestamp DESC",
                    (f"%{make_model}%", since)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM vehicle_log WHERE timestamp>? ORDER BY timestamp DESC LIMIT 50",
                    (since,)
                ).fetchall()
        return [dict(r) for r in rows]

    def query_audio_history(self, sound_class: str = None, hours: int = 24) -> list:
        since = time.time() - hours * 3600
        with _connect(self._db_path) as conn:
            if sound_class:
                rows = conn.execute(
                    "SELECT * FROM audio_events WHERE sound_class=? AND timestamp>? ORDER BY timestamp DESC",
                    (sound_class, since)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audio_events WHERE timestamp>? ORDER BY timestamp DESC",
                    (since,)
                ).fetchall()
        return [dict(r) for r in rows]

    def summary_stats(self, hours: int = 24) -> dict:
        """Quick stats snapshot — used in daily brief."""
        since = time.time() - hours * 3600
        with _connect(self._db_path) as conn:
            def count(q, *args):
                return conn.execute(q, args).fetchone()[0]
            return {
                "hours": hours,
                "total_events":       count("SELECT COUNT(*) FROM events WHERE timestamp>?", since),
                "unknown_faces":      count("SELECT COUNT(*) FROM face_sightings WHERE identity LIKE 'unknown%' AND timestamp>?", since),
                "known_faces":        count("SELECT COUNT(*) FROM face_sightings WHERE identity NOT LIKE 'unknown%' AND timestamp>?", since),
                "total_vehicles":     count("SELECT COUNT(*) FROM vehicle_log WHERE timestamp>?", since),
                "untrusted_vehicles": count("SELECT COUNT(*) FROM vehicle_log WHERE trusted=0 AND timestamp>?", since),
                "packages_present":   count("SELECT COUNT(*) FROM package_log WHERE status='present'"),
                "packages_stolen":    count("SELECT COUNT(*) FROM package_log WHERE status='stolen' AND detected_at>?", since),
                "suspicious_audio":   count("SELECT COUNT(*) FROM audio_events WHERE sound_class NOT IN ('ambient','silence') AND timestamp>?", since),
            }
