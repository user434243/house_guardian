"""
tools/event_logger.py
Logs all security events to a local SQLite database.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger("eventlog")
DB_PATH = "data/logs/events.db"


class EventLogger:
    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    metadata TEXT,
                    pan_degrees REAL,
                    tilt_degrees REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON events (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events (event_type)")
            conn.commit()
        log.debug(f"Event DB ready: {self._db_path}")

    async def log(
        self,
        event_type: str,
        description: str,
        metadata: Dict[str, Any] = None,
        pan: float = 0.0,
        tilt: float = 0.0,
    ):
        await asyncio.get_event_loop().run_in_executor(
            None, self._log_sync, event_type, description, metadata or {}, pan, tilt
        )

    def _log_sync(self, event_type, description, metadata, pan, tilt):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO events (timestamp, event_type, description, metadata, pan_degrees, tilt_degrees) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), event_type, description, json.dumps(metadata), pan, tilt),
            )
            conn.commit()
        log.debug(f"Event logged: [{event_type}] {description}")

    def query_recent(self, hours: int = 24, event_type: str = None) -> list:
        since = time.time() - hours * 3600
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM events WHERE timestamp > ? AND event_type = ? ORDER BY timestamp DESC",
                    (since, event_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events WHERE timestamp > ? ORDER BY timestamp DESC",
                    (since,),
                ).fetchall()
        return [dict(r) for r in rows]
