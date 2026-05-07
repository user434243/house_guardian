"""
tools/query_history.py
Gives the agent structured memory — lets it query past events
before making a decision. This is what separates pattern detection
from one-shot reactive alerts.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

log = logging.getLogger("query_history")


class HistoryQuery:
    """
    Called by the agent every reasoning cycle when it wants to check
    whether a face, vehicle, or event has been seen before.
    Returns a structured dict the agent reads as natural language context.
    """

    def __init__(self, event_logger):
        self.db = event_logger

    async def query(
        self,
        query_type: str,
        identifier: str = "",
        hours: int = 168,        # default: 7 days back
    ) -> Dict[str, Any]:
        """
        query_type: "face" | "vehicle" | "plate" | "audio" | "group" | "summary"
        identifier: face identity, plate text, make_model string, or sound class
        hours: how far back to look
        """
        if query_type == "face":
            return await self._query_face(identifier, hours)
        elif query_type in ("vehicle", "plate"):
            return await self._query_vehicle(identifier, hours)
        elif query_type == "audio":
            return await self._query_audio(identifier, hours)
        elif query_type == "group":
            return await self._query_groups(hours)
        elif query_type == "summary":
            return await self._query_summary(hours)
        else:
            return {"error": f"Unknown query_type: {query_type}"}

    async def _query_face(self, identity: str, hours: int) -> dict:
        rows = self.db.query_face_history(identity=identity or None, hours=hours)
        if not rows:
            return {
                "found": False,
                "identity": identity,
                "message": f"No face records for '{identity}' in the last {hours}h.",
            }

        visit_count = max(r["visit_count"] for r in rows)
        first_seen = min(r["timestamp"] for r in rows)
        last_seen = max(r["timestamp"] for r in rows)
        avg_threat = sum(r["threat_score"] for r in rows) / len(rows)
        zones = list(set(r["zone"] for r in rows if r["zone"]))

        is_repeat = visit_count >= 2
        hours_since_last = (time.time() - last_seen) / 3600

        result = {
            "found": True,
            "identity": identity or "unknown",
            "visit_count": visit_count,
            "first_seen_hours_ago": round((time.time() - first_seen) / 3600, 1),
            "last_seen_hours_ago": round(hours_since_last, 1),
            "zones_seen": zones,
            "avg_threat_score": round(avg_threat, 2),
            "is_repeat_visitor": is_repeat,
            "total_sightings_in_window": len(rows),
        }

        # Build natural language summary for the agent
        if identity and not identity.startswith("unknown"):
            result["assessment"] = f"Known person '{identity}'. Seen {visit_count} times. Trusted."
        elif is_repeat:
            result["assessment"] = (
                f"⚠ REPEAT UNKNOWN — this unidentified person has appeared {visit_count} times "
                f"(last {hours_since_last:.0f}h ago, zones: {', '.join(zones)}). "
                f"Elevated threat — treat as high priority."
            )
        else:
            result["assessment"] = "First-time unknown visitor. Monitor carefully."

        return result

    async def _query_vehicle(self, identifier: str, hours: int) -> dict:
        # Try as plate first, then make_model
        rows = self.db.query_vehicle_history(plate=identifier, hours=hours)
        if not rows:
            rows = self.db.query_vehicle_history(make_model=identifier, hours=hours)

        if not rows:
            return {
                "found": False,
                "identifier": identifier,
                "message": f"No vehicle records matching '{identifier}' in last {hours}h.",
            }

        visit_count = max(r["visit_count"] for r in rows)
        first_seen = min(r["timestamp"] for r in rows)
        last_seen = max(r["timestamp"] for r in rows)
        trusted = any(r["trusted"] for r in rows)
        plates = list(set(r["plate_text"] for r in rows if r["plate_text"]))
        models = list(set(r["make_model"] for r in rows if r["make_model"]))
        colors = list(set(r["color"] for r in rows if r["color"]))

        hours_since_last = (time.time() - last_seen) / 3600
        is_repeat = visit_count >= 2

        result = {
            "found": True,
            "plates_seen": plates,
            "make_models": models,
            "colors": colors,
            "trusted": trusted,
            "visit_count": visit_count,
            "first_seen_hours_ago": round((time.time() - first_seen) / 3600, 1),
            "last_seen_hours_ago": round(hours_since_last, 1),
            "is_repeat": is_repeat,
        }

        if trusted:
            result["assessment"] = f"Trusted vehicle. Seen {visit_count} times. No action needed."
        elif is_repeat:
            result["assessment"] = (
                f"⚠ REPEAT UNKNOWN VEHICLE — appeared {visit_count} times "
                f"(last {hours_since_last:.0f}h ago). "
                f"Plates: {plates}. Consider alerting owner."
            )
        else:
            result["assessment"] = "Unknown vehicle, first appearance. Monitor."

        return result

    async def _query_audio(self, sound_class: str, hours: int) -> dict:
        rows = self.db.query_audio_history(sound_class=sound_class or None, hours=hours)
        if not rows:
            return {
                "found": False,
                "message": f"No audio events matching '{sound_class}' in last {hours}h.",
            }

        classes = {}
        for r in rows:
            classes[r["sound_class"]] = classes.get(r["sound_class"], 0) + 1

        dangerous = [c for c in classes if c in ("glass_break", "shout", "scream", "gunshot")]
        return {
            "found": True,
            "event_count": len(rows),
            "classes": classes,
            "dangerous_sounds_detected": dangerous,
            "assessment": (
                f"⚠ DANGEROUS SOUNDS detected: {dangerous}" if dangerous
                else f"Audio events: {classes}. No dangerous sounds."
            ),
        }

    async def _query_groups(self, hours: int) -> dict:
        since = time.time() - hours * 3600
        rows = self.db.query_recent(hours=hours, event_type=None)
        # Pull from group_log directly
        import sqlite3
        from tools.event_logger import _connect, DB_PATH
        with _connect(DB_PATH) as conn:
            group_rows = conn.execute(
                "SELECT * FROM group_log WHERE timestamp>? ORDER BY timestamp DESC LIMIT 20",
                (since,)
            ).fetchall()
        group_rows = [dict(r) for r in group_rows]

        if not group_rows:
            return {"found": False, "message": f"No group events in last {hours}h."}

        high_threat = [r for r in group_rows if r["threat_score"] > 0.5]
        return {
            "found": True,
            "total_group_events": len(group_rows),
            "high_threat_groups": len(high_threat),
            "assessment": (
                f"⚠ {len(high_threat)} high-threat group events detected in last {hours}h."
                if high_threat else f"{len(group_rows)} group events, none high-threat."
            ),
        }

    async def _query_summary(self, hours: int) -> dict:
        stats = self.db.summary_stats(hours=hours)
        lines = [
            f"Last {hours}h summary:",
            f"  Events logged: {stats['total_events']}",
            f"  Known faces: {stats['known_faces']}",
            f"  Unknown faces: {stats['unknown_faces']}",
            f"  Vehicles: {stats['total_vehicles']} total, {stats['untrusted_vehicles']} untrusted",
            f"  Packages: {stats['packages_present']} on porch, {stats['packages_stolen']} stolen",
            f"  Suspicious audio events: {stats['suspicious_audio']}",
        ]
        stats["natural_language"] = "\n".join(lines)
        return stats
