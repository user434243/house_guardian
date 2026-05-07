"""
tools/estimate_group_intent.py
Analyzes groups of people in the scene and estimates their collective intent.

A group of 3 people loitering at 2am near a parked car means something
very different from 3 people walking past at noon. This tool reasons
about the COMBINATION of: count, dwell time, time of day, zone, movement.

Called by the vision pipeline when 2+ persons are tracked simultaneously.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

log = logging.getLogger("group_intent")


@dataclass
class GroupObservation:
    person_count: int
    dwell_seconds: float        # how long the group has been stationary/near
    zone: str
    behaviors: List[str]        # list of per-person behavior hints from trajectory
    time_of_day: str            # "day" | "evening" | "night"
    near_vehicle: bool = False
    near_door: bool = False
    any_unknown_face: bool = False


@dataclass
class IntentEstimate:
    threat_score: float         # 0.0 – 1.0
    intent_label: str           # "passing" | "social" | "waiting" | "suspicious" | "threat"
    reasoning: str              # natural language explanation for the agent
    should_alert: bool
    should_record: bool


class GroupIntentEstimator:
    def __init__(self, event_logger):
        self.db = event_logger
        # Track dwell start times per set of track_ids
        # key: frozenset of track_ids, value: first_seen timestamp
        self._group_first_seen: dict[frozenset, float] = {}

    async def estimate(self, obs: GroupObservation,
                       track_ids: frozenset,
                       snapshot_path: str = "") -> IntentEstimate:
        """Estimate the intent of a group and log to DB."""

        # Calculate how long THIS group has been together
        key = track_ids
        if key not in self._group_first_seen:
            self._group_first_seen[key] = time.time()
        dwell = time.time() - self._group_first_seen[key]
        obs.dwell_seconds = dwell

        estimate = self._reason(obs)

        # Log to DB
        await self.db.log_group(
            person_count=obs.person_count,
            dwell_seconds=dwell,
            zone=obs.zone,
            behavior_hint=estimate.intent_label,
            threat_score=estimate.threat_score,
            snapshot_path=snapshot_path,
        )

        if estimate.threat_score > 0.3:
            log.info(f"Group intent [{estimate.intent_label}] score={estimate.threat_score:.2f}: {estimate.reasoning}")

        return estimate

    def clear_group(self, track_ids: frozenset):
        """Call when a group disperses."""
        self._group_first_seen.pop(track_ids, None)

    def _reason(self, obs: GroupObservation) -> IntentEstimate:
        score = 0.0
        reasons = []

        # ── Time of day weight ──────────────────────────────────────
        time_weight = {"day": 1.0, "evening": 1.3, "night": 1.8}[obs.time_of_day]

        # ── Dwell time scoring ──────────────────────────────────────
        if obs.dwell_seconds > 180:       # 3+ min
            score += 0.4
            reasons.append(f"Group stationary for {obs.dwell_seconds/60:.0f} min")
        elif obs.dwell_seconds > 60:      # 1+ min
            score += 0.2
            reasons.append(f"Group dwell {obs.dwell_seconds:.0f}s")

        # ── Group size ──────────────────────────────────────────────
        if obs.person_count >= 4:
            score += 0.25
            reasons.append(f"Large group ({obs.person_count} people)")
        elif obs.person_count >= 3:
            score += 0.15
            reasons.append(f"Group of {obs.person_count}")

        # ── Location context ────────────────────────────────────────
        if obs.near_door:
            score += 0.3
            reasons.append("Group near door/entry point")
        if obs.near_vehicle:
            score += 0.15
            reasons.append("Group near parked vehicle")

        # ── Behavior mix ────────────────────────────────────────────
        loiterers = obs.behaviors.count("loitering")
        runners   = obs.behaviors.count("running")

        if loiterers >= 2:
            score += 0.35
            reasons.append(f"{loiterers} people loitering simultaneously")
        elif loiterers == 1:
            score += 0.2
            reasons.append("1 person loitering in group")

        if runners >= 1:
            score += 0.1
            reasons.append("Running detected in group")

        # ── Unknown faces ───────────────────────────────────────────
        if obs.any_unknown_face:
            score += 0.2
            reasons.append("Unknown faces in group")

        # ── Apply time-of-day multiplier ────────────────────────────
        score = min(score * time_weight, 1.0)

        # ── Classify intent ─────────────────────────────────────────
        if score < 0.15:
            intent = "passing"
        elif score < 0.30:
            intent = "social"
        elif score < 0.50:
            intent = "waiting"
        elif score < 0.70:
            intent = "suspicious"
        else:
            intent = "threat"

        reasoning = (
            f"{obs.person_count} people in {obs.zone} at {obs.time_of_day}. "
            + " | ".join(reasons) + f". Intent: {intent}."
        )

        return IntentEstimate(
            threat_score=round(score, 2),
            intent_label=intent,
            reasoning=reasoning,
            should_alert=score >= 0.5,
            should_record=score >= 0.3,
        )
