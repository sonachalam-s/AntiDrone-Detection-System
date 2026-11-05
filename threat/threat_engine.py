"""
Threat Engine – Stage 8

Threat levels:
  LOW | MEDIUM | HIGH | CRITICAL

Inputs considered:
  - Authorization status (AUTHORIZED / UNKNOWN / UNCONFIRMED)
  - Restricted-zone membership
  - Classification confidence / availability
  - Track persistence (age / hits)
  - Estimated range
  - RF observation (presence + optional correlation)
  - Configurable behaviour rules

IMPORTANT:
- Generates only alerts (visual / log / dashboard).
- Never issues physical countermeasures, jamming, or takeover commands.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


class ThreatEngine:
    """
    Rule-based threat assessment for each fused track.
    """

    def __init__(
        self,
        # Configurable thresholds
        close_range_m: float = 100.0,
        very_close_range_m: float = 40.0,
        persistent_hits: int = 15,
        high_rf_corr_conf: float = 0.55,
        enabled: bool = True,
    ):
        self.close_range_m = close_range_m
        self.very_close_range_m = very_close_range_m
        self.persistent_hits = persistent_hits
        self.high_rf_corr_conf = high_rf_corr_conf
        self.enabled = enabled
        logger.info(
            "ThreatEngine ready | close=%.0fm | very_close=%.0fm | "
            "persistent_hits=%d",
            close_range_m, very_close_range_m, persistent_hits,
        )

    def assess(
        self,
        track: Dict[str, Any],
        rf_obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute threat level and reasons for one track.

        Returns:
        {
            "level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
            "score": int,                 # 0-100 internal score
            "reasons": [str, ...],
            "alert": bool,                # True if level >= HIGH
            "timestamp": iso str
        }
        """
        if not self.enabled:
            return {
                "level": "LOW",
                "score": 0,
                "reasons": ["Threat engine disabled"],
                "alert": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        score = 0
        reasons: List[str] = []

        # --- Authorization ---
        auth = track.get("authorization", {}) or {}
        auth_status = auth.get("status") or track.get("auth_status", "UNCONFIRMED")

        if auth_status == "AUTHORIZED":
            score += 0
            reasons.append("Authorized drone")
        elif auth_status == "UNKNOWN":
            score += 35
            reasons.append("Unknown (not in authorized registry)")
        else:  # UNCONFIRMED
            score += 20
            reasons.append("Unconfirmed identity")

        # --- Restricted zone ---
        rz = track.get("restricted_zone", {}) or {}
        if rz.get("inside"):
            score += 30
            reasons.append(f"Inside restricted zone: {rz.get('zone_name', '?')}")

        # --- Range ---
        dist = track.get("estimated_range_m")
        if dist is not None:
            if dist <= self.very_close_range_m:
                score += 25
                reasons.append(f"Very close range (~{dist:.0f} m)")
            elif dist <= self.close_range_m:
                score += 12
                reasons.append(f"Close range (~{dist:.0f} m)")

        # --- Track persistence ---
        hits = track.get("hits", 0) or 0
        age = track.get("age", 0) or 0
        if hits >= self.persistent_hits or age >= self.persistent_hits:
            score += 10
            reasons.append(f"Persistent track (hits={hits}, age={age})")

        # --- Classification confidence ---
        cls = track.get("classification", {}) or {}
        if cls.get("status") == "ok":
            conf = float(cls.get("confidence", 0))
            if conf < 0.5 and auth_status != "AUTHORIZED":
                score += 5
                reasons.append("Low classification confidence")
        else:
            if auth_status != "AUTHORIZED":
                score += 5
                reasons.append("Classifier offline / unavailable")

        # --- RF ---
        fused = track.get("fused", {}) or {}
        rf_info = fused.get("rf", {}) or {}
        if rf_info.get("status") == "ok" and rf_info.get("detected"):
            score += 8
            reasons.append("RF activity detected")
            corr_conf = float(rf_info.get("correlation_confidence", 0) or 0)
            if rf_info.get("correlated_to_track") and corr_conf >= self.high_rf_corr_conf:
                score += 10
                reasons.append(f"RF correlated to track (conf={corr_conf:.2f})")

        # Clamp & map to level
        score = max(0, min(100, score))
        level = self._score_to_level(score, auth_status)

        # Special case: pure AUTHORIZED + not in restricted zone → force LOW
        if auth_status == "AUTHORIZED" and not rz.get("inside"):
            level = "LOW"
            score = min(score, 15)
            if "Authorized drone" not in reasons:
                reasons.insert(0, "Authorized drone")

        alert = level in ("HIGH", "CRITICAL")

        return {
            "level": level,
            "score": score,
            "reasons": reasons,
            "alert": alert,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _score_to_level(self, score: int, auth_status: str) -> str:
        if score >= 75:
            return "CRITICAL"
        if score >= 50:
            return "HIGH"
        if score >= 25:
            return "MEDIUM"
        return "LOW"

    def assess_tracks(
        self,
        tracks: List[Dict[str, Any]],
        rf_obs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Attach threat assessment to every track.
        Adds keys: threat, threat_level
        """
        for trk in tracks:
            threat = self.assess(trk, rf_obs)
            trk["threat"] = threat
            trk["threat_level"] = threat["level"]
        return tracks

    def get_active_alerts(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return list of tracks that currently raise an alert (HIGH/CRITICAL)."""
        alerts = []
        for trk in tracks:
            threat = trk.get("threat") or {}
            if threat.get("alert"):
                alerts.append({
                    "track_id": trk.get("track_id"),
                    "level": threat.get("level"),
                    "reasons": threat.get("reasons", []),
                    "auth_status": trk.get("auth_status"),
                    "estimated_range_m": trk.get("estimated_range_m"),
                    "timestamp": threat.get("timestamp"),
                })
        return alerts
