"""
Sensor Fusion – Stage 7

Combines:
  - YOLO visual detection
  - DeepSORT tracking
  - EfficientNet classification
  - Authorization registry result
  - Monocular range estimate
  - Optional RF observation

Produces a single fused state per track.

IMPORTANT:
- RF is never automatically attributed to a specific visual track
  unless a (configurable) correlation heuristic succeeds.
- When correlation is attempted, a correlation confidence is reported.
- Missing sensors are clearly marked; values are never invented.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SensorFusion:
    """
    Fuse multi-modal observations into a unified track state.
    """

    def __init__(
        self,
        rf_correlation_enabled: bool = True,
        rf_min_correlation_conf: float = 0.35,
    ):
        self.rf_correlation_enabled = rf_correlation_enabled
        self.rf_min_correlation_conf = rf_min_correlation_conf
        logger.info(
            "SensorFusion ready | RF correlation=%s",
            "ON" if rf_correlation_enabled else "OFF",
        )

    # ------------------------------------------------------------------
    # RF ↔ visual correlation (conservative heuristic)
    # ------------------------------------------------------------------
    def _correlate_rf(
        self,
        track: Dict[str, Any],
        rf_obs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Attempt a weak association between an RF observation and a visual track.

        Returns:
        {
            "correlated": bool,
            "confidence": float,          # 0-1
            "reason": str,
            "rf_status": str,
            "frequency_mhz": float | None,
            "signal_strength_dbm": float | None,
        }
        """
        result = {
            "correlated": False,
            "confidence": 0.0,
            "reason": "No RF data",
            "rf_status": "RF SENSOR OFFLINE",
            "frequency_mhz": None,
            "signal_strength_dbm": None,
        }

        if rf_obs is None:
            return result

        result["rf_status"] = rf_obs.get("status", "RF SENSOR OFFLINE")
        result["frequency_mhz"] = rf_obs.get("frequency_mhz")
        result["signal_strength_dbm"] = rf_obs.get("signal_strength_dbm")

        if result["rf_status"] != "ok":
            result["reason"] = "RF sensor offline"
            return result

        if not rf_obs.get("detected", False):
            result["reason"] = "No RF activity detected"
            return result

        if not self.rf_correlation_enabled:
            result["reason"] = "RF correlation disabled – observation kept global"
            return result

        # Conservative heuristic:
        # - RF is present and detected
        # - Track has a reasonable estimated range (not None / not huge)
        # - Classification is at least available or track is confirmed
        conf = 0.25  # base when RF is simply "present while we see something"

        range_m = track.get("estimated_range_m")
        if range_m is not None and 5 < range_m < 1500:
            conf += 0.20

        auth = track.get("auth_status") or track.get("authorization", {}).get("status")
        if auth in ("UNKNOWN", "UNCONFIRMED"):
            conf += 0.10  # slightly more interesting for unknown objects

        cls = track.get("classification", {})
        if cls.get("status") == "ok" and cls.get("confidence", 0) > 0.5:
            conf += 0.15

        if track.get("status") == "confirmed":
            conf += 0.10

        conf = min(1.0, conf)
        result["confidence"] = round(conf, 2)

        if conf >= self.rf_min_correlation_conf:
            result["correlated"] = True
            result["reason"] = (
                f"Weak temporal co-occurrence (conf={conf:.2f}). "
                "Not a definitive RF-to-track lock."
            )
        else:
            result["reason"] = (
                f"RF present but correlation confidence too low ({conf:.2f})"
            )

        return result

    # ------------------------------------------------------------------
    # Fuse one track
    # ------------------------------------------------------------------
    def fuse_track(
        self,
        track: Dict[str, Any],
        rf_obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a fused state dictionary for a single track.
        """
        cls = track.get("classification", {})
        auth = track.get("authorization", {})
        rng = track.get("range", {})

        rf_corr = self._correlate_rf(track, rf_obs)

        # Overall identity summary
        auth_status = auth.get("status", track.get("auth_status", "UNCONFIRMED"))
        cls_name = cls.get("class_name", track.get("class_name", "N/A"))
        cls_conf = cls.get("confidence", 0.0)
        cls_status = cls.get("status", "CLASSIFIER OFFLINE")

        dist_m = rng.get("distance_m")
        range_conf = rng.get("confidence", 0.0)

        fused = {
            "track_id": track.get("track_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),

            # Visual
            "bbox": track.get("bbox"),
            "visual_class": track.get("class_name"),
            "visual_confidence": track.get("confidence", 0.0),
            "track_status": track.get("status", "unknown"),
            "centroid": track.get("centroid"),
            "velocity": track.get("velocity"),
            "trajectory": track.get("trajectory"),

            # Classification
            "classification": {
                "class_name": cls_name,
                "confidence": cls_conf,
                "status": cls_status,
                "model": cls.get("model", "N/A"),
            },

            # Authorization
            "authorization": {
                "status": auth_status,
                "drone_id": auth.get("drone_id"),
                "reason": auth.get("reason", ""),
                "matched_model": auth.get("matched_model"),
            },

            # Range
            "range": {
                "estimated_m": dist_m,
                "confidence": range_conf,
                "status": rng.get("status", "N/A"),
                "method": rng.get("method", "monocular_estimation"),
            },

            # RF (global observation + optional weak correlation)
            "rf": {
                "status": rf_corr["rf_status"],
                "detected": rf_obs.get("detected", False) if rf_obs else False,
                "frequency_mhz": rf_corr["frequency_mhz"],
                "signal_strength_dbm": rf_corr["signal_strength_dbm"],
                "correlated_to_track": rf_corr["correlated"],
                "correlation_confidence": rf_corr["confidence"],
                "correlation_reason": rf_corr["reason"],
            },

            # High-level summary label
            "summary": self._make_summary(
                auth_status, cls_name, cls_status, dist_m, rf_corr
            ),
        }

        return fused

    def _make_summary(
        self,
        auth_status: str,
        cls_name: str,
        cls_status: str,
        dist_m: Optional[float],
        rf_corr: Dict[str, Any],
    ) -> str:
        parts = []

        if auth_status == "AUTHORIZED":
            parts.append("AUTHORIZED")
        elif auth_status == "UNKNOWN":
            parts.append("UNKNOWN DRONE")
        else:
            parts.append("UNCONFIRMED")

        if cls_status == "ok" and cls_name not in ("N/A", "unknown_drone"):
            parts.append(cls_name)

        if dist_m is not None:
            parts.append(f"~{dist_m:.0f}m")

        if rf_corr.get("correlated"):
            parts.append(f"RF~{rf_corr['correlation_confidence']:.2f}")
        elif rf_corr.get("rf_status") == "ok" and rf_corr.get("frequency_mhz"):
            parts.append("RF present")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def fuse_tracks(
        self,
        tracks: List[Dict[str, Any]],
        rf_obs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fuse every track and attach the result under track["fused"].
        Also returns the list of fused dictionaries.
        """
        fused_list = []
        for trk in tracks:
            fused = self.fuse_track(trk, rf_obs)
            trk["fused"] = fused
            # Convenience top-level summary
            trk["fusion_summary"] = fused["summary"]
            fused_list.append(fused)
        return fused_list

    def get_global_rf_state(self, rf_obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a clean global RF state for the dashboard (not per-track)."""
        if rf_obs is None:
            return {
                "status": "RF SENSOR OFFLINE",
                "detected": False,
                "frequency_mhz": None,
                "signal_strength_dbm": None,
            }
        return {
            "status": rf_obs.get("status", "RF SENSOR OFFLINE"),
            "detected": rf_obs.get("detected", False),
            "frequency_mhz": rf_obs.get("frequency_mhz"),
            "signal_strength_dbm": rf_obs.get("signal_strength_dbm"),
            "backend": rf_obs.get("backend"),
            "timestamp": rf_obs.get("timestamp"),
        }
