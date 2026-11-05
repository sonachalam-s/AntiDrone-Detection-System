"""
Thread-safe shared runtime state between the CV pipeline and the FastAPI server.
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RuntimeState:
    """Singleton-style shared state."""

    def __init__(self):
        self._lock = threading.RLock()
        self._started_at = time.time()

        self.system: Dict[str, Any] = {
            "status": "starting",
            "uptime_s": 0,
            "camera_fps": 0.0,
            "inference_fps": 0.0,
            "device": "cpu",
            "device_name": "CPU",
            "cuda_available": False,
            "classifier_status": "DISABLED",
            "rf_status": "RF SENSOR OFFLINE",
            "tracks_count": 0,
            "alerts_count": 0,
        }

        self.tracks: List[Dict[str, Any]] = []
        self.detections: List[Dict[str, Any]] = []
        self.rf: Dict[str, Any] = {"status": "RF SENSOR OFFLINE"}
        self.alerts: List[Dict[str, Any]] = []
        self.config_snapshot: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []  # recent system events
        self.frame_jpeg: Optional[bytes] = None  # latest annotated JPEG for MJPEG / snapshot
        self._max_events = 200
        self._max_alerts = 100

    # ------------------------------------------------------------------
    def update_system(self, **kwargs):
        with self._lock:
            self.system.update(kwargs)
            self.system["uptime_s"] = round(time.time() - self._started_at, 1)

    def set_tracks(self, tracks: List[Dict[str, Any]]):
        with self._lock:
            # Store a lightweight serialisable copy
            clean = []
            for t in tracks:
                clean.append(self._serialise_track(t))
            self.tracks = clean
            self.system["tracks_count"] = len(clean)

    def set_detections(self, detections: List[Dict[str, Any]]):
        with self._lock:
            self.detections = deepcopy(detections)

    def set_rf(self, rf_obs: Dict[str, Any]):
        with self._lock:
            self.rf = deepcopy(rf_obs) if rf_obs else {"status": "RF SENSOR OFFLINE"}
            self.system["rf_status"] = self.rf.get("status", "RF SENSOR OFFLINE")

    def set_alerts(self, alerts: List[Dict[str, Any]]):
        with self._lock:
            self.alerts = (alerts or [])[-self._max_alerts:]
            self.system["alerts_count"] = len(self.alerts)

    def add_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        with self._lock:
            self.events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "message": message,
                "data": data or {},
            })
            if len(self.events) > self._max_events:
                self.events = self.events[-self._max_events:]

    def set_frame_jpeg(self, jpeg_bytes: Optional[bytes]):
        with self._lock:
            self.frame_jpeg = jpeg_bytes

    def set_config_snapshot(self, cfg: Dict[str, Any]):
        with self._lock:
            self.config_snapshot = deepcopy(cfg)

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Full state snapshot for WebSocket / API."""
        with self._lock:
            return {
                "system": deepcopy(self.system),
                "tracks": deepcopy(self.tracks),
                "detections": deepcopy(self.detections),
                "rf": deepcopy(self.rf),
                "alerts": deepcopy(self.alerts),
                "events": deepcopy(self.events[-50:]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_system(self) -> Dict[str, Any]:
        with self._lock:
            s = deepcopy(self.system)
            s["uptime_s"] = round(time.time() - self._started_at, 1)
            return s

    def get_tracks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return deepcopy(self.tracks)

    def get_detections(self) -> List[Dict[str, Any]]:
        with self._lock:
            return deepcopy(self.detections)

    def get_rf(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self.rf)

    def get_alerts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return deepcopy(self.alerts)

    def get_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return deepcopy(self.events)

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self.config_snapshot)

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.frame_jpeg

    # ------------------------------------------------------------------
    @staticmethod
    def _serialise_track(t: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only JSON-friendly fields."""
        return {
            "track_id": t.get("track_id"),
            "bbox": t.get("bbox"),
            "confidence": t.get("confidence"),
            "class_name": t.get("class_name"),
            "status": t.get("status"),
            "centroid": t.get("centroid"),
            "velocity": t.get("velocity"),
            "hits": t.get("hits"),
            "age": t.get("age"),
            "classification": t.get("classification"),
            "authorization": t.get("authorization"),
            "auth_status": t.get("auth_status"),
            "range": t.get("range"),
            "estimated_range_m": t.get("estimated_range_m"),
            "restricted_zone": t.get("restricted_zone"),
            "in_restricted_zone": t.get("in_restricted_zone"),
            "threat": t.get("threat"),
            "threat_level": t.get("threat_level"),
            "fusion_summary": t.get("fusion_summary"),
            "fused": t.get("fused"),
        }


# Global instance
STATE = RuntimeState()
