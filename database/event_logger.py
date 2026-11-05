"""
Event Logger + Snapshot Capture – Stage 11

Persists detections, tracks, classifications, RF, range, alerts,
system events and optional JPEG snapshots to SQLite + disk.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from database.database import init_db, get_connection

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Writes pipeline events to SQLite and optionally saves snapshots.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        snapshots_dir: Optional[str | Path] = None,
        save_snapshots: bool = True,
        snapshot_on_alert_only: bool = False,
        snapshot_min_conf: float = 0.40,
        log_every_n_frames: int = 1,
    ):
        self.db_path = init_db(db_path)
        self.snapshots_dir = Path(snapshots_dir) if snapshots_dir else (
            Path(__file__).resolve().parent.parent / "events" / "images"
        )
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.save_snapshots = save_snapshots
        self.snapshot_on_alert_only = snapshot_on_alert_only
        self.snapshot_min_conf = snapshot_min_conf
        self.log_every_n_frames = max(1, log_every_n_frames)

        self._frame_counter = 0
        self._last_alert_keys: set = set()

        logger.info(
            "EventLogger ready | db=%s | snapshots=%s | save_snapshots=%s",
            self.db_path, self.snapshots_dir, self.save_snapshots,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_frame(
        self,
        tracks: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        rf_obs: Optional[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
        frame_idx: int = 0,
    ):
        """
        Log everything for the current processed frame.
        Called from the main pipeline after threat assessment.
        """
        self._frame_counter += 1
        if self._frame_counter % self.log_every_n_frames != 0:
            return

        ts = datetime.now(timezone.utc).isoformat()

        try:
            with get_connection(self.db_path) as conn:
                # Detections
                for det in detections or []:
                    conn.execute(
                        """INSERT INTO detections
                           (timestamp, track_id, bbox, confidence, class_name, class_id, frame_idx)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ts,
                            None,
                            json.dumps(det.get("bbox")),
                            det.get("confidence"),
                            det.get("class_name"),
                            det.get("class_id"),
                            frame_idx,
                        ),
                    )

                # Tracks + classification + range + alerts
                for trk in tracks or []:
                    tid = trk.get("track_id")
                    conn.execute(
                        """INSERT INTO tracks
                           (timestamp, track_id, status, bbox, confidence, class_name,
                            centroid, hits, age)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ts, tid, trk.get("status"),
                            json.dumps(trk.get("bbox")),
                            trk.get("confidence"),
                            trk.get("class_name"),
                            json.dumps(trk.get("centroid")),
                            trk.get("hits"),
                            trk.get("age"),
                        ),
                    )

                    cls = trk.get("classification") or {}
                    if cls:
                        conn.execute(
                            """INSERT INTO classifications
                               (timestamp, track_id, class_name, confidence, status, model)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                ts, tid,
                                cls.get("class_name"),
                                cls.get("confidence"),
                                cls.get("status"),
                                cls.get("model"),
                            ),
                        )

                    rng = trk.get("range") or {}
                    if rng.get("distance_m") is not None:
                        conn.execute(
                            """INSERT INTO range_estimates
                               (timestamp, track_id, distance_m, confidence, method, status)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                ts, tid,
                                rng.get("distance_m"),
                                rng.get("confidence"),
                                rng.get("method"),
                                rng.get("status"),
                            ),
                        )

                    threat = trk.get("threat") or {}
                    if threat.get("alert"):
                        reason = "; ".join(threat.get("reasons") or [])
                        conn.execute(
                            """INSERT INTO alerts
                               (timestamp, track_id, alert_level, reason, status,
                                auth_status, estimated_range_m)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                ts, tid,
                                threat.get("level"),
                                reason,
                                "active",
                                trk.get("auth_status"),
                                trk.get("estimated_range_m"),
                            ),
                        )

                        # Snapshot on alert
                        if self.save_snapshots and frame is not None:
                            self._save_snapshot(conn, frame, trk, threat.get("level"), ts)

                    elif self.save_snapshots and not self.snapshot_on_alert_only and frame is not None:
                        conf = float(trk.get("confidence") or 0)
                        if conf >= self.snapshot_min_conf:
                            # Avoid spamming: only one snapshot per track per session window
                            key = f"{tid}"
                            if key not in self._last_alert_keys:
                                self._save_snapshot(conn, frame, trk, None, ts)
                                self._last_alert_keys.add(key)
                                if len(self._last_alert_keys) > 50:
                                    self._last_alert_keys.clear()

                # RF
                if rf_obs:
                    conn.execute(
                        """INSERT INTO rf_observations
                           (timestamp, detected, frequency_hz, signal_strength_dbm,
                            bandwidth_hz, status, backend)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ts,
                            1 if rf_obs.get("detected") else 0,
                            rf_obs.get("frequency_hz") or (
                                (rf_obs.get("frequency_mhz") or 0) * 1e6
                                if rf_obs.get("frequency_mhz") else None
                            ),
                            rf_obs.get("signal_strength_dbm"),
                            rf_obs.get("bandwidth_hz"),
                            rf_obs.get("status"),
                            rf_obs.get("backend"),
                        ),
                    )
        except Exception as e:
            logger.error(f"EventLogger.log_frame failed: {e}")

    def log_system_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with get_connection(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO system_events (timestamp, event_type, message, data)
                       VALUES (?, ?, ?, ?)""",
                    (ts, event_type, message, json.dumps(data or {})),
                )
        except Exception as e:
            logger.error(f"log_system_event failed: {e}")

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def _save_snapshot(
        self,
        conn,
        frame: np.ndarray,
        track: Dict[str, Any],
        alert_level: Optional[str],
        ts: str,
    ):
        tid = track.get("track_id", 0)
        # Filename: YYYYMMDD_HHMMSS_TRACKID.jpg
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stamp}_{tid}.jpg"
        filepath = self.snapshots_dir / filename

        try:
            # Crop around bbox with padding if possible
            img = frame
            bbox = track.get("bbox")
            if bbox and len(bbox) == 4:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    img = crop

            cv2.imwrite(str(filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            conn.execute(
                """INSERT INTO snapshots
                   (timestamp, track_id, filepath, alert_level, class_name, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    ts, tid, str(filepath),
                    alert_level,
                    track.get("class_name"),
                    track.get("confidence"),
                ),
            )
            logger.debug("Snapshot saved: %s", filepath.name)
        except Exception as e:
            logger.warning(f"Snapshot save failed: {e}")

    # ------------------------------------------------------------------
    # Query helpers (for API later)
    # ------------------------------------------------------------------
    def recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM system_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_snapshots(self, limit: int = 20) -> List[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
