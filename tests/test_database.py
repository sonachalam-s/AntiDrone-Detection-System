"""Unit tests – EventLogger / SQLite schema"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from database.event_logger import EventLogger
from database.database import get_connection


def test_event_logger_writes(tmp_path):
    db = tmp_path / "events.db"
    snap = tmp_path / "snaps"
    logger = EventLogger(
        db_path=db,
        snapshots_dir=snap,
        save_snapshots=True,
        snapshot_on_alert_only=False,
        snapshot_min_conf=0.3,
        log_every_n_frames=1,
    )
    logger.log_system_event("test", "hello")

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    tracks = [{
        "track_id": 1,
        "bbox": [10, 10, 80, 60],
        "confidence": 0.9,
        "class_name": "airplane",
        "status": "confirmed",
        "centroid": (45, 35),
        "hits": 5,
        "age": 5,
        "classification": {"class_name": "quadcopter", "confidence": 0.8,
                           "status": "ok", "model": "EfficientNet"},
        "range": {"distance_m": 100.0, "confidence": 0.5, "status": "ok",
                  "method": "monocular_estimation"},
        "threat": {"alert": False, "level": "LOW", "reasons": []},
        "auth_status": "UNKNOWN",
    }]
    detections = [{"bbox": [10, 10, 80, 60], "confidence": 0.9,
                   "class_name": "airplane", "class_id": 4}]
    rf = {"status": "RF SENSOR OFFLINE", "detected": False}

    logger.log_frame(tracks, detections, rf, frame=frame, frame_idx=1)

    with get_connection(db) as conn:
        n_det = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        n_trk = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        n_sys = conn.execute("SELECT COUNT(*) FROM system_events").fetchone()[0]
    assert n_det >= 1
    assert n_trk >= 1
    assert n_sys >= 1
