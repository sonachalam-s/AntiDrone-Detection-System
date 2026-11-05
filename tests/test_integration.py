"""
Integration test – pipeline pieces without live camera

camera-like frame → YOLO (optional) → tracker fallback → classifier offline
→ authorization → range → fusion → threat → event log
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


def test_pipeline_offline_modules(tmp_path):
    from classification.classifier import EfficientNetClassifier
    from authorization.registry import AuthorizationRegistry
    from range.range_estimator import RangeEstimator
    from fusion.sensor_fusion import SensorFusion
    from geo.restricted_zone import RestrictedZoneManager
    from threat.threat_engine import ThreatEngine
    from database.event_logger import EventLogger

    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

    # Synthetic detections (simulate YOLO)
    detections = [{
        "bbox": [100, 80, 220, 180],
        "confidence": 0.88,
        "class_name": "airplane",
        "class_id": 4,
    }]

    # Tracker fallback
    tracks = []
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["bbox"]
        tracks.append({
            "track_id": i,
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "class_name": det["class_name"],
            "class_id": det["class_id"],
            "status": "confirmed",
            "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
            "velocity": (0.0, 0.0),
            "trajectory": [((x1 + x2) / 2, (y1 + y2) / 2)],
            "hits": 10,
            "age": 10,
        })

    clf = EfficientNetClassifier(weights_path=None)
    tracks = clf.classify_tracks(frame, tracks)
    assert tracks[0]["classification"]["status"] == "CLASSIFIER OFFLINE"

    db = tmp_path / "int.db"
    reg = AuthorizationRegistry(db_path=db)
    reg.seed_demo_entries()
    tracks = reg.authorize_tracks(tracks)
    assert tracks[0]["auth_status"] in ("UNKNOWN", "UNCONFIRMED", "AUTHORIZED")

    est = RangeEstimator(enabled=True)
    tracks = est.estimate_tracks(tracks)

    zone = RestrictedZoneManager(zones=[])
    tracks = zone.annotate_tracks(tracks)

    fusion = SensorFusion()
    fusion.fuse_tracks(tracks, {"status": "RF SENSOR OFFLINE", "detected": False})
    assert "fusion_summary" in tracks[0]

    threat = ThreatEngine()
    tracks = threat.assess_tracks(tracks)
    assert "threat_level" in tracks[0]

    elog = EventLogger(db_path=db, snapshots_dir=tmp_path / "snaps",
                       save_snapshots=False, log_every_n_frames=1)
    elog.log_frame(tracks, detections, {"status": "RF SENSOR OFFLINE"}, frame=frame)
    elog.log_system_event("integration", "ok")
