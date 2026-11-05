"""Unit tests – SensorFusion"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.sensor_fusion import SensorFusion


def test_fuse_track_basic():
    fusion = SensorFusion(rf_correlation_enabled=True)
    track = {
        "track_id": 7,
        "bbox": [10, 10, 100, 80],
        "confidence": 0.9,
        "class_name": "airplane",
        "status": "confirmed",
        "centroid": (55, 45),
        "velocity": (1.0, 0.5),
        "trajectory": [(50, 40), (55, 45)],
        "classification": {
            "class_name": "quadcopter",
            "confidence": 0.85,
            "status": "ok",
            "model": "EfficientNet-B0",
        },
        "authorization": {"status": "UNKNOWN", "reason": "no match"},
        "auth_status": "UNKNOWN",
        "range": {"distance_m": 120.0, "confidence": 0.6, "status": "ok",
                  "method": "monocular_estimation"},
        "estimated_range_m": 120.0,
    }
    rf = {"status": "RF SENSOR OFFLINE", "detected": False}
    fused = fusion.fuse_track(track, rf)
    assert fused["track_id"] == 7
    assert fused["authorization"]["status"] == "UNKNOWN"
    assert fused["range"]["estimated_m"] == 120.0
    assert "summary" in fused
    assert fused["rf"]["status"] == "RF SENSOR OFFLINE"


def test_fuse_tracks_batch():
    fusion = SensorFusion()
    tracks = [
        {"track_id": 1, "bbox": [0, 0, 50, 50], "confidence": 0.8,
         "class_name": "drone", "status": "confirmed",
         "classification": {"status": "CLASSIFIER OFFLINE"},
         "authorization": {"status": "UNCONFIRMED"},
         "range": {"status": "ok", "distance_m": 50}},
    ]
    out = fusion.fuse_tracks(tracks, None)
    assert len(out) == 1
    assert "fused" in tracks[0]
    assert "fusion_summary" in tracks[0]
