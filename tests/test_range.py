"""Unit tests – RangeEstimator"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from range.range_estimator import RangeEstimator


def test_range_basic():
    est = RangeEstimator(enabled=True)
    # Fake bbox ~100 px wide on 1280-wide image
    result = est.estimate([100, 100, 200, 180], class_name="quadcopter")
    assert result["status"] in ("ok", "invalid")
    if result["status"] == "ok":
        assert result["distance_m"] is not None
        assert result["distance_m"] > 0
        assert result["method"] == "monocular_estimation"


def test_range_offline():
    est = RangeEstimator(enabled=False)
    result = est.estimate([0, 0, 50, 50], class_name="drone")
    assert result["status"] == "RANGE ESTIMATOR OFFLINE"
    assert result["distance_m"] is None


def test_range_tiny_box():
    est = RangeEstimator(enabled=True, min_pixel_width=20)
    result = est.estimate([10, 10, 15, 15], class_name="quadcopter")
    assert result["status"] == "invalid"


def test_estimate_tracks():
    est = RangeEstimator(enabled=True)
    tracks = [
        {"bbox": [50, 50, 150, 120], "class_name": "quadcopter"},
        {"bbox": [200, 200, 280, 260], "class_name": "racing_drone"},
    ]
    out = est.estimate_tracks(tracks)
    assert len(out) == 2
    assert "range" in out[0]
    assert "estimated_range_m" in out[0]
