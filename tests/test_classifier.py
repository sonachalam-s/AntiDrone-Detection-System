"""Unit tests – EfficientNetClassifier offline behaviour"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from classification.classifier import EfficientNetClassifier


def test_classifier_offline_without_weights(tmp_path):
    """Missing weights must NOT load random weights – status OFFLINE."""
    fake = tmp_path / "missing.pth"
    clf = EfficientNetClassifier(weights_path=fake)
    assert clf.is_online() is False
    assert clf.status == "CLASSIFIER OFFLINE"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = clf.classify(frame, [100, 100, 200, 200])
    assert result["status"] == "CLASSIFIER OFFLINE"
    assert result["class_name"] == "N/A"


def test_classify_tracks_offline():
    clf = EfficientNetClassifier(weights_path=None)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tracks = [{"bbox": [10, 10, 50, 50], "track_id": 1}]
    out = clf.classify_tracks(frame, tracks)
    assert out[0]["classification"]["status"] == "CLASSIFIER OFFLINE"
