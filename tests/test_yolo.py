"""Unit / smoke tests – YOLO detector (skips if model cannot load)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


def test_yolo_detect_smoke():
    try:
        from detection.yolo_detector import YOLODroneDetector
    except Exception as e:
        pytest.skip(f"Cannot import YOLO: {e}")

    try:
        det = YOLODroneDetector(
            model_name="yolo11n.pt",
            confidence=0.25,
            image_size=320,
            device="cpu",  # force CPU for CI-friendly test
            max_det=5,
            use_fp16=False,
        )
    except Exception as e:
        pytest.skip(f"Cannot load YOLO model: {e}")

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # draw a white rectangle so something *might* be detected; mostly we check no crash
    frame[80:160, 100:220] = 255
    results = det.detect(frame)
    assert isinstance(results, list)
    for r in results:
        assert "bbox" in r
        assert "confidence" in r
        assert "class_name" in r
    info = det.get_device_info()
    assert "device" in info
