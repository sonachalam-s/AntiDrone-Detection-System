"""
Image preprocessing helpers for EfficientNet classification.
"""

from typing import Tuple
import cv2
import numpy as np


def crop_and_pad(
    frame: np.ndarray,
    bbox: list,
    target_size: int = 224,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Crop bbox and resize with letterbox padding to keep aspect ratio.
    Returns BGR image of shape (target_size, target_size, 3).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    scale = min(target_size / cw, target_size / ch)
    nw, nh = int(cw * scale), int(ch * scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas
