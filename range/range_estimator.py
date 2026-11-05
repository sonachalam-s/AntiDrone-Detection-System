"""
Monocular Range Estimator – Stage 5

Estimates approximate distance to a tracked object using the
pinhole-camera / similar-triangles model:

    distance = (real_width * focal_length_px) / pixel_width

IMPORTANT:
- This is an ESTIMATE only.
- Never presented as exact range.
- Designed so a future stereo / LiDAR / radar module can replace it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
import numpy as np

logger = logging.getLogger(__name__)


# Approximate real-world widths (metres) for common classes
DEFAULT_OBJECT_WIDTHS = {
    "quadcopter": 0.35,       # typical 5-inch / DJI Mini class
    "hexacopter": 0.55,
    "fixed_wing": 1.20,
    "racing_drone": 0.25,
    "unknown_drone": 0.40,    # generic fallback
    "drone": 0.40,
    "airplane": 15.0,         # COCO fallback (very rough)
    "bird": 0.30,
    "person": 0.50,
    "object": 0.40,
}


class RangeEstimator:
    """
    Monocular distance estimation from bounding-box width.
    """

    def __init__(
        self,
        calibration_path: Optional[str | Path] = None,
        object_widths: Optional[Dict[str, float]] = None,
        default_width_m: float = 0.40,
        min_pixel_width: int = 8,
        max_reasonable_m: float = 2000.0,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.default_width_m = default_width_m
        self.min_pixel_width = min_pixel_width
        self.max_reasonable_m = max_reasonable_m
        self.object_widths = object_widths or DEFAULT_OBJECT_WIDTHS.copy()

        # Camera intrinsics (will be loaded or set to defaults)
        self.focal_length_px: float = 1000.0   # approximate
        self.image_width: int = 1280
        self.image_height: int = 720
        self.sensor_width_mm: float = 6.17     # typical 1/2.3" sensor
        self.focal_length_mm: float = 4.5
        self.method = "monocular_estimation"
        self.status = "ok"

        if calibration_path:
            self._load_calibration(calibration_path)
        else:
            logger.info(
                "No calibration file provided – using default intrinsics "
                f"(focal≈{self.focal_length_px:.0f} px)"
            )

        if not self.enabled:
            self.status = "RANGE ESTIMATOR OFFLINE"
            logger.info("Range estimator disabled by config")

    def _load_calibration(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            logger.warning(f"Calibration file not found: {path} – using defaults")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            cam = data.get("camera", data)

            self.image_width = int(cam.get("image_width", self.image_width))
            self.image_height = int(cam.get("image_height", self.image_height))
            self.sensor_width_mm = float(cam.get("sensor_width_mm", self.sensor_width_mm))
            self.focal_length_mm = float(cam.get("focal_length_mm", self.focal_length_mm))

            # Prefer explicit focal length in pixels if provided
            if "focal_length_px" in cam:
                self.focal_length_px = float(cam["focal_length_px"])
            else:
                # Convert mm → pixels: f_px = (f_mm / sensor_width_mm) * image_width
                if self.sensor_width_mm > 0:
                    self.focal_length_px = (
                        self.focal_length_mm / self.sensor_width_mm
                    ) * self.image_width

            # Optional per-class widths override
            if "object_widths_m" in data:
                self.object_widths.update(data["object_widths_m"])

            logger.info(
                f"Calibration loaded from {path.name} | "
                f"focal={self.focal_length_px:.1f} px | "
                f"img={self.image_width}x{self.image_height}"
            )
        except Exception as e:
            logger.error(f"Failed to load calibration: {e} – using defaults")

    def _get_real_width(self, class_name: str) -> float:
        key = (class_name or "unknown_drone").lower().strip()
        return self.object_widths.get(key, self.default_width_m)

    def estimate(
        self,
        bbox: List[int],
        class_name: str = "unknown_drone",
        classification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate range for a single bounding box.

        Returns:
        {
            "distance_m": float or None,
            "confidence": float,          # 0-1 rough confidence
            "method": "monocular_estimation",
            "status": "ok" | "RANGE ESTIMATOR OFFLINE" | "invalid",
            "pixel_width": int,
            "assumed_width_m": float
        }
        """
        result = {
            "distance_m": None,
            "confidence": 0.0,
            "method": self.method,
            "status": self.status,
            "pixel_width": 0,
            "assumed_width_m": self.default_width_m,
        }

        if not self.enabled:
            result["status"] = "RANGE ESTIMATOR OFFLINE"
            return result

        if not bbox or len(bbox) < 4:
            result["status"] = "invalid"
            return result

        x1, y1, x2, y2 = bbox
        pixel_w = max(0, x2 - x1)
        pixel_h = max(0, y2 - y1)
        result["pixel_width"] = pixel_w

        if pixel_w < self.min_pixel_width:
            result["status"] = "invalid"
            result["confidence"] = 0.0
            return result

        # Prefer classification class name if available and valid
        use_name = class_name
        if classification and classification.get("status") == "ok":
            use_name = classification.get("class_name", class_name)

        real_w = self._get_real_width(use_name)
        result["assumed_width_m"] = real_w

        # Core formula
        distance = (real_w * self.focal_length_px) / pixel_w

        # Sanity clamp
        if distance <= 0 or distance > self.max_reasonable_m:
            result["status"] = "invalid"
            result["distance_m"] = None
            result["confidence"] = 0.0
            return result

        # Rough confidence: higher when object is reasonably sized in frame
        # and we have a known class width
        size_ratio = pixel_w / max(self.image_width, 1)
        conf = min(1.0, 0.35 + size_ratio * 2.5)
        if use_name in self.object_widths and use_name not in ("unknown_drone", "object"):
            conf = min(1.0, conf + 0.15)
        conf = round(float(conf), 2)

        result.update({
            "distance_m": round(float(distance), 1),
            "confidence": conf,
            "status": "ok",
        })
        return result

    def estimate_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Attach range estimate to every track.
        Adds key: range = {distance_m, confidence, ...}
        Also top-level convenience: estimated_range_m
        """
        for trk in tracks:
            bbox = trk.get("bbox")
            class_name = trk.get("class_name", "unknown_drone")
            classification = trk.get("classification")

            rng = self.estimate(bbox, class_name, classification)
            trk["range"] = rng
            trk["estimated_range_m"] = rng.get("distance_m")

        return tracks

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "method": self.method,
            "focal_length_px": self.focal_length_px,
            "image_size": f"{self.image_width}x{self.image_height}",
        }
