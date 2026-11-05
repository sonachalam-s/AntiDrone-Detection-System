"""
DeepSORT-based multi-object tracker for drones.
Assigns persistent Track IDs, maintains trajectory history,
and handles temporary occlusions.
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning(
        "deep_sort_realtime not installed. Tracking will be disabled. "
        "Install with: pip install deep-sort-realtime"
    )


class DroneTracker:
    """
    Multi-drone tracker using DeepSORT.

    Each active track contains:
    - track_id
    - bbox
    - confidence
    - class_name
    - age / hits
    - trajectory history (image-plane centroids)
    - approximate velocity (pixels / frame)
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: Optional[int] = 100,
        embedder: str = "mobilenet",
        half: bool = False,
        bgr: bool = True,
        embedder_gpu: bool = True,
        trajectory_maxlen: int = 60,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.trajectory_maxlen = trajectory_maxlen

        self.tracker = None
        self.enabled = False

        if not DEEPSORT_AVAILABLE:
            logger.error("DeepSORT unavailable – tracking disabled")
            return

        try:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                max_cosine_distance=max_cosine_distance,
                nn_budget=nn_budget,
                embedder=embedder,
                half=half,
                bgr=bgr,
                embedder_gpu=embedder_gpu,
            )
            self.enabled = True
            logger.info(
                f"DeepSORT tracker ready | max_age={max_age} | n_init={n_init}"
            )
        except Exception as e:
            logger.error(f"Failed to initialise DeepSORT: {e}")
            self.enabled = False

        # Per-track history: track_id → deque of (cx, cy, timestamp)
        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trajectory_maxlen)
        )
        # Last known velocity (dx, dy) in pixels per frame
        self.velocities: Dict[int, Tuple[float, float]] = {}
        # Last update time
        self.last_seen: Dict[int, float] = {}

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with current YOLO detections.

        Args:
            detections: list of dicts from YOLODroneDetector
                        [{"bbox": [x1,y1,x2,y2], "confidence": float, "class_name": str, ...}]
            frame: optional BGR frame (used by DeepSORT embedder)

        Returns:
            List of tracked objects:
            {
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_name": str,
                "class_id": int,
                "age": int,
                "hits": int,
                "time_since_update": int,
                "centroid": (cx, cy),
                "velocity": (vx, vy),          # pixels / frame
                "trajectory": [(cx, cy), ...], # recent history
                "status": "confirmed" | "tentative"
            }
        """
        if not self.enabled or self.tracker is None:
            # Fallback: return detections with temporary sequential IDs
            return self._fallback_tracks(detections)

        # Convert to DeepSORT format: list of ([left, top, w, h], confidence, class_name)
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            conf = det.get("confidence", 0.0)
            class_name = det.get("class_name", "object")
            raw_dets.append(([x1, y1, w, h], conf, class_name))

        # Run DeepSORT
        tracks = self.tracker.update_tracks(raw_dets, frame=frame)

        now = time.time()
        active: List[Dict[str, Any]] = []

        for track in tracks:
            if not track.is_confirmed():
                # Still tentative – optionally skip or mark as tentative
                status = "tentative"
            else:
                status = "confirmed"

                        # deep_sort_realtime may return int or str; normalize to int
            try:
                track_id = int(track.track_id)
            except (TypeError, ValueError):
                track_id = abs(hash(str(track.track_id))) % (10**9)
            ltrb = track.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Update trajectory
            traj = self.trajectories[track_id]
            traj.append((cx, cy))

            # Estimate simple velocity (pixels per frame) from last two points
            vx, vy = 0.0, 0.0
            if len(traj) >= 2:
                prev_cx, prev_cy = traj[-2]
                vx = cx - prev_cx
                vy = cy - prev_cy
            self.velocities[track_id] = (vx, vy)
            self.last_seen[track_id] = now

            # Find original detection confidence / class if possible
            conf = 0.0
            class_name = "object"
            class_id = -1
            for det in detections:
                dx1, dy1, dx2, dy2 = det["bbox"]
                # Simple IoU-ish match
                if abs(dx1 - x1) < 30 and abs(dy1 - y1) < 30:
                    conf = det.get("confidence", 0.0)
                    class_name = det.get("class_name", "object")
                    class_id = det.get("class_id", -1)
                    break

            active.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": class_name,
                "class_id": class_id,
                "age": track.age,
                "hits": track.hits,
                "time_since_update": track.time_since_update,
                "centroid": (cx, cy),
                "velocity": (round(vx, 2), round(vy, 2)),
                "trajectory": list(traj),
                "status": status,
            })

        # Clean up old trajectories
        self._cleanup_stale(now)

        return active

    def _fallback_tracks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """When DeepSORT is unavailable, invent temporary sequential IDs."""
        results = []
        for i, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            results.append({
                "track_id": i,
                "bbox": det["bbox"],
                "confidence": det.get("confidence", 0.0),
                "class_name": det.get("class_name", "object"),
                "class_id": det.get("class_id", -1),
                "age": 1,
                "hits": 1,
                "time_since_update": 0,
                "centroid": (cx, cy),
                "velocity": (0.0, 0.0),
                "trajectory": [(cx, cy)],
                "status": "fallback",
            })
        return results

    def _cleanup_stale(self, now: float, max_idle: float = 10.0):
        """Remove trajectories of tracks that have not been seen recently."""
        stale = [tid for tid, t in self.last_seen.items() if now - t > max_idle]
        for tid in stale:
            self.trajectories.pop(tid, None)
            self.velocities.pop(tid, None)
            self.last_seen.pop(tid, None)

    def get_active_track_ids(self) -> List[int]:
        return list(self.trajectories.keys())

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        return list(self.trajectories.get(track_id, []))

    def reset(self):
        """Clear all tracks and history."""
        if self.tracker is not None:
            # deep_sort_realtime does not expose a clean reset; recreate
            try:
                self.tracker = DeepSort(
                    max_age=self.max_age,
                    n_init=self.n_init,
                )
            except Exception:
                pass
        self.trajectories.clear()
        self.velocities.clear()
        self.last_seen.clear()
        logger.info("Tracker reset")
