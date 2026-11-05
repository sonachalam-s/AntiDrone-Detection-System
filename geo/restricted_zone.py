"""
Restricted Zone Manager – used by the Threat Engine (Stage 8)

Supports:
- Circular zone (centre + radius in image pixels or metres)
- Polygon zone (list of points)

Because GPS is often unavailable, zones can be defined in
image-plane coordinates for development. When real geo data
becomes available the same interface can consume lat/lon.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RestrictedZoneManager:
    """
    Check whether a track centroid falls inside any configured restricted zone.
    """

    def __init__(self, zones: Optional[List[Dict[str, Any]]] = None):
        """
        zones example:
        [
            {
                "name": "runway_approach",
                "type": "circle",
                "center": [640, 360],      # image pixels or (lat, lon)
                "radius": 200,             # pixels or metres
                "coordinate_system": "image"  # or "geo"
            },
            {
                "name": "hangar",
                "type": "polygon",
                "points": [[100,100], [400,100], [400,300], [100,300]],
                "coordinate_system": "image"
            }
        ]
        """
        self.zones = zones or []
        logger.info("RestrictedZoneManager ready | zones=%d", len(self.zones))

    def is_inside(
        self,
        centroid: Optional[Tuple[float, float]],
        estimated_range_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            "inside": bool,
            "zone_name": str | None,
            "reason": str
        }
        """
        result = {"inside": False, "zone_name": None, "reason": "No restricted zones configured"}

        if not self.zones:
            return result

        if centroid is None:
            result["reason"] = "No centroid available"
            return result

        cx, cy = centroid

        for zone in self.zones:
            ztype = zone.get("type", "circle")
            name = zone.get("name", "unnamed")
            coord_sys = zone.get("coordinate_system", "image")

            # For now we only evaluate image-plane zones
            if coord_sys != "image":
                continue

            if ztype == "circle":
                center = zone.get("center", [0, 0])
                radius = float(zone.get("radius", 0))
                dx = cx - center[0]
                dy = cy - center[1]
                if (dx * dx + dy * dy) <= (radius * radius):
                    return {
                        "inside": True,
                        "zone_name": name,
                        "reason": f"Inside circular zone '{name}'",
                    }

            elif ztype == "polygon":
                points = zone.get("points", [])
                if len(points) >= 3 and self._point_in_polygon(cx, cy, points):
                    return {
                        "inside": True,
                        "zone_name": name,
                        "reason": f"Inside polygon zone '{name}'",
                    }

        result["reason"] = "Outside all restricted zones"
        return result

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
        """Ray-casting algorithm."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            ):
                inside = not inside
            j = i
        return inside

    def annotate_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach restricted_zone info to every track."""
        for trk in tracks:
            centroid = trk.get("centroid")
            range_m = trk.get("estimated_range_m")
            rz = self.is_inside(centroid, range_m)
            trk["restricted_zone"] = rz
            trk["in_restricted_zone"] = rz["inside"]
        return tracks
