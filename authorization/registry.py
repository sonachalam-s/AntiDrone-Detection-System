"""
Authorization Registry – Stage 4

Determines whether a visually classified drone is:
  - AUTHORIZED
  - UNKNOWN
  - UNCONFIRMED

Rules:
- Match is performed primarily on the classification result (model / type).
- Never claims that an UNKNOWN drone is hostile.
- If classifier is offline → status becomes UNCONFIRMED.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from database.database import init_db, get_connection

logger = logging.getLogger(__name__)


class AuthorizationRegistry:
    """
    SQLite-backed registry of known / authorized drones.
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = init_db(db_path)
        logger.info(f"AuthorizationRegistry ready | db={self.db_path}")

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    def register(
        self,
        drone_id: str,
        model: str,
        type_: str = "",
        description: str = "",
        status: str = "AUTHORIZED",
    ) -> bool:
        """Insert or update an authorized drone entry."""
        try:
            with get_connection(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO authorized_drones
                        (drone_id, model, type, description, status, registered_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                    ON CONFLICT(drone_id) DO UPDATE SET
                        model       = excluded.model,
                        type        = excluded.type,
                        description = excluded.description,
                        status      = excluded.status,
                        updated_at  = datetime('now')
                    """,
                    (drone_id, model, type_, description, status),
                )
            logger.info(f"Registered / updated drone: {drone_id} ({model})")
            return True
        except Exception as e:
            logger.error(f"Failed to register drone {drone_id}: {e}")
            return False

    def remove(self, drone_id: str) -> bool:
        try:
            with get_connection(self.db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM authorized_drones WHERE drone_id = ?",
                    (drone_id,),
                )
            return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to remove drone {drone_id}: {e}")
            return False

    def list_all(self) -> List[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM authorized_drones ORDER BY registered_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def find_by_model(self, model: str) -> Optional[Dict[str, Any]]:
        """Return first matching authorized entry for a given model name."""
        if not model or model in ("N/A", "unknown_drone", "invalid_crop"):
            return None
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM authorized_drones
                WHERE lower(model) = lower(?)
                  AND status = 'AUTHORIZED'
                LIMIT 1
                """,
                (model,),
            ).fetchone()
        return dict(row) if row else None

    def find_by_drone_id(self, drone_id: str) -> Optional[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM authorized_drones WHERE drone_id = ?",
                (drone_id,),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Authorization decision
    # ------------------------------------------------------------------
    def authorize(
        self,
        classification: Dict[str, Any],
        track: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Decide authorization status for one track / classification result.

        Returns:
        {
            "status": "AUTHORIZED" | "UNKNOWN" | "UNCONFIRMED",
            "drone_id": str or None,
            "matched_model": str or None,
            "reason": str,
            "confidence": float   # classification confidence
        }
        """
        result = {
            "status": "UNCONFIRMED",
            "drone_id": None,
            "matched_model": None,
            "reason": "",
            "confidence": 0.0,
        }

        if not classification:
            result["reason"] = "No classification data"
            return result

        cls_status = classification.get("status", "CLASSIFIER OFFLINE")
        cls_name = classification.get("class_name", "N/A")
        cls_conf = float(classification.get("confidence", 0.0))

        result["confidence"] = cls_conf

        # Classifier offline → cannot decide
        if cls_status != "ok":
            result["status"] = "UNCONFIRMED"
            result["reason"] = "Classifier offline – visual type unknown"
            return result

        # Very low confidence → treat as unconfirmed
        if cls_conf < 0.40:
            result["status"] = "UNCONFIRMED"
            result["reason"] = f"Low classification confidence ({cls_conf:.2f})"
            return result

        # Look up in registry by model / type name
        match = self.find_by_model(cls_name)
        if match:
            result["status"] = "AUTHORIZED"
            result["drone_id"] = match["drone_id"]
            result["matched_model"] = match["model"]
            result["reason"] = f"Matched authorized model '{match['model']}' → {match['drone_id']}"
            return result

        # No match → UNKNOWN (not hostile, just unknown)
        result["status"] = "UNKNOWN"
        result["reason"] = f"No authorized entry for classified type '{cls_name}'"
        return result

    def authorize_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run authorization on every track.
        Adds key: authorization = {status, drone_id, reason, ...}
        Also adds convenience top-level key: auth_status
        """
        for trk in tracks:
            cls = trk.get("classification", {})
            auth = self.authorize(cls, track=trk)
            trk["authorization"] = auth
            trk["auth_status"] = auth["status"]
        return tracks

    # ------------------------------------------------------------------
    # Seed helpers (for development / demo)
    # ------------------------------------------------------------------
    def seed_demo_entries(self):
        """Insert a few example authorized drones for testing."""
        demos = [
            ("DRONE-001", "quadcopter", "quadcopter", "Company survey drone"),
            ("DRONE-002", "racing_drone", "racing_drone", "Racing team Alpha"),
            ("DRONE-003", "hexacopter", "hexacopter", "Inspection hex"),
        ]
        for drone_id, model, type_, desc in demos:
            self.register(drone_id, model, type_, desc)
        logger.info("Demo authorized drones seeded")
