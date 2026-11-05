"""Unit tests – AuthorizationRegistry"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authorization.registry import AuthorizationRegistry


def test_register_and_authorize(tmp_path):
    db = tmp_path / "test.db"
    reg = AuthorizationRegistry(db_path=db)
    assert reg.register("DRONE-001", "quadcopter", "quadcopter", "test")

    # Match
    cls = {"status": "ok", "class_name": "quadcopter", "confidence": 0.9}
    result = reg.authorize(cls)
    assert result["status"] == "AUTHORIZED"
    assert result["drone_id"] == "DRONE-001"

    # No match
    cls2 = {"status": "ok", "class_name": "fixed_wing", "confidence": 0.9}
    result2 = reg.authorize(cls2)
    assert result2["status"] == "UNKNOWN"

    # Classifier offline
    cls3 = {"status": "CLASSIFIER OFFLINE", "class_name": "N/A", "confidence": 0}
    result3 = reg.authorize(cls3)
    assert result3["status"] == "UNCONFIRMED"


def test_remove(tmp_path):
    db = tmp_path / "test2.db"
    reg = AuthorizationRegistry(db_path=db)
    reg.register("DRONE-X", "racing_drone")
    assert reg.remove("DRONE-X") is True
    assert reg.find_by_drone_id("DRONE-X") is None


def test_list_all(tmp_path):
    db = tmp_path / "test3.db"
    reg = AuthorizationRegistry(db_path=db)
    reg.seed_demo_entries()
    rows = reg.list_all()
    assert len(rows) >= 3
