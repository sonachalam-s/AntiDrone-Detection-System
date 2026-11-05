"""Unit tests – ThreatEngine"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threat.threat_engine import ThreatEngine


def _track(**kwargs):
    base = {
        "track_id": 1,
        "authorization": {"status": "UNKNOWN"},
        "auth_status": "UNKNOWN",
        "restricted_zone": {"inside": False},
        "estimated_range_m": 200,
        "hits": 5,
        "age": 5,
        "classification": {"status": "ok", "confidence": 0.8},
        "fused": {"rf": {"status": "RF SENSOR OFFLINE", "detected": False}},
    }
    base.update(kwargs)
    return base


def test_authorized_is_low():
    eng = ThreatEngine()
    t = _track(authorization={"status": "AUTHORIZED"}, auth_status="AUTHORIZED")
    r = eng.assess(t)
    assert r["level"] == "LOW"
    assert r["alert"] is False


def test_unknown_close_elevated():
    eng = ThreatEngine()
    t = _track(
        authorization={"status": "UNKNOWN"},
        auth_status="UNKNOWN",
        restricted_zone={"inside": True, "zone_name": "zone_a"},
        estimated_range_m=30,
        hits=20,
        age=20,
    )
    r = eng.assess(t)
    assert r["level"] in ("HIGH", "CRITICAL", "MEDIUM")
    assert len(r["reasons"]) > 0


def test_assess_tracks():
    eng = ThreatEngine()
    tracks = [
        _track(track_id=1),
        _track(track_id=2, auth_status="AUTHORIZED",
               authorization={"status": "AUTHORIZED"}),
    ]
    out = eng.assess_tracks(tracks)
    assert all("threat" in t for t in out)
    assert all("threat_level" in t for t in out)


def test_disabled():
    eng = ThreatEngine(enabled=False)
    r = eng.assess(_track())
    assert r["level"] == "LOW"
    assert r["alert"] is False
