"""Unit tests – FastAPI endpoints"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from dashboard.api import app
from dashboard.state import STATE


client = TestClient(app)


def test_status():
    STATE.update_system(status="running", camera_fps=30.0)
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "data" in body


def test_tracks_empty():
    STATE.set_tracks([])
    r = client.get("/api/tracks")
    assert r.status_code == 200
    assert body_ok(r)


def test_rf():
    STATE.set_rf({"status": "RF SENSOR OFFLINE", "detected": False})
    r = client.get("/api/rf")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["status"] == "RF SENSOR OFFLINE"


def test_alerts():
    STATE.set_alerts([])
    r = client.get("/api/alerts")
    assert r.status_code == 200


def test_events():
    STATE.add_event("test", "unit test event")
    r = client.get("/api/events")
    assert r.status_code == 200
    assert len(r.json()["data"]) >= 1


def test_dashboard_root():
    r = client.get("/")
    # Either HTML dashboard or JSON fallback
    assert r.status_code == 200


def body_ok(r):
    j = r.json()
    return j.get("ok") is True
