"""Unit tests – RestrictedZoneManager"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo.restricted_zone import RestrictedZoneManager


def test_no_zones():
    mgr = RestrictedZoneManager(zones=[])
    r = mgr.is_inside((100, 100))
    assert r["inside"] is False


def test_circle_zone():
    zones = [{
        "name": "center",
        "type": "circle",
        "center": [100, 100],
        "radius": 50,
        "coordinate_system": "image",
    }]
    mgr = RestrictedZoneManager(zones=zones)
    assert mgr.is_inside((100, 100))["inside"] is True
    assert mgr.is_inside((200, 200))["inside"] is False


def test_polygon_zone():
    zones = [{
        "name": "box",
        "type": "polygon",
        "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "coordinate_system": "image",
    }]
    mgr = RestrictedZoneManager(zones=zones)
    assert mgr.is_inside((50, 50))["inside"] is True
    assert mgr.is_inside((150, 150))["inside"] is False


def test_annotate_tracks():
    zones = [{
        "name": "z",
        "type": "circle",
        "center": [50, 50],
        "radius": 30,
        "coordinate_system": "image",
    }]
    mgr = RestrictedZoneManager(zones=zones)
    tracks = [{"centroid": (50, 50)}, {"centroid": (200, 200)}]
    out = mgr.annotate_tracks(tracks)
    assert out[0]["in_restricted_zone"] is True
    assert out[1]["in_restricted_zone"] is False
