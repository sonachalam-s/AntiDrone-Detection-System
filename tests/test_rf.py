"""Unit tests – RFMonitor (offline path)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rf.rf_monitor import RFMonitor


def test_rf_offline_by_default():
    """Without SDR hardware, monitor must report OFFLINE – never fake values."""
    mon = RFMonitor(enabled=True)
    obs = mon.poll(force=True)
    assert obs["status"] == "RF SENSOR OFFLINE"
    assert obs["detected"] is False
    assert obs["frequency_hz"] is None
    assert obs["signal_strength_dbm"] is None
    assert mon.is_online() is False


def test_rf_disabled():
    mon = RFMonitor(enabled=False)
    obs = mon.get_latest()
    assert obs["status"] == "RF SENSOR OFFLINE"


def test_rf_status_dict():
    mon = RFMonitor(enabled=True)
    st = mon.get_status()
    assert "status" in st
    assert "libraries" in st
