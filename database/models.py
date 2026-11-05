"""
Database schema helpers – Stage 11
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS authorized_drones (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    drone_id        TEXT UNIQUE NOT NULL,
    model           TEXT,
    type            TEXT,
    description     TEXT,
    status          TEXT DEFAULT 'AUTHORIZED',
    registered_at   TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_authorized_model
    ON authorized_drones(model);

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    track_id        INTEGER,
    bbox            TEXT,
    confidence      REAL,
    class_name      TEXT,
    class_id        INTEGER,
    frame_idx       INTEGER
);

CREATE TABLE IF NOT EXISTS tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    track_id        INTEGER,
    status          TEXT,
    bbox            TEXT,
    confidence      REAL,
    class_name      TEXT,
    centroid        TEXT,
    hits            INTEGER,
    age             INTEGER
);

CREATE TABLE IF NOT EXISTS classifications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    track_id        INTEGER,
    class_name      TEXT,
    confidence      REAL,
    status          TEXT,
    model           TEXT
);

CREATE TABLE IF NOT EXISTS rf_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    detected        INTEGER,
    frequency_hz    REAL,
    signal_strength_dbm REAL,
    bandwidth_hz    REAL,
    status          TEXT,
    backend         TEXT
);

CREATE TABLE IF NOT EXISTS range_estimates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    track_id        INTEGER,
    distance_m      REAL,
    confidence      REAL,
    method          TEXT,
    status          TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    track_id        INTEGER,
    alert_level     TEXT,
    reason          TEXT,
    status          TEXT DEFAULT 'active',
    auth_status     TEXT,
    estimated_range_m REAL
);

CREATE TABLE IF NOT EXISTS system_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    event_type      TEXT,
    message         TEXT,
    data            TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    track_id        INTEGER,
    filepath        TEXT,
    alert_level     TEXT,
    class_name      TEXT,
    confidence      REAL
);
"""
