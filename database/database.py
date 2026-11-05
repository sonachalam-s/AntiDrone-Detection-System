"""
SQLite helper – Stage 4 + Stage 11
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from database.models import SCHEMA_SQL

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "antidrone.db"


def get_db_path(custom: Optional[str | Path] = None) -> Path:
    if custom:
        return Path(custom)
    return DEFAULT_DB_PATH


def init_db(db_path: Optional[str | Path] = None) -> Path:
    """Create database file and all required tables."""
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info(f"Database initialised: {path}")
    finally:
        conn.close()

    return path


@contextmanager
def get_connection(db_path: Optional[str | Path] = None):
    """Context manager yielding a SQLite connection with row_factory."""
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
