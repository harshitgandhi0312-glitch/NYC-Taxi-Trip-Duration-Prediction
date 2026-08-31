"""
src/db.py
---------
Minimal DuckDB data-access helper for the NYC Taxi Trip Duration project.

Reads dataset paths from config.local.ini (git-ignored, machine-specific).
Falls back to config.ini if the local override is absent.

Usage
-----
    from src.db import get_connection, get_taxi_paths

    con = get_connection()
    paths = get_taxi_paths()
    df = con.execute(f"SELECT COUNT(*) FROM read_parquet({paths})").df()
"""

import configparser
from pathlib import Path

# ── Locate config files ───────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent          # project root
_LOCAL_CFG = _ROOT / "config.local.ini"
_DEFAULT_CFG = _ROOT / "config.ini"


def _load_config() -> configparser.ConfigParser:
    """Load config.local.ini if present, otherwise fall back to config.ini."""
    cfg = configparser.ConfigParser()
    if _LOCAL_CFG.exists():
        cfg.read(_LOCAL_CFG)
    elif _DEFAULT_CFG.exists():
        cfg.read(_DEFAULT_CFG)
    else:
        raise FileNotFoundError(
            "No config file found. Copy config.ini to config.local.ini "
            "and set the correct dataset paths."
        )
    return cfg


def get_taxi_paths() -> list[str]:
    """Return [path_2023, path_2024] as a Python list of strings."""
    cfg = _load_config()
    paths = [
        cfg["data"]["taxi_2023"],
        cfg["data"]["taxi_2024"],
    ]
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Parquet file not found: {p}")
    return paths


def get_connection():
    """Return a DuckDB in-memory connection (read-only queries against Parquet)."""
    try:
        import duckdb
    except ImportError:
        raise ImportError("duckdb is not installed. Run: pip install duckdb")
    return duckdb.connect()
