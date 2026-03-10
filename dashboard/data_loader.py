"""Load pre-computed Parquet data and station metadata for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PARQUET = _DATA_DIR / "hdm_tg_obs_all_stations.parquet"
_STATIONS_JSON = _DATA_DIR / "stations.json"

# ── Module-level cache ───────────────────────────────────────────────────
_df: pd.DataFrame | None = None
_stations: list[dict] | None = None


def _ensure_loaded():
    global _df, _stations
    if _df is None:
        if not _PARQUET.exists():
            raise FileNotFoundError(
                f"Data file not found: {_PARQUET}\n"
                "Run  python prepare_data.py  first."
            )
        _df = pd.read_parquet(_PARQUET)
        _df["valid_time"] = pd.to_datetime(_df["valid_time"])
    if _stations is None:
        with open(_STATIONS_JSON) as f:
            _stations = json.load(f)


def get_stations() -> list[dict]:
    """Return list of station dicts with keys: id, name, lat, lon."""
    _ensure_loaded()
    return _stations  # type: ignore[return-value]


def get_dataframe() -> pd.DataFrame:
    """Return the full DataFrame (cached in memory)."""
    _ensure_loaded()
    return _df  # type: ignore[return-value]


def get_station_data(
    station_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Slice the data for one station within an optional time window."""
    _ensure_loaded()
    assert _df is not None
    mask = _df["station_id"] == station_id
    if start:
        ts = pd.Timestamp(start)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        mask &= _df["valid_time"] >= ts
    if end:
        ts = pd.Timestamp(end)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        mask &= _df["valid_time"] <= ts
    return _df.loc[mask].copy()


def get_time_range() -> tuple[str, str]:
    """Return (min_date, max_date) strings from the dataset."""
    _ensure_loaded()
    assert _df is not None
    return (
        _df["valid_time"].min().strftime("%Y-%m-%d"),
        _df["valid_time"].max().strftime("%Y-%m-%d"),
    )
