"""Load per-station Parquet data for the dashboard.

Reads from data/per_station/*.parquet (built by build_station_datasets.py).
Exposes the same API as the previous data_loader so layout/callbacks/figures
require no structural changes.

Column mapping (parquet → dashboard):
  time            → valid_time
  forcoast_p82_m  → dkss_p82_m   (alias for backward-compat with figures.py)
  error_m         → computed as dkss_p82_m - tg_obs_m
  bias_m          → temporal mean of error_m over the full record (per station)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path("/Users/nicolocaron/Desktop/MASTER PROJECT/DATASET COMPLETI 6 STAZIONI INIZIALI")
_PER_STATION_DIR = _DATA_DIR / "per_station"

# ── Module-level cache ───────────────────────────────────────────────────
_station_cache: dict[str, pd.DataFrame] = {}
_stations: list[dict] | None = None
_all_df: pd.DataFrame | None = None
# Per-station temporal mean bias: {station_id: bias_m}
_station_bias: dict[str, float] = {}


def _load_station_parquets() -> None:
    """Scan per_station/ dir, load all parquets, build unified cache."""
    global _stations, _all_df

    if not _PER_STATION_DIR.exists():
        raise FileNotFoundError(
            f"Per-station data directory not found: {_PER_STATION_DIR}\n"
            "Run  python build_station_datasets.py  first."
        )

    parquet_files = sorted(_PER_STATION_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in {_PER_STATION_DIR}\n"
            "Run  python build_station_datasets.py  first."
        )

    parts: list[pd.DataFrame] = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        # Normalise column names for dashboard compatibility
        df = df.rename(columns={
            "time": "valid_time",
            "forcoast_p82_m": "dkss_p82_m",
        })
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df["station_id"] = df["station_id"].astype(str)
        df["error_m"] = df["dkss_p82_m"] - df["tg_obs_m"]
        # Compute and store temporal mean bias for this station
        sid = str(df["station_id"].iloc[0])
        _station_bias[sid] = float(df["error_m"].mean())
        df["bias_m"] = _station_bias[sid]   # constant column for convenience
        # Per-station cache keyed by str ID
        _station_cache[sid] = df
        parts.append(df)

    _all_df = pd.concat(parts, ignore_index=True)

    # Build stations list from parquet metadata (no stations.json needed)
    station_meta = (
        _all_df[["station_id", "station_name", "lat", "lon"]]
        .drop_duplicates(subset="station_id")
        .sort_values("station_name")
    )
    _stations = [
        {
            "id": str(row["station_id"]),
            "name": row["station_name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }
        for _, row in station_meta.iterrows()
    ]


def _ensure_loaded() -> None:
    if _all_df is None:
        _load_station_parquets()


def get_stations() -> list[dict]:
    """Return list of station dicts with keys: id, name, lat, lon."""
    _ensure_loaded()
    return _stations  # type: ignore[return-value]


def get_dataframe() -> pd.DataFrame:
    """Return the full concatenated DataFrame (cached in memory)."""
    _ensure_loaded()
    return _all_df  # type: ignore[return-value]


def get_station_data(
    station_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return data for one station within an optional time window."""
    _ensure_loaded()
    sid = str(station_id)
    df = _station_cache.get(sid)
    if df is None:
        df = _all_df[_all_df["station_id"] == sid].copy()  # type: ignore[index]
    else:
        df = df.copy()
    if start:
        df = df[df["valid_time"] >= pd.Timestamp(start)]
    if end:
        df = df[df["valid_time"] <= pd.Timestamp(end)]
    return df


def get_time_range() -> tuple[str, str]:
    """Return (min_date, max_date) strings from the dataset."""
    _ensure_loaded()
    return (
        _all_df["valid_time"].min().strftime("%Y-%m-%d"),  # type: ignore[index]
        _all_df["valid_time"].max().strftime("%Y-%m-%d"),  # type: ignore[index]
    )


def get_station_bias(station_id: str) -> float:
    """Return the temporal mean bias (Model − TG) for *station_id* over the full record.

    This value can be subtracted from ``error_m`` to obtain a bias-corrected error.
    """
    _ensure_loaded()
    return _station_bias.get(str(station_id), 0.0)
