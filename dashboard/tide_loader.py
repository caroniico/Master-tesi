"""DTU10 tidal model loader for the dashboard.

Reads GOT4.7-format (.d) grids from the DTU10 tidal model volume and
predicts tidal height at an arbitrary (lat, lon) for a time series.

Grid format (ASCII, 7 header lines):
  Line 0: "GOT4.7  <constituent>"
  Line 1: "<constituent> tide"
  Line 2: "<ny> <nx>"
  Line 3: "<lat_min> <lat_max>"
  Line 4: "<lon_min> <lon_max>"
  Line 5: "<undef_value>"
  Line 6: "<undef_value>" (repeated)
  Lines 7 .. 7+ny-1       : amplitude grid [cm], ny rows of nx values
  Lines 7+ny .. 7+ny+5    : phase header (same 6 header lines)
  Lines 7+ny+7 .. end     : phase grid [deg], ny rows of nx values

Tidal prediction uses pyTMD.constituents.arguments() for astronomical
arguments + nodal corrections, matching the notebook implementation.

Module-level dict ``_GRID_CACHE`` prevents repeated disk reads.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# ── Configuration ─────────────────────────────────────────────────────────
_GOT47_DIR = Path("/Volumes/DTU10_TIDEMODEL/OCEAN_TIDE_GRIDS/GOT47_FORMAT")

# Constituents to include (short periods only — avoids aliasing on hourly data)
CONSTITUENTS: dict[str, str] = {
    "m2": "M2.d",
    "s2": "S2.d",
    "k2": "K2.d",
    "n2": "N2.d",
    "k1": "K1.d",
    "o1": "O1.d",
    "p1": "P1.d",
    "q1": "Q1.d",
    "m4": "M4.d",
}

# Module-level grid cache: {filename_stem → (amp_grid, phase_grid, lat_vec, lon_vec)}
_GRID_CACHE: dict[str, tuple] = {}

# pyTMD ALTIM epoch for GOT corrections
_PYTMD_EPOCH = pd.Timestamp("1992-01-01")
_PYTMD_MJD0  = 48622.0   # MJD of 1992-01-01


# ── Grid reader ───────────────────────────────────────────────────────────

def _read_got47_grid(filepath: str | Path) -> tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """Read one DTU10 GOT4.7 .d file.

    Returns
    -------
    amp   : 2-D array  [ny × nx]  amplitude in cm
    phase : 2-D array  [ny × nx]  phase in degrees
    lat   : 1-D array  [ny]       latitudes  (ascending)
    lon   : 1-D array  [nx]       longitudes (ascending)
    """
    key = str(filepath)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    with open(filepath, "r") as fh:
        lines = fh.readlines()

    ny, nx       = int(lines[2].split()[0]), int(lines[2].split()[1])
    lat_min, lat_max = float(lines[3].split()[0]), float(lines[3].split()[1])
    lon_min, lon_max = float(lines[4].split()[0]), float(lines[4].split()[1])
    undef        = float(lines[5].strip())

    lat_vec = np.linspace(lat_min, lat_max, ny)
    lon_vec = np.linspace(lon_min, lon_max, nx)

    amp = np.zeros((ny, nx))
    for i in range(ny):
        amp[i, :] = np.fromstring(lines[7 + i], sep=" ")

    phase_start = 7 + ny + 7        # skip 7 header lines of the phase block
    phase = np.zeros((ny, nx))
    for i in range(ny):
        phase[i, :] = np.fromstring(lines[phase_start + i], sep=" ")

    # Mask undefined values
    amp[amp >= undef - 0.01]     = np.nan
    phase[phase >= undef - 0.01] = np.nan

    _GRID_CACHE[key] = (amp, phase, lat_vec, lon_vec)
    return amp, phase, lat_vec, lon_vec


# ── Point interpolator ────────────────────────────────────────────────────

def _interpolate_at_point(amp: np.ndarray, phase: np.ndarray,
                           lat_vec: np.ndarray, lon_vec: np.ndarray,
                           lat: float, lon: float) -> tuple[float, float]:
    """Bilinear interpolation of amplitude [cm] and phase [°] at (lat, lon)."""
    pt = np.array([[lat, lon]])
    ia = RegularGridInterpolator(
        (lat_vec, lon_vec), amp,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    ip = RegularGridInterpolator(
        (lat_vec, lon_vec), phase,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    return float(ia(pt)[0]), float(ip(pt)[0])


# ── Main public API ───────────────────────────────────────────────────────

def get_tide_series(
    lat: float,
    lon: float,
    times: pd.Series | pd.DatetimeIndex,
    grid_dir: str | Path | None = None,
    constituents: dict[str, str] | None = None,
) -> np.ndarray:
    """Predict DTU10 tidal height [m] at (lat, lon) for each timestamp.

    Parameters
    ----------
    lat, lon      : station coordinates [degrees]
    times         : UTC timestamps (DatetimeSeries or DatetimeIndex)
    grid_dir      : override path to GOT47 directory (default: _GOT47_DIR)
    constituents  : override dict {name: filename} (default: CONSTITUENTS)

    Returns
    -------
    tide_m : np.ndarray of shape (len(times),) in metres.
             Returns zeros on import failure of pyTMD.
    """
    import importlib
    try:
        pytmd_const = importlib.import_module("pyTMD.constituents")
    except ImportError:
        warnings.warn("pyTMD not installed — returning zero tide series.")
        return np.zeros(len(times))

    if grid_dir is None:
        grid_dir = _GOT47_DIR
    if constituents is None:
        constituents = CONSTITUENTS

    grid_dir = Path(grid_dir)
    times_dt = pd.to_datetime(times)

    # Build arrays for available constituents at this point
    const_names: list[str] = []
    amps_m:      list[float] = []
    phases_rad:  list[float] = []

    for cname, fname in constituents.items():
        fpath = grid_dir / fname
        if not fpath.exists():
            warnings.warn(f"DTU10: missing file {fpath} — skipping {cname}")
            continue
        try:
            amp_g, ph_g, lat_v, lon_v = _read_got47_grid(fpath)
        except Exception as exc:
            warnings.warn(f"DTU10: error reading {fpath}: {exc} — skipping")
            continue
        a_cm, ph_deg = _interpolate_at_point(amp_g, ph_g, lat_v, lon_v, lat, lon)
        if np.isnan(a_cm) or np.isnan(ph_deg):
            # Station is on land or outside grid
            continue
        const_names.append(cname)
        amps_m.append(a_cm / 100.0)          # cm → m
        phases_rad.append(np.deg2rad(ph_deg))

    if not const_names:
        warnings.warn("DTU10: no valid constituents found at this location.")
        return np.zeros(len(times))

    amps_m_arr     = np.array(amps_m)
    phases_rad_arr = np.array(phases_rad)

    # pyTMD astronomical arguments
    t_days = (times_dt - _PYTMD_EPOCH).total_seconds().values / 86400.0
    MJD    = t_days + _PYTMD_MJD0

    try:
        pu, pf, G = pytmd_const.arguments(MJD, const_names, corrections="GOT")
    except Exception as exc:
        warnings.warn(f"DTU10: pyTMD.constituents.arguments failed: {exc}")
        return np.zeros(len(times))

    # η(t) = Σ f_i(t) · A_i · cos(G_i(t) + u_i(t) − φ_i)
    theta = np.radians(G) + pu                    # (n_times, n_const)
    tide  = np.zeros(len(t_days))
    for i in range(len(const_names)):
        tide += pf[:, i] * amps_m_arr[i] * np.cos(theta[:, i] - phases_rad_arr[i])

    return tide


def is_tide_available() -> bool:
    """Return True if the DTU10 grid volume is mounted and pyTMD is present."""
    try:
        import pyTMD.constituents  # noqa: F401
    except ImportError:
        return False
    # Check at least the M2 file
    return (_GOT47_DIR / "M2.d").exists()
