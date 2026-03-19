#!/usr/bin/env python3
"""
build_station_datasets.py
=========================
Builds one Parquet file per tide-gauge station, combining:
  1. TG observations  (local CSV, 10 min -> hourly on-the-hour)
  2. FORCOAST HDM p82  (local GRIB, nearest-neighbor non-NaN)
  3. Atmospheric forcings SLP/t2m/u10/v10  (FTP monthly .tar.zst -> daily .nc)

Output: data/per_station/station_{id}_{name}.parquet
See DATASET_ARCHITECTURE.md for full specification.

Usage:
  python build_station_datasets.py                     # all 15 stations
  python build_station_datasets.py --stations 30336 31616 28366  # subset by ID
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time as _time
from datetime import datetime
from ftplib import FTP
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zstandard as zstd

# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_DIR = Path(__file__).resolve().parent
STATIONS_JSON = PROJECT_DIR / "data" / "stations.json"
OUTPUT_DIR = PROJECT_DIR / "data" / "per_station"

# Local data paths
TG_DIR = Path("/home/nicaro/DATA/HIDRA3_training_data/2013_2022_Tidal_Gauges")
FORCOAST_DIR = Path(
    "/home/nicaro/DATA/sealevel_from_forcoast_2009-2020/sealevel_from_forcoast"
)

# FTP for atmospheric forcings
FTP_HOST = os.environ.get("FTP_HOST", "ocean.dmi.dk")
FTP_USER = os.environ.get("FTP_USER", "oceanftp")
FTP_PASS = os.environ.get("FTP_PASS", "NYEflinte.stene")
FTP_FOLDER = os.environ.get("FTP_FOLDER", "MBL/HIDRA3_training_data")

# Overlap period
START = "2013-01-01"
END = "2020-03-31"

# ASCII name -> CSV filename mapping (CSV uses ASCII names)
NAME_TO_ASCII = {
    "Rødby": "Rodby",
    "København": "Kobenhavn",
    "Hesnæs": "Hesnaes",
    "Sønderborg": "Sonderborg",
    "Dragør": "Dragor",
    "Køge": "Koege",
}

# Atmospheric NetCDF variable mapping
ATMO_VAR_MAP = {
    "var1": "SLP",
    "var11": "t2m",
    "var33": "u10",
    "var34": "v10",
}


# ======================================================================
# UTILITIES
# ======================================================================


def haversine_vec(lat0, lon0, lats, lons):
    """
    Vectorised haversine: distance in km from (lat0, lon0)
    to every (lats[i], lons[j]).  Returns shape (len(lats), len(lons)).
    """
    R = 6371.0
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r[:, None] - lat0_r
    dlon = lons_r[None, :] - lon0_r
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat0_r) * np.cos(lats_r[:, None]) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_scalar(lat1, lon1, lat2, lon2):
    """Distance in km between two points (thin wrapper around vec version)."""
    d = haversine_vec(lat1, lon1, np.array([lat2]), np.array([lon2]))
    return float(d[0, 0])


def load_stations(subset_ids=None):
    """Load station metadata from stations.json, optionally filtering."""
    with open(STATIONS_JSON) as f:
        stations = json.load(f)
    # Normalise IDs to int for consistent comparison
    for s in stations:
        s["id"] = int(s["id"])
    if subset_ids:
        id_set = set(int(i) for i in subset_ids)
        stations = [s for s in stations if s["id"] in id_set]
        if not stations:
            sys.exit(f"ERROR: no stations matched IDs {subset_ids}")
    return stations


# ======================================================================
# 1. TIDE GAUGES
# ======================================================================


def load_tg(station):
    """
    Load TG CSV for a station, filter on-the-hour rows, convert to metres.
    Returns DataFrame with columns: [time, tg_obs_m].
    """
    name = station["name"]
    sid = station["id"]
    ascii_name = NAME_TO_ASCII.get(name, name)
    csv_path = TG_DIR / f"station_{ascii_name}_{sid}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"TG CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"timestamp": str, "value": float})
    df["time"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M")

    # On-the-hour only
    df = df[df["time"].dt.minute == 0].copy()

    # 999 -> NaN, cm -> m
    df["tg_obs_m"] = df["value"].replace(999, np.nan) / 100.0

    # Trim to overlap period
    mask = (df["time"] >= START) & (df["time"] <= END + " 23:00:00")
    df = df.loc[mask, ["time", "tg_obs_m"]].reset_index(drop=True)
    return df


# ======================================================================
# 2. FORCOAST HDM
# ======================================================================


def find_nearest_water_cells(lats, lons, water_mask, stations):
    """
    Vectorised nearest non-NaN grid cell for all stations.
    Returns {station_id: (i, j, cell_lat, cell_lon, dist_km)}.
    """
    result = {}
    for stn in stations:
        dist_2d = haversine_vec(stn["lat"], stn["lon"], lats, lons)
        dist_2d[~water_mask] = np.inf
        flat_idx = int(np.argmin(dist_2d))
        i, j = np.unravel_index(flat_idx, dist_2d.shape)
        d = float(dist_2d[i, j])
        if d == np.inf:
            raise ValueError(
                f"No water cell found near {stn['name']} "
                f"({stn['lat']}, {stn['lon']})"
            )
        result[stn["id"]] = (int(i), int(j), float(lats[i]), float(lons[j]), d)
    return result


def list_forcoast_grib_files():
    """List and date-filter FORCOAST GRIB files to the overlap period."""
    grib_files = sorted(
        f
        for f in FORCOAST_DIR.iterdir()
        if f.name.startswith("forcoast_grib_sealev.")
        and not f.name.endswith(".idx")
    )
    start_tag = datetime.strptime(START, "%Y-%m-%d").strftime("%Y%m%d")
    end_tag = datetime.strptime(END, "%Y-%m-%d").strftime("%Y%m%d")
    return [
        f for f in grib_files
        if start_tag <= f.name.split(".")[-1][:8] <= end_tag
    ]


def build_forcoast_all_stations(stations):
    """
    One pass over all GRIBs, extracting p82 at the nearest water cell
    for every station.  Returns dicts keyed by station_id.
    """
    grib_files = list_forcoast_grib_files()
    n_files = len(grib_files)
    print(f"  FORCOAST: {n_files} daily GRIB files in overlap period")

    # Nearest water cell from first GRIB
    ds0 = xr.open_dataset(
        str(grib_files[0]), engine="cfgrib",
        backend_kwargs={"indexpath": ""}
    )
    lats = ds0.latitude.values
    lons = ds0.longitude.values
    water_mask = ~np.isnan(ds0["p82"].isel(step=0).values)
    ds0.close()

    cell_map = find_nearest_water_cells(lats, lons, water_mask, stations)
    meta = {}
    for stn in stations:
        i, j, clat, clon, d = cell_map[stn["id"]]
        meta[stn["id"]] = (clat, clon, d)
        print(
            f"    {stn['name']:14s}: water cell ({clat:.4f}, {clon:.4f}), "
            f"dist = {d:.3f} km"
        )

    # Extract time series — accumulate arrays, not single tuples
    accum_t = {s["id"]: [] for s in stations}   # list of numpy datetime64 arrays
    accum_v = {s["id"]: [] for s in stations}   # list of float64 arrays
    t0 = _time.time()

    for fi, gf in enumerate(grib_files):
        if fi % 200 == 0:
            elapsed = _time.time() - t0
            pct = fi / max(n_files, 1) * 100
            print(
                f"    [{fi:>5}/{n_files}] {pct:5.1f}%  "
                f"elapsed {elapsed:.0f}s  {gf.name}",
                flush=True,
            )
        try:
            ds = xr.open_dataset(
                str(gf), engine="cfgrib",
                backend_kwargs={"indexpath": ""}
            )
        except Exception as e:
            print(f"    WARN: skip {gf.name}: {e}")
            continue

        vt = ds["valid_time"].values                       # (24,) datetime64
        for stn in stations:
            sid = stn["id"]
            ci, cj = cell_map[sid][:2]
            vals = ds["p82"][:, ci, cj].values.astype(np.float64)  # (24,)
            accum_t[sid].append(vt)
            accum_v[sid].append(vals)
        ds.close()

    elapsed = _time.time() - t0
    print(f"    FORCOAST done: {n_files} files in {elapsed:.0f}s")

    fc_data = {}
    for stn in stations:
        sid = stn["id"]
        if not accum_t[sid]:
            fc_data[sid] = pd.DataFrame(columns=["time", "forcoast_p82_m"])
            continue
        all_t = np.concatenate(accum_t[sid])
        all_v = np.concatenate(accum_v[sid])
        all_v[all_v > 1e30] = np.nan          # GRIB fill → NaN
        fc_data[sid] = pd.DataFrame(
            {"time": pd.DatetimeIndex(all_t), "forcoast_p82_m": all_v}
        )

    return fc_data, meta


# ======================================================================
# 3. ATMOSPHERIC FORCINGS (from FTP)
# ======================================================================


def ftp_connect():
    ftp = FTP(FTP_HOST, timeout=120)
    ftp.login(FTP_USER, FTP_PASS)
    ftp.cwd(FTP_FOLDER)
    return ftp


def download_atmo_month(ftp, yyyymm, tmpdir):
    """
    Download one monthly .tar.zst from FTP and return list of daily
    xr.Datasets (renamed vars, dropped singleton vertical dims, loaded).
    """
    archive_name = f"{yyyymm}.tar.zst"
    tmp_path = os.path.join(tmpdir, archive_name)

    with open(tmp_path, "wb") as fp:
        ftp.retrbinary(f"RETR {archive_name}", fp.write, blocksize=1024 * 1024)

    datasets = []
    with open(tmp_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".nc"):
                        continue
                    extracted = tar.extractfile(member)
                    nc_path = os.path.join(tmpdir, os.path.basename(member.name))
                    with open(nc_path, "wb") as nf:
                        shutil.copyfileobj(extracted, nf)

                    ds = xr.open_dataset(nc_path)
                    ds = ds.rename(
                        {k: v for k, v in ATMO_VAR_MAP.items() if k in ds.data_vars}
                    )
                    # Squeeze ALL singleton dims except the ones we need.
                    # Handles alt, height, alt_2, and any future surprises.
                    for dim_name in list(ds.dims):
                        if dim_name not in ("time", "lat", "lon") and ds.sizes[dim_name] == 1:
                            ds = ds.isel({dim_name: 0})
                            if dim_name in ds.coords:
                                ds = ds.drop_vars(dim_name)
                    ds = ds.load()
                    datasets.append(ds)
                    os.remove(nc_path)

    os.remove(tmp_path)
    return datasets


def build_atmo_series(stations):
    """
    Download atmospheric forcings for the overlap period (201301-202003)
    and extract nearest-neighbor time series for ALL stations in one pass.
    """
    start_dt = datetime.strptime(START, "%Y-%m-%d")
    end_dt = datetime.strptime(END, "%Y-%m-%d")
    months = []
    d = start_dt.replace(day=1)
    while d <= end_dt:
        months.append(d.strftime("%Y%m"))
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)

    print(f"  ATMO: {len(months)} monthly archives to download from FTP")

    atmo_accum = {s["id"]: [] for s in stations}
    atmo_meta = {}
    nearest_idx = {}

    tmpdir = tempfile.mkdtemp(prefix="atmo_")
    ftp = ftp_connect()
    t0 = _time.time()

    for mi, yyyymm in enumerate(months):
        elapsed = _time.time() - t0
        print(
            f"    [{mi+1:>3}/{len(months)}] {yyyymm} "
            f"(elapsed {elapsed:.0f}s) ...",
            end=" ", flush=True,
        )
        for attempt in range(3):
            try:
                day_datasets = download_atmo_month(ftp, yyyymm, tmpdir)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"SKIP after 3 attempts: {e}")
                    day_datasets = None
                    break
                try:
                    ftp.quit()
                except Exception:
                    pass
                _time.sleep(5 * (attempt + 1))
                ftp = ftp_connect()
        if day_datasets is None:
            continue

        if not day_datasets:
            print("empty")
            continue

        monthly = xr.concat(day_datasets, dim="time").sortby("time")
        print(f"OK ({len(monthly.time)} h)")

        # Determine nearest atmo cell on first successful month
        # Store integer indices to avoid float comparison issues with .sel()
        if not nearest_idx:
            atmo_lats = monthly.lat.values
            atmo_lons = monthly.lon.values
            for s in stations:
                i_lat = int(np.abs(atmo_lats - s["lat"]).argmin())
                i_lon = int(np.abs(atmo_lons - s["lon"]).argmin())
                nlat = float(atmo_lats[i_lat])
                nlon = float(atmo_lons[i_lon])
                dist = haversine_scalar(s["lat"], s["lon"], nlat, nlon)
                nearest_idx[s["id"]] = (i_lat, i_lon)
                atmo_meta[s["id"]] = (nlat, nlon, dist)
                print(
                    f"      Atmo nearest for {s['name']}: "
                    f"({nlat:.2f}, {nlon:.2f}), dist={dist:.2f} km"
                )

        # Accumulate monthly DataFrames per station (no row-by-row loop)
        for s in stations:
            i_lat, i_lon = nearest_idx[s["id"]]
            pt_ts = monthly.isel(lat=i_lat, lon=i_lon)
            df_month = pd.DataFrame({
                "time": pd.DatetimeIndex(pt_ts["time"].values),
                "SLP":  pt_ts["SLP"].values.astype(np.float64),
                "t2m":  pt_ts["t2m"].values.astype(np.float64),
                "u10":  pt_ts["u10"].values.astype(np.float64),
                "v10":  pt_ts["v10"].values.astype(np.float64),
            })
            atmo_accum[s["id"]].append(df_month)

        monthly.close()
        for dd in day_datasets:
            dd.close()

    try:
        ftp.quit()
    except Exception:
        pass
    shutil.rmtree(tmpdir, ignore_errors=True)

    atmo_data = {}
    for sid, dfs in atmo_accum.items():
        if dfs:
            atmo_data[sid] = pd.concat(dfs, ignore_index=True)
        else:
            atmo_data[sid] = pd.DataFrame(
                columns=["time", "SLP", "t2m", "u10", "v10"]
            )
    return atmo_data, atmo_meta


# ======================================================================
# MAIN PIPELINE
# ======================================================================


def sanitise_name(name):
    return (
        name.replace("ø", "o").replace("æ", "ae").replace("å", "aa")
        .replace("Ø", "O").replace("Æ", "Ae").replace("Å", "Aa")
    )


def main():
    parser = argparse.ArgumentParser(description="Build per-station datasets")
    parser.add_argument(
        "--stations", nargs="+", type=int, default=None,
        help="Station IDs to process (default: all 15)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BUILD STATION DATASETS")
    print("=" * 70)

    stations = load_stations(args.stations)
    print(f"Stations to process: {len(stations)}")
    for s in stations:
        print(f"  {s['id']}  {s['name']}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Step 1: FORCOAST (one pass over all GRIBs) --
    print("-" * 70)
    print("STEP 1: FORCOAST HDM (local GRIBs)")
    print("-" * 70)
    fc_data, fc_meta = build_forcoast_all_stations(stations)
    print()

    # -- Step 2: Atmospheric forcings (one FTP pass) --
    print("-" * 70)
    print("STEP 2: Atmospheric forcings (FTP download)")
    print("-" * 70)
    atmo_data, atmo_meta = build_atmo_series(stations)
    print()

    # -- Step 3: TG load + merge + save --
    print("-" * 70)
    print("STEP 3: Per-station TG + merge + save")
    print("-" * 70)

    for si, stn in enumerate(stations):
        sid = stn["id"]
        sname = stn["name"]
        print(f"\n  [{si+1}/{len(stations)}] {sname} ({sid})")

        try:
            tg_df = load_tg(stn)
            print(f"    TG: {len(tg_df)} hourly observations")
        except FileNotFoundError as e:
            print(f"    SKIP (TG not found): {e}")
            continue

        fc_df = fc_data.get(sid, pd.DataFrame())
        at_df = atmo_data.get(sid, pd.DataFrame())
        fc_lat, fc_lon, fc_dist = fc_meta.get(sid, (np.nan, np.nan, np.nan))
        at_lat, at_lon, at_dist = atmo_meta.get(sid, (np.nan, np.nan, np.nan))

        print(f"    FORCOAST: {len(fc_df)} values")
        print(f"    ATMO:     {len(at_df)} values")

        n_tg = len(tg_df)
        merged = tg_df.merge(fc_df, on="time", how="inner")
        n_after_fc = len(merged)
        merged = merged.merge(at_df, on="time", how="inner")
        n_final = len(merged)

        # ---- Merge diagnostics ----
        # Theoretical hours in overlap period
        theoretical_h = int(
            (pd.Timestamp(END + " 23:00:00") - pd.Timestamp(START))
            / pd.Timedelta(hours=1)
        ) + 1
        coverage     = n_final / n_tg if n_tg else 0.0
        fc_coverage  = n_after_fc / n_tg if n_tg else 0.0
        atmo_coverage = n_final / n_after_fc if n_after_fc else 0.0

        print(
            f"    Merge: TG={n_tg} -> +FC={n_after_fc} -> +ATMO={n_final}  "
            f"(dropped {n_tg - n_final} TG rows)"
        )
        print(
            f"    Theoretical hours: {theoretical_h}  |  "
            f"TG fill: {n_tg}/{theoretical_h} ({n_tg/theoretical_h:.3f})"
        )
        print(
            f"    coverage = {coverage:.3f}  "
            f"(FC step: {fc_coverage:.3f}, ATMO step: {atmo_coverage:.3f})"
        )

        # Detect months with zero FORCOAST or ATMO coverage
        if n_final > 0:
            merged_sorted = merged.sort_values("time")
            all_months = pd.date_range(
                START, END, freq="MS"
            ).strftime("%Y%m").tolist()
            fc_months = set(
                fc_df["time"].dt.strftime("%Y%m")
            ) if len(fc_df) else set()
            atmo_months = set(
                at_df["time"].dt.strftime("%Y%m")
            ) if len(at_df) else set()
            fc_gaps = sorted(m for m in all_months if m not in fc_months)
            atmo_gaps = sorted(m for m in all_months if m not in atmo_months)
            if fc_gaps:
                print(f"    ⚠ FC missing months   ({len(fc_gaps)}): {fc_gaps[:10]}{'...' if len(fc_gaps) > 10 else ''}")
            if atmo_gaps:
                print(f"    ⚠ ATMO missing months ({len(atmo_gaps)}): {atmo_gaps[:10]}{'...' if len(atmo_gaps) > 10 else ''}")

        if coverage < 0.90:
            print(f"    ⚠ LOW COVERAGE: only {coverage:.1%} of TG rows survived merge!")

        merged = merged.sort_values("time").reset_index(drop=True)

        merged["station_id"] = sid
        merged["station_name"] = sname
        merged["lat"] = stn["lat"]
        merged["lon"] = stn["lon"]
        merged["forcoast_lat"] = fc_lat
        merged["forcoast_lon"] = fc_lon
        merged["forcoast_dist_km"] = round(fc_dist, 4)
        merged["atmo_lat"] = at_lat
        merged["atmo_lon"] = at_lon
        merged["atmo_dist_km"] = round(at_dist, 4)

        safe = sanitise_name(sname)
        out_path = OUTPUT_DIR / f"station_{sid}_{safe}.parquet"
        merged.to_parquet(out_path, index=False)

        t_min = merged["time"].min()
        t_max = merged["time"].max()
        n_tg_nan = int(merged["tg_obs_m"].isna().sum())
        n_fc_nan = int(merged["forcoast_p82_m"].isna().sum())
        print(
            f"    -> {out_path.name}: {len(merged)} rows "
            f"[{t_min.date()} -> {t_max.date()}], "
            f"TG NaN={n_tg_nan}, FC NaN={n_fc_nan}"
        )

    print("\n" + "=" * 70)
    print("DONE -- output in", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
