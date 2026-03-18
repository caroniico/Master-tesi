"""
Pre-compute DKSS model vs real Tide-Gauge observation errors.

Uses ALL available DKSS model configurations (dkss_uwcw > dkss2020 > dkss2019
> dkss2013) with priority to the most recent model version on overlaps.

Spatial collocation: median of DKSS grid points within RADIUS_KM of each TG.
Temporal alignment: merge_asof with ±TIME_TOL tolerance.

Outputs:
    data/hdm_tg_obs_all_stations.parquet
    data/stations.json

Usage:
    python prepare_data.py [--radius-km 3] [--time-tol-min 30] [--out-dir data]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════
VAR = "p82"

# DKSS sources, ordered by priority (highest = index 0)
DKSS_SOURCES = [
    {
        "label": "dkss_uwcw",
        "priority": 0,
        "path": Path("/home/nicaro/DATA/sealevel_from_dkss_uwcw_2024-2026/sealevel_from_dkss_uwcw"),
        "prefix": "dkss_grib_sealev.",
    },
    {
        "label": "dkss2020",
        "priority": 1,
        "path": Path("/home/nicaro/DATA/sealevel_from_dkss2020_2021-2024/sealevel_from_dkss2020"),
        "prefix": "dkss_grib_sealev.",
    },
    {
        "label": "dkss2019",
        "priority": 2,
        "path": Path("/home/nicaro/DATA/sealevel_from_dkss2019_2013-2021/sealevel_from_dkss2019"),
        "prefix": "dkss_grib_sealev.",
    },
    {
        "label": "dkss2013",
        "priority": 3,
        "path": Path("/home/nicaro/DATA/sealevel_from_dkss2013_2013-2019/sealevel_from_dkss2013"),
        "prefix": "dkss_grib_sealev.",
    },
]

# Tide-gauge data
TG_DIR = Path("/home/nicaro/DATA/HIDRA3_training_data/2013_2022_Tidal_Gauges")
TG_SUMMARY = TG_DIR / "stations_summary.csv"

# Display-name mapping (ASCII file names → proper Danish)
NAME_MAP = {
    "Rodby": "Rødby",
    "Kobenhavn": "København",
    "Hesnaes": "Hesnæs",
    "Sonderborg": "Sønderborg",
    "Dragor": "Dragør",
    "Koege": "Køge",
}


# ═══════════════════════════════════════════════════════════════════════
#  Haversine
# ═══════════════════════════════════════════════════════════════════════
def _haversine_km(lat0: float, lon0: float,
                  lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6371.0
    dlat = np.radians(lats - lat0)
    dlon = np.radians(lons - lon0)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat0)) * np.cos(np.radians(lats))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ═══════════════════════════════════════════════════════════════════════
#  Load stations
# ═══════════════════════════════════════════════════════════════════════
def load_stations() -> list[dict]:
    """Read station metadata from stations_summary.csv."""
    df = pd.read_csv(TG_SUMMARY)
    stations = []
    for _, row in df.iterrows():
        ascii_name = row["Station Name"]
        display_name = NAME_MAP.get(ascii_name, ascii_name)
        stations.append({
            "id": str(int(row["Station Number"])),
            "name": display_name,
            "ascii_name": ascii_name,
            "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]),
            "nan_pct": float(row["NaN Percentage"]),
        })
    return stations


# ═══════════════════════════════════════════════════════════════════════
#  Load TG observations
# ═══════════════════════════════════════════════════════════════════════
def load_tg_obs(stations: list[dict]) -> dict[str, pd.Series]:
    """Load TG CSVs into {station_id: pd.Series(value_m, index=UTC time)}."""
    tg = {}
    for st in stations:
        csv_path = TG_DIR / f"station_{st['ascii_name']}_{st['id']}.csv"
        if not csv_path.exists():
            print(f"  ⚠ TG file not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path, dtype={"timestamp": str, "value": float})
        df = df[df["value"].notna() & (df["value"] != 999)]
        df["time"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M", utc=True)
        s = pd.Series(df["value"].values / 100.0, index=df["time"])  # cm → m
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="first")]
        tg[st["id"]] = s
        print(f"  {st['name']:12s} ({st['id']}): "
              f"{s.index[0].date()} → {s.index[-1].date()}  N={len(s):>7}")
    return tg


# ═══════════════════════════════════════════════════════════════════════
#  Build spatial masks
# ═══════════════════════════════════════════════════════════════════════
def build_masks(stations: list[dict], lat2d: np.ndarray, lon2d: np.ndarray,
                radius_km: float) -> dict[str, np.ndarray]:
    """Build boolean mask (shape lat×lon) per station."""
    masks = {}
    for st in stations:
        dist = _haversine_km(st["lat"], st["lon"], lat2d, lon2d)
        m = dist <= radius_km
        n_water = int(m.sum())
        if n_water == 0:
            # Fallback: use nearest pixel
            m = dist == dist.min()
            print(f"  ⚠ {st['name']:12s}: 0 pixels within {radius_km} km → "
                  f"using nearest (dist={dist.min():.2f} km)")
        else:
            print(f"  {st['name']:12s}: {n_water} pixels within {radius_km} km")
        masks[st["id"]] = m
    return masks


# ═══════════════════════════════════════════════════════════════════════
#  List and sort all DKSS files across sources
# ═══════════════════════════════════════════════════════════════════════
def list_all_dkss_files() -> list[tuple[Path, int, str]]:
    """Return list of (filepath, priority, label) for all DKSS GRIB files."""
    all_files = []
    for src in DKSS_SOURCES:
        d = src["path"]
        if not d.exists():
            print(f"  ⚠ Skipping {src['label']}: {d} not found")
            continue
        files = sorted(
            f for f in d.iterdir()
            if f.name.startswith(src["prefix"]) and not f.name.endswith(".idx")
        )
        for f in files:
            all_files.append((f, src["priority"], src["label"]))
        print(f"  {src['label']:12s}: {len(files)} files  "
              f"({files[0].name} → {files[-1].name})" if files else
              f"  {src['label']:12s}: 0 files")
    return all_files


# ═══════════════════════════════════════════════════════════════════════
#  Extract DKSS at station locations
# ═══════════════════════════════════════════════════════════════════════
def extract_dkss(all_files: list[tuple[Path, int, str]],
                 masks: dict[str, np.ndarray],
                 station_ids: list[str],
                 checkpoint_dir: Path,
                 checkpoint_interval: int = 1000,
                 ) -> pd.DataFrame:
    """
    Extract median p82 for all station masks from all DKSS files.

    Returns DataFrame with columns:
        valid_time, station_id, dkss_p82_m, priority, source
    """
    checkpoint_file = checkpoint_dir / "_dkss_checkpoint.parquet"
    start_idx = 0
    all_records = []

    # Resume from checkpoint if exists
    if checkpoint_file.exists():
        prev = pd.read_parquet(checkpoint_file)
        all_records.append(prev)
        start_idx = int(prev.attrs.get("next_idx", 0)) if hasattr(prev, "attrs") else 0
        # Read next_idx from a sidecar
        idx_file = checkpoint_dir / "_dkss_checkpoint_idx.txt"
        if idx_file.exists():
            start_idx = int(idx_file.read_text().strip())
        print(f"  Resuming from checkpoint: file index {start_idx}, "
              f"{len(prev)} existing records")

    n_total = len(all_files)
    t0 = _time.time()
    batch_records = []

    for file_idx in range(start_idx, n_total):
        fpath, priority, label = all_files[file_idx]

        try:
            ds = xr.open_dataset(str(fpath), engine="cfgrib",
                                 backend_kwargs={"indexpath": ""})
            p82 = ds[VAR].values           # (6, 482, 396)
            vtimes = ds["valid_time"].values  # (6,)
            ds.close()
        except Exception:
            continue

        for step_idx in range(len(vtimes)):
            vt = vtimes[step_idx]
            field = p82[step_idx]  # (482, 396)
            for sid in station_ids:
                mask = masks[sid]
                pts = field[mask]
                pts = pts[np.isfinite(pts) & (pts < 1e29)]
                if len(pts) == 0:
                    continue
                batch_records.append((
                    pd.Timestamp(vt, tz="UTC"),
                    sid,
                    float(np.median(pts)),
                    priority,
                    label,
                ))

        # Progress + checkpoint
        done = file_idx - start_idx + 1
        if done % 200 == 0 or file_idx == n_total - 1:
            elapsed = _time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - start_idx - done) / rate if rate > 0 else 0
            print(f"  [{file_idx + 1}/{n_total}] {fpath.name}  "
                  f"({rate:.1f} files/s, ETA {eta/60:.0f} min)")

        if done % checkpoint_interval == 0 and batch_records:
            batch_df = pd.DataFrame(batch_records,
                                    columns=["valid_time", "station_id",
                                             "dkss_p82_m", "priority", "source"])
            all_records.append(batch_df)
            batch_records = []
            # Save checkpoint
            merged = pd.concat(all_records, ignore_index=True)
            merged.to_parquet(checkpoint_file, index=False)
            (checkpoint_dir / "_dkss_checkpoint_idx.txt").write_text(
                str(file_idx + 1))
            print(f"  💾 Checkpoint saved: {len(merged)} records, "
                  f"next_idx={file_idx + 1}")
            del merged
            gc.collect()

    # Final batch
    if batch_records:
        batch_df = pd.DataFrame(batch_records,
                                columns=["valid_time", "station_id",
                                         "dkss_p82_m", "priority", "source"])
        all_records.append(batch_df)

    result = pd.concat(all_records, ignore_index=True)

    # Clean up checkpoint files
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    idx_file = checkpoint_dir / "_dkss_checkpoint_idx.txt"
    if idx_file.exists():
        idx_file.unlink()

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Dedup by priority
# ═══════════════════════════════════════════════════════════════════════
def dedup_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    """For each (valid_time, station_id) keep the row with lowest priority value
    (= most recent model version)."""
    df = df.sort_values(["station_id", "valid_time", "priority"])
    df = df.drop_duplicates(subset=["station_id", "valid_time"], keep="first")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
#  Align TG ↔ DKSS
# ═══════════════════════════════════════════════════════════════════════
def align_tg_dkss(dkss_df: pd.DataFrame,
                  tg_obs: dict[str, pd.Series],
                  stations: list[dict],
                  time_tol: pd.Timedelta) -> pd.DataFrame:
    """Merge TG observations to DKSS timestamps per station."""
    station_map = {s["id"]: s for s in stations}
    parts = []

    for sid, grp in dkss_df.groupby("station_id"):
        if sid not in tg_obs:
            print(f"  ⚠ No TG data for station {sid}")
            continue
        tg_s = tg_obs[sid]
        st = station_map[sid]

        # Build TG DataFrame for merge_asof
        tg_df = pd.DataFrame({"valid_time": tg_s.index, "tg_obs_m": tg_s.values})
        tg_df = tg_df.sort_values("valid_time").reset_index(drop=True)
        tg_df["valid_time"] = tg_df["valid_time"].dt.as_unit("ns")

        # DKSS DataFrame
        model_df = grp[["valid_time", "dkss_p82_m", "source"]].copy()
        model_df = model_df.sort_values("valid_time").reset_index(drop=True)
        model_df["valid_time"] = model_df["valid_time"].dt.as_unit("ns")

        # merge_asof: for each DKSS timestamp, find nearest TG observation
        merged = pd.merge_asof(
            model_df, tg_df,
            on="valid_time",
            direction="nearest",
            tolerance=time_tol,
        )
        merged = merged.dropna(subset=["tg_obs_m", "dkss_p82_m"])
        merged["station_id"] = sid
        merged["station_name"] = st["name"]
        merged["lat"] = st["lat"]
        merged["lon"] = st["lon"]
        merged["error_m"] = merged["dkss_p82_m"] - merged["tg_obs_m"]

        # Diagnostics
        n = len(merged)
        if n == 0:
            print(f"  ⚠ {st['name']:12s}: 0 matched timestamps — skip")
            continue
        bias = merged["error_m"].mean()
        rmse = np.sqrt((merged["error_m"] ** 2).mean())
        mae = merged["error_m"].abs().mean()
        t0 = merged["valid_time"].min()
        t1 = merged["valid_time"].max()

        # Gap detection
        dt = merged["valid_time"].diff()
        max_gap = dt.max()
        gap_str = ""
        if max_gap > pd.Timedelta("24h"):
            n_big_gaps = (dt > pd.Timedelta("24h")).sum()
            gap_str = f"  ⚠ {n_big_gaps} gaps >24h (max {max_gap})"

        print(f"  {st['name']:12s}: N={n:>6}  "
              f"Bias={bias:+.4f} m  RMSE={rmse:.4f} m  MAE={mae:.4f} m  "
              f"[{t0.date()} → {t1.date()}]" + gap_str)

        parts.append(merged)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius-km", type=float, default=3.0,
                    help="Spatial collocation radius in km (default: 3)")
    ap.add_argument("--time-tol-min", type=int, default=30,
                    help="Temporal tolerance in minutes for TG↔DKSS matching (default: 30)")
    ap.add_argument("--out-dir", default="data",
                    help="Output directory (default: data)")
    ap.add_argument("--checkpoint-interval", type=int, default=1000,
                    help="Save checkpoint every N files (default: 1000)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_tol = pd.Timedelta(minutes=args.time_tol_min)

    print("=" * 70)
    print("  TG Observations vs DKSS Model — Data Preparation")
    print("=" * 70)
    print(f"  Radius:     {args.radius_km} km")
    print(f"  Time tol:   ±{args.time_tol_min} min")
    print(f"  Output:     {out_dir.resolve()}")
    print()

    # 1. Load stations
    print("── Loading station metadata ──")
    stations = load_stations()
    print(f"  {len(stations)} stations\n")

    # 2. Load TG observations
    print("── Loading TG observations ──")
    tg_obs = load_tg_obs(stations)
    print(f"  {len(tg_obs)} stations loaded\n")

    # 3. Load grid from a sample GRIB
    print("── Loading DKSS grid coordinates ──")
    sample_file = None
    for src in DKSS_SOURCES:
        candidates = sorted(src["path"].glob(src["prefix"] + "*"))
        candidates = [c for c in candidates if not c.name.endswith(".idx")]
        if candidates:
            sample_file = candidates[0]
            break
    if sample_file is None:
        sys.exit("No DKSS GRIB files found.")

    ds0 = xr.open_dataset(str(sample_file), engine="cfgrib",
                          backend_kwargs={"indexpath": ""})
    lat2d, lon2d = np.meshgrid(
        ds0["latitude"].values, ds0["longitude"].values, indexing="ij"
    )
    print(f"  Grid: {lat2d.shape[0]} × {lat2d.shape[1]}  "
          f"lat [{lat2d.min():.2f}, {lat2d.max():.2f}]  "
          f"lon [{lon2d.min():.2f}, {lon2d.max():.2f}]")
    ds0.close()
    print()

    # 4. Build spatial masks
    print(f"── Building spatial masks (radius = {args.radius_km} km) ──")
    masks = build_masks(stations, lat2d, lon2d, args.radius_km)
    print()

    # 5. List all DKSS files
    print("── Listing DKSS GRIB files ──")
    all_files = list_all_dkss_files()
    print(f"  Total: {len(all_files)} files\n")

    if not all_files:
        sys.exit("No DKSS files found.")

    # 6. Extract DKSS model values
    print("── Extracting DKSS model values ──")
    station_ids = [s["id"] for s in stations]
    dkss_raw = extract_dkss(
        all_files, masks, station_ids, out_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(f"  Raw records: {len(dkss_raw):,}\n")

    # 7. Dedup by priority
    print("── Deduplicating overlaps (priority: most recent model wins) ──")
    before = len(dkss_raw)
    dkss_dedup = dedup_by_priority(dkss_raw)
    after = len(dkss_dedup)
    print(f"  {before:,} → {after:,} ({before - after:,} duplicates removed)")

    # Source breakdown
    src_counts = dkss_dedup.groupby("source").size()
    for src, cnt in src_counts.items():
        print(f"    {src}: {cnt:,} timestamps")
    print()

    # 8. Align TG ↔ DKSS
    print(f"── Aligning TG ↔ DKSS (tolerance ±{args.time_tol_min} min) ──")
    result = align_tg_dkss(dkss_dedup, tg_obs, stations, time_tol)

    if result.empty:
        sys.exit("No matched data — check time overlap between TG and DKSS.")

    print(f"\n  Total matched rows: {len(result):,}")
    print(f"  Stations with data: {result['station_id'].nunique()}")
    print(f"  Time range: {result['valid_time'].min()} → {result['valid_time'].max()}")

    # 9. Write Parquet
    parquet_cols = ["station_id", "station_name", "lat", "lon",
                    "valid_time", "tg_obs_m", "dkss_p82_m", "error_m",
                    "source"]
    parquet_path = out_dir / "hdm_tg_obs_all_stations.parquet"
    result[parquet_cols].to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"\n  Parquet: {parquet_path.resolve()}  "
          f"({parquet_path.stat().st_size / 1e6:.1f} MB)")

    # 10. Write stations JSON
    stations_out = [
        {"id": s["id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"],
         "nan_pct": s["nan_pct"]}
        for s in stations
    ]
    stations_path = out_dir / "stations.json"
    with open(stations_path, "w") as f:
        json.dump(stations_out, f, indent=2, ensure_ascii=False)
    print(f"  Stations JSON: {stations_path.resolve()}")

    # 11. Summary table
    print("\n" + "=" * 78)
    print(f"{'Station':<14} {'ID':>6}  {'N':>7}  {'Bias[m]':>9}  "
          f"{'RMSE[m]':>9}  {'MAE[m]':>9}  {'σ[m]':>9}")
    print("=" * 78)
    for sid in sorted(result["station_id"].unique()):
        sub = result[result["station_id"] == sid]
        name = sub["station_name"].iloc[0]
        err = sub["error_m"]
        bias = err.mean()
        rmse = np.sqrt((err ** 2).mean())
        mae = err.abs().mean()
        std_e = err.std()
        print(f"{name:<14} {sid:>6}  {len(sub):>7}  {bias:>+9.4f}  "
              f"{rmse:>9.4f}  {mae:>9.4f}  {std_e:>9.4f}")
    print("=" * 78)
    print("\nDone.")


if __name__ == "__main__":
    main()
