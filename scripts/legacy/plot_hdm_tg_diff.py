"""
Plot HDM (DKSS) − TG (FORCOAST) difference in time and frequency domain.

Both datasets sit on the same 482×396 lat-lon grid.
We compute the pointwise difference over a set of representative
coastal grid-points, then show:
  1. Time-domain plot  (spatial-mean ± 1σ of r(t) = DKSS − FORCOAST)
  2. Frequency-domain  (Welch PSD of the spatial-mean difference)

Usage:
    python plot_hdm_tg_diff.py [--start 2013-11-01] [--end 2015-12-31]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import cfgrib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

# ── paths ────────────────────────────────────────────────────────────────
DKSS_DIR = Path("/home/nicaro/DATA/sealevel_from_dkss2013_2013-2019/sealevel_from_dkss2013")
FORC_DIR = Path("/home/nicaro/DATA/sealevel_from_forcoast_2009-2020/sealevel_from_forcoast")

VAR = "p82"                 # sea-level variable in both GRIB sources
DKSS_PREFIX = "dkss_grib_sealev."
FORC_PREFIX = "forcoast_grib_sealev."


# ── helpers ──────────────────────────────────────────────────────────────
def _parse_ts(name: str, prefix: str) -> dt.datetime:
    """Extract base datetime from filename like 'prefix.YYYYMMDDHH'."""
    stamp = name.replace(prefix, "")
    return dt.datetime.strptime(stamp, "%Y%m%d%H")


def list_files(directory: Path, prefix: str,
               start: dt.datetime, end: dt.datetime) -> list[Path]:
    """Return sorted list of GRIB files whose base time is in [start, end]."""
    paths = []
    for f in sorted(directory.iterdir()):
        if f.name.startswith(prefix) and not f.name.endswith(".idx"):
            try:
                ts = _parse_ts(f.name, prefix)
            except ValueError:
                continue
            if start <= ts <= end:
                paths.append(f)
    return paths


def load_grib_timeseries(path: Path, lat_idx: np.ndarray, lon_idx: np.ndarray):
    """Load p82 from one GRIB file → (valid_times, values[n_times, n_pts])."""
    try:
        ds = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    except Exception:
        return None, None
    # pick the dataset that contains our variable
    for d in ds:
        if VAR in d.data_vars:
            arr = d[VAR]  # shape: (step, lat, lon)
            break
    else:
        return None, None

    valid_times = pd.to_datetime(arr.coords["valid_time"].values)
    vals = arr.values  # (n_step, lat, lon)

    # extract at selected grid points → (n_step, n_pts)
    out = vals[:, lat_idx, lon_idx]
    return valid_times, out


def _spread_select_indices(candidate_flat_idx: np.ndarray, n_select: int, width: int) -> np.ndarray:
    """Select indices from candidates while enforcing minimal spacing in grid space."""
    selected: list[int] = []
    min_dist2 = 20 * 20

    for flat in candidate_flat_idx:
        row = int(flat // width)
        col = int(flat % width)
        if not selected:
            selected.append(int(flat))
            if len(selected) >= n_select:
                break
            continue

        ok = True
        for old_flat in selected:
            old_row = old_flat // width
            old_col = old_flat % width
            dist2 = (row - old_row) ** 2 + (col - old_col) ** 2
            if dist2 < min_dist2:
                ok = False
                break
        if ok:
            selected.append(int(flat))
            if len(selected) >= n_select:
                break

    if len(selected) < n_select:
        for flat in candidate_flat_idx:
            as_int = int(flat)
            if as_int not in selected:
                selected.append(as_int)
                if len(selected) >= n_select:
                    break

    return np.array(selected[:n_select], dtype=int)


def pick_grid_coords(sample_file: Path, n_coastal: int = 20):
    """Pick n_coastal grid points along the coastline (shallow-water mask).

    Strategy: open one file, compute spatial variance of sea-level across
    steps — high-variance pixels are coastal / shallow.
    Return (lat_idx, lon_idx) arrays of shape (n_coastal,).
    """
    ds = cfgrib.open_datasets(str(sample_file), backend_kwargs={"indexpath": ""})
    for d in ds:
        if VAR in d.data_vars:
            arr = d[VAR]
            break
    else:
        raise RuntimeError(f"Cannot find {VAR} in {sample_file}")

    data = arr.values  # (step, lat, lon)
    if data.shape[0] < 2:
        var_map = np.nanmax(np.abs(data), axis=0)
    else:
        finite_count = np.sum(np.isfinite(data), axis=0)
        with np.errstate(invalid="ignore"):
            var_map = np.nanvar(data, axis=0)
        var_map[finite_count < 2] = np.nan
    # mask NaN pixels
    valid = ~np.isnan(var_map)
    var_map[~valid] = 0.0

    flat_sorted = np.argsort(var_map.ravel())[::-1]
    top_pool = flat_sorted[: max(n_coastal * 50, n_coastal)]
    flat_idx = _spread_select_indices(top_pool, n_coastal, var_map.shape[1])
    lat_idx = flat_idx // var_map.shape[1]
    lon_idx = flat_idx % var_map.shape[1]

    lats = arr.coords["latitude"].values[lat_idx]
    lons = arr.coords["longitude"].values[lon_idx]
    print(f"Selected {n_coastal} coastal grid points:")
    for i in range(n_coastal):
        print(f"  pt{i}: lat={lats[i]:.3f}  lon={lons[i]:.3f}")
    return lat_idx, lon_idx, lats, lons


# ── main pipeline ────────────────────────────────────────────────────────
def build_timeseries(files: list[Path], prefix: str,
                     lat_idx: np.ndarray, lon_idx: np.ndarray,
                     label: str):
    """Read multiple GRIB files → DataFrame[valid_time, pt0..ptN]."""
    records = []
    n_pts = len(lat_idx)
    for i, f in enumerate(files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{label}] {i+1}/{len(files)}  {f.name}")
        try:
            vt, vals = load_grib_timeseries(f, lat_idx, lon_idx)
        except Exception as e:
            print(f"  ⚠ skip {f.name}: {e}")
            continue
        if vt is None:
            continue
        for t_idx in range(len(vt)):
            rec = {"valid_time": vt[t_idx]}
            for p in range(n_pts):
                rec[f"pt{p}"] = vals[t_idx, p]
            records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("valid_time").drop_duplicates(subset="valid_time", keep="last")
    df = df.set_index("valid_time")
    return df


def sample_at_delta_t(df: pd.DataFrame, delta_hours: int = 6) -> pd.DataFrame:
    """Keep only rows whose valid_time aligns with the model DeltaT grid."""
    mask = df.index.hour % delta_hours == 0
    return df.loc[mask].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2013-11-01",
                    help="Start date (YYYY-MM-DD). DKSS begins 2013-10-29.")
    ap.add_argument("--end", default="2015-12-31",
                    help="End date (YYYY-MM-DD).")
    ap.add_argument("--n-points", type=int, default=15,
                    help="Number of coastal grid-points to sample.")
    ap.add_argument("--delta-t", type=int, default=6,
                    help="Sampling interval in hours (model DeltaT).")
    ap.add_argument("--out-prefix", default="hdm_tg_diff",
                    help="Output prefix for png/csv/npy files.")
    args = ap.parse_args()

    t_start = dt.datetime.strptime(args.start, "%Y-%m-%d")
    t_end   = dt.datetime.strptime(args.end,   "%Y-%m-%d")
    n_pts   = args.n_points
    delta_t = args.delta_t

    print(f"Period:  {t_start:%Y-%m-%d} → {t_end:%Y-%m-%d}")
    print(f"Points:  {n_pts}  |  ΔT = {delta_t}h\n")

    # 1. list files
    dkss_files = list_files(DKSS_DIR, DKSS_PREFIX, t_start, t_end)
    forc_files = list_files(FORC_DIR, FORC_PREFIX, t_start, t_end)
    print(f"DKSS files: {len(dkss_files)}   FORCOAST files: {len(forc_files)}\n")

    if not dkss_files:
        sys.exit("No DKSS files found for the requested period.")
    if not forc_files:
        sys.exit("No FORCOAST files found for the requested period.")

    # 2. pick coastal grid-points from first DKSS file
    lat_idx, lon_idx, lats, lons = pick_grid_coords(dkss_files[0], n_pts)

    # 3. extract time series
    print("\n── Extracting DKSS (HDM) ──")
    df_dkss = build_timeseries(dkss_files, DKSS_PREFIX, lat_idx, lon_idx, "DKSS")
    print(f"  → {len(df_dkss)} timestamps\n")

    print("── Extracting FORCOAST (TG) ──")
    df_forc = build_timeseries(forc_files, FORC_PREFIX, lat_idx, lon_idx, "FORC")
    print(f"  → {len(df_forc)} timestamps\n")

    # 4. align on valid_time & sample at ΔT
    common = df_dkss.index.intersection(df_forc.index)
    print(f"Common valid_times: {len(common)}")
    df_dkss = df_dkss.loc[common]
    df_forc = df_forc.loc[common]

    # r(t) = HDM − TG
    df_diff = df_dkss - df_forc

    # sample at ΔT
    df_diff = sample_at_delta_t(df_diff, delta_t)
    print(f"After ΔT={delta_t}h sampling: {len(df_diff)} timestamps\n")

    if df_diff.empty:
        sys.exit("No data after alignment and sampling.")

    # spatial mean and std across points
    mean_r = df_diff.mean(axis=1)
    std_r  = df_diff.std(axis=1)

    # ── 5. Time-domain plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.plot(mean_r.index, mean_r.values, linewidth=0.5, color="steelblue",
            label="spatial mean  r(t)")
    ax.fill_between(mean_r.index,
                    (mean_r - std_r).values,
                    (mean_r + std_r).values,
                    alpha=0.2, color="steelblue", label="±1σ across points")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Sea-level difference  [m]")
    ax.set_title(f"r(t) = DKSS − FORCOAST   ({args.start}  →  {args.end},  "
                 f"ΔT = {delta_t}h,  {n_pts} coastal pts)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── 6. Frequency-domain plot (Welch PSD) ─────────────────────────────
    # compute sampling frequency in cycles per hour
    fs = 1.0 / delta_t  # samples per hour
    signal = mean_r.dropna().values
    nperseg = min(256, len(signal) // 2)
    if nperseg < 16:
        print("⚠ Not enough data for Welch PSD")
    else:
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

        ax2 = axes[1]
        ax2.semilogy(freqs, psd, color="coral", linewidth=0.8)
        ax2.set_xlabel("Frequency  [cycles / hour]")
        ax2.set_ylabel("PSD  [m² / (cycles/h)]")
        ax2.set_title("Welch PSD of spatial-mean r(t)")
        ax2.grid(True, alpha=0.3, which="both")

        # add a secondary x-axis with period in hours
        ax2_top = ax2.twiny()
        tick_periods = [6, 12, 24, 48, 72, 168, 336, 720]
        tick_freqs = [1.0 / p for p in tick_periods]
        ax2_top.set_xlim(ax2.get_xlim())
        ax2_top.set_xticks(tick_freqs)
        ax2_top.set_xticklabels([f"{p}h" for p in tick_periods], fontsize=8)
        ax2_top.set_xlabel("Period")

    plt.tight_layout()
    out_path = Path(f"{args.out_prefix}_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path.resolve()}")
    plt.close(fig)

    mean_path = Path(f"{args.out_prefix}_mean_timeseries.csv")
    psd_path = Path(f"{args.out_prefix}_welch_psd.csv")
    points_path = Path(f"{args.out_prefix}_points.csv")

    pd.DataFrame({
        "valid_time": mean_r.index,
        "mean_diff_m": mean_r.values,
        "std_diff_m": std_r.values,
    }).to_csv(mean_path, index=False)

    if nperseg >= 16:
        pd.DataFrame({"freq_cph": freqs, "psd": psd}).to_csv(psd_path, index=False)
    pd.DataFrame({"point": np.arange(len(lats)), "latitude": lats, "longitude": lons}).to_csv(points_path, index=False)

    print(f"Time-series CSV: {mean_path.resolve()}")
    if nperseg >= 16:
        print(f"Welch PSD CSV:  {psd_path.resolve()}")
    print(f"Points CSV:     {points_path.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
