# Dataset Architecture — Station-level Training Data

## Objective

Build a unified hourly dataset **per tide gauge station** combining:
1. Sea level observations (TG)
2. Sea level model output (FORCOAST HDM)
3. Atmospheric forcings (MSL, t2m, u10, v10)

All variables are extracted as **point time series** at each station location.

---

## Data Sources

| # | Dataset | Source | Format | Temporal Res. | Spatial Res. | Grid | Period | Variable(s) |
|---|---------|--------|--------|--------------|-------------|------|--------|-------------|
| 1 | **Tide Gauges (TG)** | FTP `2013_2022_Tidal_Gauges.tar.zst` | CSV | **10 min** | Point (15 stations) | — | 2013 – 2022 | `value` (sea level, cm) |
| 2 | **Atmospheric Forcings** | FTP `YYYYMM.tar.zst` → daily `.nc` | NetCDF | **1 hour** | 141×401 (~5.5 km) | lat 53.03–60.03, lon 5.03–25.03 | 2013/01 – 2024/05 | `var1`=MSL, `var11`=t2m, `var33`=u10, `var34`=v10 |
| 3 | **FORCOAST (HDM model)** | Local GRIB | GRIB | **1 hour** (24 steps/file, 1 file/day) | 482×396 (~0.9 km) | lat 53.59–57.60, lon 9.34–14.83 | 2008/07 – 2020/03 | `p82` (sea level, m) |

> **Overlap period: 2013/01 → 2020/03** (~7 years, ~61,000 hours)

---

## FTP Access

```
Host:   ocean.dmi.dk
User:   oceanftp
Pass:   NYEflinte.stene
Folder: MBL/HIDRA3_training_data
```

### FTP Contents
- `2013_2022_Tidal_Gauges.tar.zst` → 15 station CSVs + `stations_summary.csv`
- `YYYYMM.tar.zst` (201301–202405, 137 files) → daily NetCDF `crop_YYYYMMDD.nc` (each 24h @ 1h)

---

## Variable Mapping (Atmospheric Forcings)

| NetCDF var | Physical variable | Unit | Dimensions |
|-----------|------------------|------|------------|
| `var1`  | Sea Level Pressure (MSL) | Pa   | (time, alt=0, lat, lon) |
| `var11` | 2m Temperature (t2m)     | K    | (time, height=2m, lat, lon) |
| `var33` | U wind component (u10)   | m/s  | (time, lat, lon) |
| `var34` | V wind component (v10)   | m/s  | (time, lat, lon) |

---

## 15 Tide Gauge Stations

Stations are in Danish waters (Belt Sea / Western Baltic).  
Coordinates from `stations_summary.csv`:

| Station | ID | Approx. lat | Approx. lon |
|---------|-----|------------|------------|
| Assens | 28366 | 55.27 | 9.88 |
| Bagenkop | 28548 | 54.75 | 10.67 |
| Dragør | 30361 | 55.59 | 12.67 |
| Faaborg | 28397 | 55.09 | 10.24 |
| Fynshav | 26457 | 55.01 | 9.98 |
| Gedser | 31616 | 54.57 | 11.93 |
| Haderslev | 26088 | 55.25 | 9.49 |
| Hesnæs | 31493 | 54.80 | 12.15 |
| København | 30336 | 55.69 | 12.60 |
| Køge | 30478 | 55.45 | 12.20 |
| Kolding | 23322 | 55.49 | 9.48 |
| Rødby | 31573 | 54.65 | 11.35 |
| Sønderborg | 26473 | 54.92 | 9.79 |
| Svendborg | 27000 | 55.06 | 10.61 |
| Tejn | 32048 | 55.25 | 14.83 |

---

## Spatial Collocation Strategy

- **FORCOAST → TG**: **Nearest-neighbor non-NaN** via haversine distance on full float64 coordinates.
  Grid step: Δlat = 0.008° (~890 m), Δlon ≈ 0.0139° (~870 m).
  Land cells are NaN in the GRIB → pick the closest **water cell**.
  Typical station-to-cell distance: < 500 m.
- **Atmo → TG**: **Nearest-neighbor** via haversine (grid ~5.5 km, fields vary smoothly).
  All grid cells are valid (no land mask), so straight `sel(method='nearest')`.

Coordinate precision: TG stations have 6 decimal places (~0.1 m accuracy);
FORCOAST grid stores lat in 3 dp / lon in 8 dp, but internally float64.
The haversine search uses full precision from both sources.

No interpolation, no median, no spatial averaging.

---

## Temporal Alignment

- **Target resolution: 1 hour** (common to FORCOAST and Atmo)
- **TG (10 min → 1h)**: filter rows whose timestamp ends in `XX00` (on-the-hour).
  Every 6th row starting from index 0. One day = 24 hourly values out of 144 rows.
  **No fallback / no interpolation**: if the on-the-hour value is `999` → store as NaN.
- **FORCOAST**: directly at hourly timestamps (24 steps per daily GRIB file)
- **Atmo**: directly at hourly timestamps (24 steps per daily `.nc` file)
- **Merge**: inner join on hourly UTC timestamps

### Key principle

The output dataset contains **only raw values** taken directly from the source datasets.
No subtraction, no bias correction, no derived columns.
All elaboration (error computation, demeaning, feature engineering) happens downstream.

---

## Vertical Datum & Bias Decomposition

The two sea-level sources use **different vertical references**:

| Dataset | Vertical Datum | Unit | Typical Range |
|---------|---------------|------|---------------|
| **TG** | **DVR90** (Danish Vertical Reference 1990) | cm (integer) | −163 … +155 cm |
| **HDM (FORCOAST p82)** | **Model geoid / internal MSL** | m (float32) | −0.81 … +0.61 m |

Their difference decomposes into a **static** and a **dynamic** component:

$$\Delta(t) = \beta + \epsilon(t)$$

| Component | Symbol | Meaning | Example (København) |
|-----------|--------|---------|---------------------|
| Static offset | β | Datum mismatch (DVR90 vs model geoid) | ≈ +12 cm |
| Dynamic error | ε(t) | Phenomena the model does not resolve (local surge, non-linearities, bathymetry errors…) | σ ≈ 7 cm |

**Implication for ML**: the subtraction `forcoast_p82_m − tg_obs_m` is valid. A regression model absorbs β in its intercept; a neural network absorbs it in the bias term. No datum correction is required before training.

Optional post-hoc separation:
```
mean_bias    = (forcoast_p82_m − tg_obs_m).mean()   # estimate of β
anomaly_err  = (forcoast_p82_m − tg_obs_m) − mean_bias  # ε(t)
```

---

## Output: One Parquet per Station

`data/per_station/station_{id}_{name}.parquet`

| Column | Type | Source |
|--------|------|--------|
| `time` | datetime64 UTC | hourly index |
| `tg_obs_m` | float64 | TG CSV (cm → m, /100) |
| `forcoast_p82_m` | float64 | FORCOAST GRIB, nearest-neighbor |
| `msl` | float64 | Atmo NetCDF, nearest [Pa] |
| `t2m` | float64 | Atmo NetCDF, nearest [K] |
| `u10` | float64 | Atmo NetCDF, nearest [m/s] |
| `v10` | float64 | Atmo NetCDF, nearest [m/s] |
| `station_id` | str | metadata |
| `station_name` | str | metadata |
| `lat` | float64 | station lat (TG position) |
| `lon` | float64 | station lon (TG position) |
| `forcoast_lat` | float64 | lat of nearest FORCOAST grid cell |
| `forcoast_lon` | float64 | lon of nearest FORCOAST grid cell |
| `forcoast_dist_km` | float64 | distance station ↔ FORCOAST cell [km] |
| `atmo_lat` | float64 | lat of nearest Atmo grid cell |
| `atmo_lon` | float64 | lon of nearest Atmo grid cell |
| `atmo_dist_km` | float64 | distance station ↔ Atmo cell [km] |

---

## Other Model Sources (NOT used in this dataset)

These DKSS sources have **6h temporal resolution** (6 steps/file) and are not included:

| Source | Period | Temporal Res. |
|--------|--------|--------------|
| DKSS2013 | 2013–2019 | 6h |
| DKSS2019 | 2013–2021 | 6h |
| DKSS2020 | 2021–2024 | 6h |
| DKSS_UWCW | 2024–2026 | 6h |

They share the same 482×396 grid as FORCOAST.

---

## Build Run — 2026-03-16

Command: `python build_station_datasets.py --stations 30336 31616 28366 26473 30478 28548`

### Step 1 — FORCOAST HDM

- **2647 daily GRIB files** processed in **951 s**
- Zero read errors

| Station | Water cell (lat, lon) | Dist to TG |
|---------|----------------------|------------|
| Gedser | (54.5630, 11.9094) | 1.405 km |
| København | (55.7070, 12.6038) | 0.430 km |
| Sønderborg | (54.9150, 9.7983) | 0.959 km |
| Assens | (55.2670, 9.8678) | 1.442 km |
| Bagenkop | (54.7390, 10.6594) | 1.634 km |
| Køge | (55.4430, 12.2288) | 2.467 km |

### Step 2 — Atmospheric Forcings (FTP)

- **87/87 months OK** in **436 s** — zero failures or retries
- Minor anomaly: Oct 2017 and Aug 2018 have 738 h instead of 744 h (6 missing hours each)

| Station | Nearest atmo cell (lat, lon) | Dist to TG |
|---------|------------------------------|------------|
| Gedser | (54.58, 11.93) | 0.95 km |
| København | (55.68, 12.58) | 2.95 km |
| Sønderborg | (54.93, 9.78) | 2.17 km |
| Assens | (55.28, 9.88) | 1.22 km |
| Bagenkop | (54.73, 10.68) | 2.46 km |
| Køge | (55.48, 12.18) | 2.92 km |

### Step 3 — Merge & Save

Theoretical hours in overlap period: **63,528**

| Station | ID | Final rows | Dropped | Coverage | TG NaN | FC NaN |
|---------|-----|-----------|---------|----------|--------|--------|
| Gedser | 31616 | 63335 | 193 | 0.997 | 2698 | 0 |
| København | 30336 | 63335 | 193 | 0.997 | 1197 | 0 |
| Sønderborg | 26473 | 63335 | 193 | 0.997 | 2828 | 0 |
| Assens | 28366 | 63335 | 193 | 0.997 | 4862 | 0 |
| Bagenkop | 28548 | 63335 | 193 | 0.997 | 2094 | 0 |
| Køge | 30478 | 63335 | 193 | 0.997 | 810 | 0 |

**Notes:**
- TG fill = 1.000 for all stations — TG CSVs are complete over the full overlap period
- 193 dropped rows identical across all stations — caused by FORCOAST missing hours (not TG)
- FC NaN = 0 everywhere — nearest water cells are always valid
- TG NaN varies: Assens highest (~7.7%), Køge lowest (~1.3%)
- No LOW COVERAGE warnings, no missing FORCOAST or ATMO months
