from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


@dataclass
class SplitConfig:
	train_start: Optional[pd.Timestamp] = None
	train_end: Optional[pd.Timestamp] = None
	train_fraction: float = 0.7


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f"{name} missing required columns: {missing}")


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
	lat1_rad = np.radians(lat1)
	lon1_rad = np.radians(lon1)
	lat2_rad = np.radians(lat2)
	lon2_rad = np.radians(lon2)

	dlat = lat2_rad - lat1_rad
	dlon = lon2_rad - lon1_rad

	a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
	c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
	return EARTH_RADIUS_KM * c


def collocate_hdm_to_stations(
	hdm_grid_df: pd.DataFrame,
	stations_df: pd.DataFrame,
	radius_km: float,
	time_col: str = "time",
	hdm_value_col: str = "dt_hdm",
	station_id_col: str = "station_id",
	lat_col: str = "lat",
	lon_col: str = "lon",
) -> pd.DataFrame:
	_require_columns(hdm_grid_df, [time_col, lat_col, lon_col, hdm_value_col], "hdm_grid_df")
	_require_columns(stations_df, [station_id_col, lat_col, lon_col], "stations_df")

	hdm_grid_df = hdm_grid_df.copy()
	stations_df = stations_df.copy()
	hdm_grid_df[time_col] = pd.to_datetime(hdm_grid_df[time_col], utc=False)

	unique_grid = hdm_grid_df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
	collocated_parts: list[pd.DataFrame] = []

	for station in stations_df.itertuples(index=False):
		station_id = getattr(station, station_id_col)
		station_lat = float(getattr(station, lat_col))
		station_lon = float(getattr(station, lon_col))

		distances = _haversine_km(
			station_lat,
			station_lon,
			unique_grid[lat_col].to_numpy(),
			unique_grid[lon_col].to_numpy(),
		)
		selected_grid = unique_grid.loc[distances <= radius_km, [lat_col, lon_col]]

		if selected_grid.empty:
			raise ValueError(
				f"No HDM grid points found within radius_km={radius_km} for station '{station_id}'"
			)

		station_hdm = hdm_grid_df.merge(selected_grid, on=[lat_col, lon_col], how="inner")
		station_collocated = (
			station_hdm.groupby(time_col, as_index=False)[hdm_value_col]
			.median()
			.rename(columns={hdm_value_col: "dt_hdm_at_tg"})
		)
		station_collocated[station_id_col] = station_id
		collocated_parts.append(station_collocated[[station_id_col, time_col, "dt_hdm_at_tg"]])

	return pd.concat(collocated_parts, ignore_index=True).sort_values([station_id_col, time_col])


def align_tg_to_hdm_times(
	collocated_hdm_df: pd.DataFrame,
	tg_df: pd.DataFrame,
	tolerance_minutes: int = 30,
	station_id_col: str = "station_id",
	time_col: str = "time",
	tg_value_col: str = "dt_tg",
) -> pd.DataFrame:
	_require_columns(collocated_hdm_df, [station_id_col, time_col, "dt_hdm_at_tg"], "collocated_hdm_df")
	_require_columns(tg_df, [station_id_col, time_col, tg_value_col], "tg_df")

	collocated_hdm_df = collocated_hdm_df.copy()
	tg_df = tg_df.copy()

	collocated_hdm_df[time_col] = pd.to_datetime(collocated_hdm_df[time_col], utc=False)
	tg_df[time_col] = pd.to_datetime(tg_df[time_col], utc=False)

	aligned_parts: list[pd.DataFrame] = []
	tolerance = pd.Timedelta(minutes=tolerance_minutes)

	for station_id, hdm_station in collocated_hdm_df.groupby(station_id_col):
		tg_station = tg_df[tg_df[station_id_col] == station_id].sort_values(time_col)
		hdm_station = hdm_station.sort_values(time_col)

		merged = pd.merge_asof(
			hdm_station,
			tg_station[[time_col, tg_value_col]],
			on=time_col,
			direction="nearest",
			tolerance=tolerance,
		)
		merged[station_id_col] = station_id
		aligned_parts.append(merged)

	aligned = pd.concat(aligned_parts, ignore_index=True)
	aligned = aligned.rename(columns={tg_value_col: "dt_tg"})
	aligned = aligned.dropna(subset=["dt_tg", "dt_hdm_at_tg"]).sort_values([station_id_col, time_col])
	return aligned


def build_train_mask(
	df: pd.DataFrame,
	time_col: str = "time",
	train_start: Optional[pd.Timestamp] = None,
	train_end: Optional[pd.Timestamp] = None,
	train_fraction: float = 0.7,
) -> pd.Series:
	time_values = pd.to_datetime(df[time_col], utc=False)

	if train_start is not None or train_end is not None:
		mask = pd.Series(True, index=df.index)
		if train_start is not None:
			mask &= time_values >= pd.Timestamp(train_start)
		if train_end is not None:
			mask &= time_values <= pd.Timestamp(train_end)
		return mask

	if not 0.0 < train_fraction <= 1.0:
		raise ValueError("train_fraction must be in the interval (0, 1]")

	unique_times = np.array(sorted(time_values.unique()))
	cutoff_idx = max(1, int(np.floor(len(unique_times) * train_fraction)))
	cutoff_time = pd.Timestamp(unique_times[cutoff_idx - 1])
	return time_values <= cutoff_time


def compute_dynamic_error_target(
	aligned_df: pd.DataFrame,
	is_train: pd.Series,
	station_id_col: str = "station_id",
) -> pd.DataFrame:
	_require_columns(aligned_df, [station_id_col, "dt_hdm_at_tg", "dt_tg"], "aligned_df")
	if len(aligned_df) != len(is_train):
		raise ValueError("is_train length must match aligned_df length")

	out = aligned_df.copy()
	out["is_train"] = is_train.to_numpy(dtype=bool)
	out["r"] = out["dt_hdm_at_tg"] - out["dt_tg"]

	train_bias = (
		out.loc[out["is_train"]]
		.groupby(station_id_col, as_index=False)["r"]
		.mean()
		.rename(columns={"r": "bias_i"})
	)
	out = out.merge(train_bias, on=station_id_col, how="left")

	if out["bias_i"].isna().any():
		missing_station = out.loc[out["bias_i"].isna(), station_id_col].iloc[0]
		raise ValueError(
			f"Station '{missing_station}' has no training samples for bias estimation."
		)

	out["epsilon"] = out["r"] - out["bias_i"]
	return out
