from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
	SplitConfig,
	align_tg_to_hdm_times,
	build_train_mask,
	collocate_hdm_to_stations,
	compute_dynamic_error_target,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="HDM↔TG overlap and dynamic-error target construction"
	)
	parser.add_argument("--hdm-grid", required=True, help="CSV path with columns: time, lat, lon, dt_hdm")
	parser.add_argument("--stations", required=True, help="CSV path with columns: station_id, lat, lon")
	parser.add_argument("--tg", required=True, help="CSV path with columns: station_id, time, dt_tg")
	parser.add_argument("--output", required=True, help="Output CSV path")
	parser.add_argument("--radius-km", type=float, default=15.0, help="Spatial collocation radius in km")
	parser.add_argument(
		"--tg-tolerance-minutes",
		type=int,
		default=30,
		help="Nearest-time tolerance for TG alignment in minutes",
	)
	parser.add_argument("--train-start", default=None, help="Train start date (inclusive), e.g. 2010-01-01")
	parser.add_argument("--train-end", default=None, help="Train end date (inclusive), e.g. 2019-12-31")
	parser.add_argument(
		"--train-fraction",
		type=float,
		default=0.7,
		help="Fallback chronological train fraction when dates are not provided",
	)
	return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
	hdm_grid_df = pd.read_csv(args.hdm_grid)
	stations_df = pd.read_csv(args.stations)
	tg_df = pd.read_csv(args.tg)

	collocated = collocate_hdm_to_stations(
		hdm_grid_df=hdm_grid_df,
		stations_df=stations_df,
		radius_km=args.radius_km,
	)
	aligned = align_tg_to_hdm_times(
		collocated_hdm_df=collocated,
		tg_df=tg_df,
		tolerance_minutes=args.tg_tolerance_minutes,
	)

	split_cfg = SplitConfig(
		train_start=pd.Timestamp(args.train_start) if args.train_start else None,
		train_end=pd.Timestamp(args.train_end) if args.train_end else None,
		train_fraction=args.train_fraction,
	)
	is_train = build_train_mask(
		aligned,
		train_start=split_cfg.train_start,
		train_end=split_cfg.train_end,
		train_fraction=split_cfg.train_fraction,
	)
	dataset = compute_dynamic_error_target(aligned_df=aligned, is_train=is_train)
	return dataset


def main() -> None:
	args = parse_args()
	dataset = run_pipeline(args)

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	dataset.to_csv(output_path, index=False)

	print(f"Saved dataset with {len(dataset)} rows to: {output_path}")
	print("Columns:", ", ".join(dataset.columns))


if __name__ == "__main__":
	main()
