#!/usr/bin/env python3
"""Build Phase 1 demand artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_visibility_tensor import _parse_start_utc
from src.simulation.demand import (
    build_population_weighted_demand_frame,
    build_uniform_demand_frame,
    load_population_points_csv,
    load_visibility_metadata,
    write_demand_outputs,
)
from src.simulation.propagator import propagate_tle_file


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build demand artifacts for satellite-time rows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--model", choices=("uniform", "population-proxy"), default="uniform", help="Demand model to build.")
    parser.add_argument("--visibility-metadata", type=Path, default=None, help="Visibility metadata JSON with dimensions.")
    parser.add_argument("--num-satellites", type=int, default=None, help="Number of satellites if metadata is unavailable.")
    parser.add_argument("--num-times", type=int, default=None, help="Number of time steps if metadata is unavailable.")
    parser.add_argument("--weight", type=float, default=1.0, help="Uniform demand weight per satellite-time row.")
    parser.add_argument("--normalize", action="store_true", help="Normalize demand vector to sum to 1.")
    parser.add_argument("--tle-file", type=Path, default=None, help="TLE file for population-proxy demand.")
    parser.add_argument("--population-points-csv", type=Path, default=None, help="Population proxy points CSV.")
    parser.add_argument("--satellite-limit", type=int, default=None, help="Optional TLE satellite limit.")
    parser.add_argument("--start-utc", type=str, default=None, help="Propagation start time.")
    parser.add_argument("--duration-hours", type=float, default=None, help="Propagation duration in hours.")
    parser.add_argument("--step-seconds", type=int, default=None, help="Propagation step size in seconds.")
    parser.add_argument("--kernel-radius-km", type=float, default=None, help="Population demand kernel radius.")
    parser.add_argument("--population-exponent", type=float, default=0.75, help="Population dampening exponent.")
    parser.add_argument("--floor-weight", type=float, default=0.05, help="Baseline demand added before normalization.")
    parser.add_argument("--min-population", type=float, default=50000.0, help="Minimum population point retained.")
    parser.add_argument("--top-population-points", type=int, default=5000, help="Largest population points retained.")
    parser.add_argument("--row-chunk-size", type=int, default=1024, help="Satellite-time rows per distance chunk.")
    parser.add_argument("--parquet-output", type=Path, default=None, help="Demand parquet output path.")
    parser.add_argument("--npy-output", type=Path, default=None, help="Optional demand vector .npy output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)
    paths = config["paths"]
    time_grid = config["time_grid"]
    scenario = config["scenario"]
    demand_cfg = config["demand"]
    propagation_cfg = config["propagation"]

    parquet_output = args.parquet_output or PROJECT_ROOT / paths["demand_parquet"]
    npy_output = args.npy_output or PROJECT_ROOT / paths["demand_npy"]

    if args.visibility_metadata is not None:
        num_satellites, num_times = load_visibility_metadata(args.visibility_metadata)
    else:
        if args.num_satellites is None or args.num_times is None:
            default_metadata = PROJECT_ROOT / paths["visibility_metadata"]
            if default_metadata.exists():
                num_satellites, num_times = load_visibility_metadata(default_metadata)
            else:
                raise SystemExit("Provide --visibility-metadata or both --num-satellites and --num-times.")
        else:
            num_satellites = args.num_satellites
            num_times = args.num_times

    if args.model == "uniform":
        frame = build_uniform_demand_frame(
            num_satellites=num_satellites,
            num_times=num_times,
            weight=args.weight,
            normalize=args.normalize,
        )
    else:
        tle_file = args.tle_file or PROJECT_ROOT / paths["tle_file"]
        population_csv = args.population_points_csv or PROJECT_ROOT / paths["population_points_csv"]
        satellite_limit = args.satellite_limit if args.satellite_limit is not None else scenario.get("satellite_limit")
        start_utc = _parse_start_utc(args.start_utc or time_grid["start_utc"])
        duration_hours = args.duration_hours if args.duration_hours is not None else time_grid["duration_hours"]
        step_seconds = args.step_seconds if args.step_seconds is not None else time_grid["step_seconds"]
        kernel_radius = args.kernel_radius_km if args.kernel_radius_km is not None else demand_cfg["kernel_radius_km"]
        propagation = propagate_tle_file(
            tle_file,
            start_utc=start_utc,
            duration=timedelta(hours=float(duration_hours)),
            step_seconds=int(step_seconds),
            satellite_limit=satellite_limit,
            include_endpoint=bool(time_grid.get("include_endpoint", False)),
            dtype=np.dtype(propagation_cfg.get("position_dtype", "float32")),
        )
        population_points = load_population_points_csv(
            population_csv,
            min_population=args.min_population,
            top_n=args.top_population_points,
        )
        frame = build_population_weighted_demand_frame(
            propagation.ecef_km,
            population_points,
            kernel_radius_km=float(kernel_radius),
            population_exponent=args.population_exponent,
            floor_weight=args.floor_weight,
            normalize_to_rows=not args.normalize,
            row_chunk_size=args.row_chunk_size,
        )
        if args.normalize:
            frame["demand"] = frame["demand"] / float(frame["demand"].sum())
    parquet_path, npy_path = write_demand_outputs(frame, parquet_path=parquet_output, npy_path=npy_output)

    print(
        json.dumps(
            {
                "model": args.model,
                "num_satellites": num_satellites,
                "num_times": num_times,
                "num_rows": int(frame.shape[0]),
                "total_demand": float(frame["demand"].sum()),
                "parquet_output": str(parquet_path),
                "npy_output": None if npy_path is None else str(npy_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
