#!/usr/bin/env python3
"""Run elevation and station-budget sensitivity experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.sparse import load_npz, save_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_service_cost import load_parameters
from scripts.build_visibility_tensor import (
    _parse_start_utc,
    apply_backhaul_mask_csv,
    build_visibility_outputs,
    load_candidates_csv,
)
from scripts.run_optimization import load_site_feasible
from src.evaluation.sensitivity import sensitivity_results_to_frame, solve_budget_sensitivity
from src.optimization.milp import propagation_latency_cost
from src.simulation.propagator import propagate_tle_file


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensitivity over elevation thresholds and station budgets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--tle-file", type=Path, default=None, help="TLE file.")
    parser.add_argument("--candidates-csv", type=Path, default=None, help="Candidate site CSV.")
    parser.add_argument("--backhaul-mask-csv", type=Path, default=None, help="Optional backhaul mask CSV.")
    parser.add_argument("--demand-npy", type=Path, default=None, help="Optional demand vector.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "sensitivity")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "sensitivity_results.csv")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "data" / "processed" / "sensitivity_results.json")
    parser.add_argument("--elevations", type=_parse_float_list, default=(0.0, 10.0, 25.0), help="Comma-separated elevation thresholds.")
    parser.add_argument("--budgets", type=_parse_int_list, default=(5, 10, 15, 20, 30), help="Comma-separated station budgets.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Optional candidate row limit.")
    parser.add_argument("--satellite-limit", type=int, default=None, help="Optional TLE satellite limit.")
    parser.add_argument("--start-utc", type=str, default=None, help="Propagation start time.")
    parser.add_argument("--duration-hours", type=float, default=None, help="Propagation duration in hours.")
    parser.add_argument("--step-seconds", type=int, default=None, help="Propagation step size in seconds.")
    parser.add_argument("--site-chunk-size", type=int, default=None, help="Visibility site chunk size.")
    parser.add_argument("--time-chunk-size", type=int, default=None, help="Visibility time chunk size.")
    parser.add_argument("--station-cost", type=float, default=1.0, help="Uniform fixed cost per opened site.")
    parser.add_argument("--time-limit", type=int, default=120, help="Per-scenario solver time limit.")
    parser.add_argument("--mip-gap", type=float, default=0.005, help="Relative MIP gap target.")
    parser.add_argument("--solver-log", action="store_true", help="Show CBC logs.")
    return parser.parse_args()


def _scenario_paths(output_dir: Path, elevation: float) -> dict[str, Path]:
    label = f"elev_{elevation:g}".replace(".", "p")
    scenario_dir = output_dir / label
    return {
        "dir": scenario_dir,
        "visibility": scenario_dir / "visibility.npz",
        "range": scenario_dir / "slant_range.npz",
        "service_cost": scenario_dir / "service_cost.npz",
        "metadata": scenario_dir / "visibility_metadata.json",
    }


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_parameters(args.config)
    paths = config["paths"]
    scenario = config["scenario"]
    time_grid = config["time_grid"]
    visibility_cfg = config["visibility"]
    propagation_cfg = config["propagation"]
    latency_cfg = config["latency"]

    tle_file = args.tle_file or PROJECT_ROOT / paths["tle_file"]
    candidates_csv = args.candidates_csv or PROJECT_ROOT / paths["candidates_file"]
    demand_npy = args.demand_npy or PROJECT_ROOT / paths["demand_npy"]
    candidate_limit = args.candidate_limit if args.candidate_limit is not None else scenario.get("candidate_limit")
    satellite_limit = args.satellite_limit if args.satellite_limit is not None else scenario.get("satellite_limit")
    start_utc = _parse_start_utc(args.start_utc or time_grid["start_utc"])
    duration_hours = args.duration_hours if args.duration_hours is not None else time_grid["duration_hours"]
    step_seconds = args.step_seconds if args.step_seconds is not None else time_grid["step_seconds"]
    site_chunk_size = args.site_chunk_size if args.site_chunk_size is not None else visibility_cfg["site_chunk_size"]
    time_chunk_size = args.time_chunk_size if args.time_chunk_size is not None else visibility_cfg["time_chunk_size"]

    propagation = propagate_tle_file(
        tle_file,
        start_utc=start_utc,
        duration=timedelta(hours=float(duration_hours)),
        step_seconds=int(step_seconds),
        satellite_limit=satellite_limit,
        include_endpoint=bool(time_grid.get("include_endpoint", False)),
        dtype=np.dtype(propagation_cfg.get("position_dtype", "float32")),
    )
    candidates = load_candidates_csv(candidates_csv, limit=candidate_limit)
    if args.backhaul_mask_csv is not None:
        candidates = apply_backhaul_mask_csv(candidates, args.backhaul_mask_csv)

    demand = np.load(demand_npy) if Path(demand_npy).exists() else None
    all_results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for elevation in args.elevations:
        scenario_paths = _scenario_paths(args.output_dir, elevation)
        scenario_paths["dir"].mkdir(parents=True, exist_ok=True)
        _, range_path, metadata_path = build_visibility_outputs(
            positions_ecef_km=propagation.ecef_km,
            candidates=candidates,
            min_elevation_deg=float(elevation),
            site_chunk_size=int(site_chunk_size),
            time_chunk_size=int(time_chunk_size),
            visibility_output=scenario_paths["visibility"],
            range_output=scenario_paths["range"],
            metadata_output=scenario_paths["metadata"],
            satellite_ids=propagation.satellite_ids,
            epochs_utc=propagation.epochs_utc,
            source="tle_sgp4_sensitivity",
        )
        if range_path is None:
            raise RuntimeError("sensitivity run expected a range matrix")
        service_cost = propagation_latency_cost(
            load_npz(range_path),
            speed_of_light_km_s=float(latency_cfg["speed_of_light_km_s"]),
        )
        service_cost.data = service_cost.data * float(latency_cfg["alpha"])
        save_npz(scenario_paths["service_cost"], service_cost, compressed=True)

        site_feasible = (
            load_site_feasible(args.backhaul_mask_csv, metadata_json=metadata_path)
            if args.backhaul_mask_csv is not None
            else None
        )
        all_results.extend(
            solve_budget_sensitivity(
                load_npz(scenario_paths["visibility"]),
                elevation_deg=float(elevation),
                budgets=args.budgets,
                service_cost=load_npz(scenario_paths["service_cost"]),
                demand=demand,
                station_cost=args.station_cost,
                site_feasible=site_feasible,
                time_limit_seconds=args.time_limit,
                mip_gap=args.mip_gap,
                msg=args.solver_log,
            )
        )

    frame = sensitivity_results_to_frame(all_results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "output_json": str(args.output_json),
                "num_scenarios": int(len(frame)),
                "best_coverage": float(frame["achieved_coverage"].max()) if not frame.empty else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
