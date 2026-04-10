#!/usr/bin/env python3
"""Run an epsilon-constraint Pareto sweep over coverage targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.sparse import load_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_optimization import load_site_feasible
from src.optimization.pareto import coverage_grid, pareto_points_to_frame, solve_pareto_sweep


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pareto coverage/cost sweep.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--visibility-npz", type=Path, default=None, help="CSR visibility matrix.")
    parser.add_argument("--service-cost-npz", type=Path, default=None, help="Optional sparse service-cost matrix.")
    parser.add_argument("--demand-npy", type=Path, default=None, help="Optional demand vector.")
    parser.add_argument("--backhaul-mask-csv", type=Path, default=None, help="Optional b_i feasibility mask CSV.")
    parser.add_argument("--visibility-metadata", type=Path, default=None, help="Optional metadata JSON for site ordering.")
    parser.add_argument("--coverage-start", type=float, default=0.02, help="First coverage target.")
    parser.add_argument("--coverage-stop", type=float, default=0.20, help="Last coverage target.")
    parser.add_argument("--coverage-step", type=float, default=0.02, help="Coverage target step.")
    parser.add_argument("--max-sites", type=int, default=None, help="Optional maximum number of sites.")
    parser.add_argument("--station-cost", type=float, default=1.0, help="Uniform station fixed cost.")
    parser.add_argument("--time-limit", type=int, default=60, help="Per-target solver time limit.")
    parser.add_argument("--mip-gap", type=float, default=0.005, help="Relative MIP gap target.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output frontier CSV path.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output frontier JSON path.")
    parser.add_argument("--solver-log", action="store_true", help="Show solver logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)
    paths = config["paths"]
    optimization_cfg = config["optimization"]

    visibility_path = args.visibility_npz or PROJECT_ROOT / paths["visibility_npz"]
    service_cost_path = args.service_cost_npz or PROJECT_ROOT / paths["service_cost_npz"]
    demand_path = args.demand_npy or PROJECT_ROOT / paths["demand_npy"]
    mask_path = args.backhaul_mask_csv or PROJECT_ROOT / paths["backhaul_mask_csv"]
    metadata_path = args.visibility_metadata or PROJECT_ROOT / paths["visibility_metadata"]
    output_csv = args.output_csv or PROJECT_ROOT / paths["pareto_frontier_csv"]
    output_json = args.output_json or PROJECT_ROOT / paths["pareto_frontier_json"]
    max_sites = args.max_sites if args.max_sites is not None else optimization_cfg.get("max_ground_stations")

    targets = coverage_grid(args.coverage_start, args.coverage_stop, args.coverage_step)
    points = solve_pareto_sweep(
        load_npz(visibility_path),
        service_cost=load_npz(service_cost_path) if Path(service_cost_path).exists() else None,
        demand=np.load(demand_path) if Path(demand_path).exists() else None,
        station_cost=args.station_cost,
        coverage_targets=targets,
        max_ground_stations=max_sites,
        site_feasible=load_site_feasible(mask_path, metadata_json=metadata_path) if Path(mask_path).exists() else None,
        time_limit_seconds=args.time_limit,
        mip_gap=args.mip_gap,
        msg=args.solver_log,
    )
    frame = pareto_points_to_frame(points)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "output_json": str(output_json),
                "num_targets": len(points),
                "num_optimal": int((frame["status"] == "Optimal").sum()),
                "max_optimal_target": None
                if frame[frame["status"] == "Optimal"].empty
                else float(frame.loc[frame["status"] == "Optimal", "coverage_target"].max()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
