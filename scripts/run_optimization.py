#!/usr/bin/env python3
"""Run the Phase 1 sparse MILP optimizer from precomputed visibility data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.milp import solve_ground_station_milp


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value) != 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def load_site_feasible(mask_csv: Path, *, metadata_json: Path | None = None) -> np.ndarray:
    """Load a site feasibility vector from ``backhaul_mask.csv``."""

    frame = pd.read_csv(mask_csv)
    column = "b_i" if "b_i" in frame.columns else "backhaul_feasible" if "backhaul_feasible" in frame.columns else None
    if column is None:
        raise ValueError("backhaul mask must include `b_i` or `backhaul_feasible`")

    if metadata_json is not None:
        site_ids = json.loads(metadata_json.read_text(encoding="utf-8"))["site_ids"]
        if "site_id" not in frame.columns:
            raise ValueError("site_id column is required when metadata_json is provided")
        feasible_by_site = {str(row["site_id"]): _parse_bool(row[column]) for _, row in frame.iterrows()}
        missing = [site_id for site_id in site_ids if site_id not in feasible_by_site]
        if missing:
            raise ValueError(f"backhaul mask missing site_ids: {missing[:5]}")
        return np.asarray([feasible_by_site[site_id] for site_id in site_ids], dtype=np.bool_)

    return np.asarray([_parse_bool(value) for value in frame[column].tolist()], dtype=np.bool_)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve satellite ground-station placement MILP.")
    parser.add_argument(
        "--mode",
        choices=("exact", "heuristic", "pareto"),
        default="exact",
        help="Optimization mode. Phase 1 implements exact only.",
    )
    parser.add_argument("--visibility-npz", type=Path, required=True, help="CSR visibility matrix: rows=sat-time, cols=site.")
    parser.add_argument("--service-cost-npz", type=Path, default=None, help="Optional sparse service-cost matrix.")
    parser.add_argument("--demand-npy", type=Path, default=None, help="Optional NumPy demand vector with one value per row.")
    parser.add_argument("--backhaul-mask-csv", type=Path, default=None, help="Optional b_i feasibility mask CSV.")
    parser.add_argument("--visibility-metadata", type=Path, default=None, help="Optional metadata JSON for site_id ordering.")
    parser.add_argument("--coverage", type=float, default=0.9, help="Required weighted coverage fraction.")
    parser.add_argument("--max-sites", type=int, default=None, help="Optional maximum number of ground stations.")
    parser.add_argument("--station-cost", type=float, default=1.0, help="Uniform fixed cost per opened site.")
    parser.add_argument("--time-limit", type=int, default=600, help="CBC time limit in seconds.")
    parser.add_argument("--mip-gap", type=float, default=0.005, help="Relative MIP gap target.")
    parser.add_argument("--solver-log", action="store_true", help="Show solver log output.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional optimization result JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode != "exact":
        raise SystemExit(f"Mode {args.mode!r} is planned but not implemented yet. Use --mode exact for Phase 1.")

    visibility = load_npz(args.visibility_npz)
    service_cost = load_npz(args.service_cost_npz) if args.service_cost_npz is not None else None
    demand = np.load(args.demand_npy) if args.demand_npy is not None else None
    site_feasible = (
        load_site_feasible(args.backhaul_mask_csv, metadata_json=args.visibility_metadata)
        if args.backhaul_mask_csv is not None
        else None
    )

    result = solve_ground_station_milp(
        visibility,
        service_cost=service_cost,
        demand=demand,
        site_feasible=site_feasible,
        station_cost=args.station_cost,
        coverage_requirement=args.coverage,
        max_ground_stations=args.max_sites,
        time_limit_seconds=args.time_limit,
        mip_gap=args.mip_gap,
        msg=args.solver_log,
    )

    payload = {
        "status": result.status,
        "objective_value": result.objective_value,
        "selected_sites": list(result.selected_sites),
        "coverage_fraction": result.coverage_fraction,
        "covered_demand": result.covered_demand,
        "total_demand": result.total_demand,
        "num_assignments": len(result.assignments),
        "assignments": [asdict(assignment) for assignment in result.assignments],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "assignments"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
