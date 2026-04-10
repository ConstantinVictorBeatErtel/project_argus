#!/usr/bin/env python3
"""Write a JSON manifest for the current Phase 1 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.run_manifest import ManifestCommand, build_run_manifest, write_run_manifest


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a run manifest for the Phase 1 proxy pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--output", type=Path, default=None, help="Output manifest JSON path.")
    parser.add_argument("--run-name", default="phase1_proxy_tle_run", help="Human-readable run name.")
    parser.add_argument("--coverage", type=float, required=True, help="Coverage target used in the optimizer.")
    parser.add_argument("--max-sites", type=int, required=True, help="Maximum ground stations used in the optimizer.")
    parser.add_argument("--min-elevation-deg", type=float, required=True, help="Visibility elevation threshold.")
    parser.add_argument("--backhaul-distance-km", type=float, required=True, help="Backhaul threshold used for b_i.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)
    paths = config["paths"]
    output = args.output or PROJECT_ROOT / paths["run_manifest_json"]

    artifacts = {
        "tle_file": str(PROJECT_ROOT / paths["tle_file"]),
        "candidates_file": str(PROJECT_ROOT / paths["candidates_file"]),
        "backhaul_points": str(PROJECT_ROOT / paths["backhaul_points"]),
        "backhaul_mask_csv": str(PROJECT_ROOT / paths["backhaul_mask_csv"]),
        "positions_h5": str(PROJECT_ROOT / paths["positions_h5"]),
        "visibility_npz": str(PROJECT_ROOT / paths["visibility_npz"]),
        "range_npz": str(PROJECT_ROOT / paths["range_npz"]),
        "service_cost_npz": str(PROJECT_ROOT / paths["service_cost_npz"]),
        "demand_parquet": str(PROJECT_ROOT / paths["demand_parquet"]),
        "demand_npy": str(PROJECT_ROOT / paths["demand_npy"]),
        "optimization_result_json": str(PROJECT_ROOT / paths["optimization_result_json"]),
    }
    commands = [
        ManifestCommand("candidate_grid", "python3 -B scripts/build_candidate_grid.py --max-sites 50 --lat-step 8 --lon-step 8"),
        ManifestCommand("proxy_backhaul", "python3 -B scripts/build_proxy_backhaul.py"),
        ManifestCommand("download_tle", "python3 -B scripts/download_tle.py --limit 20 --timeout 30"),
        ManifestCommand("backhaul_mask", f"python3 -B scripts/build_backhaul_mask.py --max-distance-km {args.backhaul_distance_km} --candidate-limit 50"),
        ManifestCommand(
            "visibility",
            "python3 -B scripts/build_visibility_tensor.py --backhaul-mask-csv data/processed/backhaul_mask.csv "
            f"--range-output data/processed/slant_range.npz --positions-h5-output data/processed/positions.h5 "
            f"--candidate-limit 50 --satellite-limit 20 --min-elevation-deg {args.min_elevation_deg}",
        ),
        ManifestCommand("service_cost", "python3 -B scripts/build_service_cost.py"),
        ManifestCommand("demand", "python3 -B scripts/build_demand.py"),
        ManifestCommand(
            "optimization",
            "python3 -B scripts/run_optimization.py --visibility-npz data/processed/visibility.npz "
            "--visibility-metadata data/processed/visibility_metadata.json --backhaul-mask-csv data/processed/backhaul_mask.csv "
            "--service-cost-npz data/processed/service_cost.npz --demand-npy data/processed/demand.npy "
            f"--coverage {args.coverage} --max-sites {args.max_sites} --time-limit 120 --mip-gap 0.005 "
            "--output-json data/processed/optimization_result.json",
        ),
    ]
    manifest = build_run_manifest(
        run_name=args.run_name,
        project_root=PROJECT_ROOT,
        commands=commands,
        artifacts=artifacts,
        parameters={
            "coverage": args.coverage,
            "max_sites": args.max_sites,
            "min_elevation_deg": args.min_elevation_deg,
            "backhaul_distance_km": args.backhaul_distance_km,
            "candidate_limit": config["scenario"]["candidate_limit"],
            "satellite_limit": config["scenario"]["satellite_limit"],
            "duration_hours": config["time_grid"]["duration_hours"],
            "step_seconds": config["time_grid"]["step_seconds"],
        },
        notes=[
            "Generated proxy candidate and backhaul inputs; not authoritative deployment data.",
            "Uniform demand baseline; GPW demand not yet integrated.",
        ],
    )
    out = write_run_manifest(manifest, output)
    print(json.dumps({"output": str(out), "run_name": args.run_name}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
