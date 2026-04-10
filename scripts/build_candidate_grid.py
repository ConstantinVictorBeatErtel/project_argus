#!/usr/bin/env python3
"""Generate proxy candidate ground-station sites."""

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

from src.simulation.candidates import generate_candidate_grid, write_candidate_grid_csv


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a proxy candidate-site grid.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--output", type=Path, default=None, help="Output candidate CSV path.")
    parser.add_argument("--lat-step", type=float, default=10.0, help="Latitude grid step in degrees.")
    parser.add_argument("--lon-step", type=float, default=10.0, help="Longitude grid step in degrees.")
    parser.add_argument("--min-lat", type=float, default=-60.0, help="Minimum latitude.")
    parser.add_argument("--max-lat", type=float, default=75.0, help="Maximum latitude.")
    parser.add_argument("--min-lon", type=float, default=-180.0, help="Minimum longitude.")
    parser.add_argument("--max-lon", type=float, default=180.0, help="Maximum longitude.")
    parser.add_argument("--include-ocean", action="store_true", help="Disable rough land bounding-box filter.")
    parser.add_argument("--max-sites", type=int, default=None, help="Optional deterministic downsample size.")
    parser.add_argument(
        "--downsample-method",
        choices=("farthest", "linspace"),
        default="farthest",
        help="Downsampling strategy when the grid exceeds max-sites.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)
    paths = config["paths"]
    scenario = config["scenario"]

    output = args.output or PROJECT_ROOT / paths["candidates_file"]
    max_sites = args.max_sites if args.max_sites is not None else scenario.get("candidate_limit")
    frame = generate_candidate_grid(
        latitude_step_deg=args.lat_step,
        longitude_step_deg=args.lon_step,
        min_latitude_deg=args.min_lat,
        max_latitude_deg=args.max_lat,
        min_longitude_deg=args.min_lon,
        max_longitude_deg=args.max_lon,
        rough_land_only=not args.include_ocean,
        max_sites=max_sites,
        downsample_method=args.downsample_method,
    )
    out = write_candidate_grid_csv(frame, output)
    print(
        json.dumps(
            {
                "output": str(out),
                "num_candidates": int(frame.shape[0]),
                "candidate_source": frame["candidate_source"].iloc[0] if not frame.empty else None,
                "lat_step": args.lat_step,
                "lon_step": args.lon_step,
                "downsample_method": args.downsample_method,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
