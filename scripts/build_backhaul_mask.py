#!/usr/bin/env python3
"""Build a binary backhaul feasibility mask for candidate ground stations."""

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

from scripts.build_visibility_tensor import load_candidates_csv
from src.constraints.backhaul import (
    compute_backhaul_mask_from_points,
    load_backhaul_points_csv,
    write_backhaul_mask_csv,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute b_i backhaul feasibility for candidate sites.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--candidates-csv", type=Path, default=None, help="Candidate site CSV.")
    parser.add_argument("--backhaul-points-csv", type=Path, default=None, help="IXP/fiber point CSV.")
    parser.add_argument("--output", type=Path, default=None, help="Output backhaul_mask.csv path.")
    parser.add_argument("--max-distance-km", type=float, default=None, help="Maximum feasible distance to backhaul.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Optional candidate row limit.")
    parser.add_argument("--point-limit", type=int, default=None, help="Optional backhaul point row limit.")
    parser.add_argument("--candidate-chunk-size", type=int, default=2048, help="Candidate distance chunk size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)
    paths = config["paths"]
    backhaul_cfg = config["backhaul"]
    scenario_cfg = config["scenario"]

    candidates_csv = args.candidates_csv or PROJECT_ROOT / paths["candidates_file"]
    points_csv = args.backhaul_points_csv or PROJECT_ROOT / paths["backhaul_points"]
    output_path = args.output or PROJECT_ROOT / paths["backhaul_mask_csv"]
    max_distance_km = (
        args.max_distance_km if args.max_distance_km is not None else backhaul_cfg["max_distance_to_backbone_km"]
    )
    candidate_limit = args.candidate_limit if args.candidate_limit is not None else scenario_cfg.get("candidate_limit")

    candidates = load_candidates_csv(candidates_csv, limit=candidate_limit)
    points = load_backhaul_points_csv(points_csv, limit=args.point_limit)
    mask = compute_backhaul_mask_from_points(
        [candidate.latitude_deg for candidate in candidates],
        [candidate.longitude_deg for candidate in candidates],
        points,
        max_distance_km=float(max_distance_km),
        candidate_chunk_size=args.candidate_chunk_size,
    )
    output = write_backhaul_mask_csv(
        output_path,
        site_ids=[candidate.site_id for candidate in candidates],
        mask=mask,
        backhaul_points=points,
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "num_candidates": len(candidates),
                "num_backhaul_points": len(points),
                "num_feasible": int(mask.feasible.sum()),
                "max_distance_km": float(max_distance_km),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
