#!/usr/bin/env python3
"""Build a geopolitical eligibility mask for candidate ground-station sites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constraints.geopolitical import DEFAULT_EXCLUSION_ZONES, build_geopolitical_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build geopolitical site eligibility mask.")
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "candidates" / "ground_station_candidates.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "geopolitical_mask.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates_csv)
    mask = build_geopolitical_mask(candidates)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(args.output_csv, index=False)

    excluded = mask[~mask["geopolitical_allowed"]]
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "num_sites": int(mask.shape[0]),
                "allowed_sites": int(mask["geopolitical_allowed"].sum()),
                "excluded_sites": int((~mask["geopolitical_allowed"]).sum()),
                "excluded_by_zone": excluded["excluded_zone"].value_counts().sort_index().to_dict(),
                "policy_scenario": "conservative U.S.-operator sanctions/licensing/market-access exclusion",
                "zones": [zone.name for zone in DEFAULT_EXCLUSION_ZONES],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
