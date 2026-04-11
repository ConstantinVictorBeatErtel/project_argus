#!/usr/bin/env python3
"""Compare Phase 2 raster demand results against the uniform baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.coverage_metrics import (
    build_coverage_comparison_frame,
    scenario_comparison_from_frame,
    write_phase2_json,
    write_phase2_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 demand and coverage results.")
    parser.add_argument("--left-csv", type=Path, required=True, help="Baseline sensitivity CSV.")
    parser.add_argument("--right-csv", type=Path, required=True, help="Phase 2 sensitivity CSV.")
    parser.add_argument("--site-comparison-json", type=Path, required=True, help="Selected-site comparison JSON.")
    parser.add_argument("--left-label", default="uniform", help="Baseline label.")
    parser.add_argument("--right-label", default="raster", help="Phase 2 label.")
    parser.add_argument("--target-elevation", type=float, default=25.0, help="Scenario elevation for summary.")
    parser.add_argument("--target-budget", type=int, default=20, help="Scenario station budget for summary.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Comparison CSV output.")
    parser.add_argument("--output-json", type=Path, required=True, help="Comparison JSON output.")
    parser.add_argument("--output-md", type=Path, required=True, help="Markdown summary output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_frame = pd.read_csv(args.left_csv)
    right_frame = pd.read_csv(args.right_csv)
    comparison_frame = build_coverage_comparison_frame(
        left_frame,
        right_frame,
        left_label=args.left_label,
        right_label=args.right_label,
    )
    scenario = scenario_comparison_from_frame(
        comparison_frame,
        elevation_deg=args.target_elevation,
        max_ground_stations=args.target_budget,
        left_label=args.left_label,
        right_label=args.right_label,
    )
    site_comparison_summary = json.loads(args.site_comparison_json.read_text(encoding="utf-8"))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(args.output_csv, index=False)
    write_phase2_json(output_path=args.output_json, target_scenario=scenario, site_comparison_summary=site_comparison_summary)
    write_phase2_markdown(
        output_path=args.output_md,
        comparison_frame=comparison_frame,
        target_scenario=scenario,
        site_comparison_summary=site_comparison_summary,
        left_label=args.left_label,
        right_label=args.right_label,
    )

    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "target_elevation": float(args.target_elevation),
                "target_budget": int(args.target_budget),
                "coverage_delta": scenario.absolute_delta,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
