#!/usr/bin/env python3
"""Compare selected-site portfolios from two sensitivity result rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.coverage_maps import load_selection_comparison_frame, write_selection_comparison_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare selected ground-station site sets.")
    parser.add_argument("--metadata", type=Path, required=True, help="Visibility metadata JSON.")
    parser.add_argument("--left-csv", type=Path, required=True, help="Left sensitivity CSV.")
    parser.add_argument("--right-csv", type=Path, required=True, help="Right sensitivity CSV.")
    parser.add_argument("--elevation", type=float, required=True, help="Elevation row to compare.")
    parser.add_argument("--budget", type=int, required=True, help="Station budget row to compare.")
    parser.add_argument("--left-label", default="uniform", help="Left portfolio label.")
    parser.add_argument("--right-label", default="population", help="Right portfolio label.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Site-level comparison CSV.")
    parser.add_argument("--output-json", type=Path, required=True, help="Summary JSON.")
    parser.add_argument("--output-html", type=Path, required=True, help="Comparison map HTML.")
    parser.add_argument("--title", default="Selected Site Comparison", help="Map title.")
    return parser.parse_args()


def _selected_sites_from_row(path: Path, *, elevation: float, budget: int) -> tuple[int, ...]:
    frame = pd.read_csv(path)
    row = frame[
        (frame["elevation_deg"].astype(float) == float(elevation))
        & (frame["max_ground_stations"].astype(int) == int(budget))
    ]
    if row.empty:
        raise ValueError(f"No sensitivity row found in {path} for elevation={elevation}, budget={budget}")
    selected = str(row.iloc[0]["selected_sites"])
    if not selected:
        return ()
    return tuple(int(site) for site in selected.split(",") if site.strip())


def main() -> None:
    args = parse_args()
    left_sites = _selected_sites_from_row(args.left_csv, elevation=args.elevation, budget=args.budget)
    right_sites = _selected_sites_from_row(args.right_csv, elevation=args.elevation, budget=args.budget)
    frame = load_selection_comparison_frame(
        args.metadata,
        left_selected_sites=left_sites,
        right_selected_sites=right_sites,
        left_label=args.left_label,
        right_label=args.right_label,
    )
    selected_frame = frame[frame["comparison_status"] != "candidate"].copy()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_frame.to_csv(args.output_csv, index=False)
    write_selection_comparison_html(
        frame,
        args.output_html,
        left_label=args.left_label,
        right_label=args.right_label,
        title=args.title,
    )

    left_set = set(left_sites)
    right_set = set(right_sites)
    summary = {
        "elevation_deg": float(args.elevation),
        "max_ground_stations": int(args.budget),
        "left_label": args.left_label,
        "right_label": args.right_label,
        "left_selected_count": len(left_set),
        "right_selected_count": len(right_set),
        "overlap_count": len(left_set & right_set),
        "left_only_count": len(left_set - right_set),
        "right_only_count": len(right_set - left_set),
        "jaccard_similarity": len(left_set & right_set) / len(left_set | right_set) if left_set | right_set else 0.0,
        "overlap_sites": sorted(left_set & right_set),
        "left_only_sites": sorted(left_set - right_set),
        "right_only_sites": sorted(right_set - left_set),
        "output_csv": str(args.output_csv),
        "output_html": str(args.output_html),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
