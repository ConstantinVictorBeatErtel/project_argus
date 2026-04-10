#!/usr/bin/env python3
"""Plot sensitivity results to HTML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.sensitivity_plot import write_sensitivity_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write sensitivity analysis HTML visualization.")
    parser.add_argument("--results-csv", type=Path, default=Path("data/processed/sensitivity_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/sensitivity_coverage.html"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_sensitivity_html(args.results_csv, args.output)
    print(json.dumps({"output": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
