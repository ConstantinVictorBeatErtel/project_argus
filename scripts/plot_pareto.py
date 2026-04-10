#!/usr/bin/env python3
"""Plot Pareto frontier artifacts to HTML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.pareto_plot import write_pareto_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Pareto frontier HTML visualization.")
    parser.add_argument("--frontier", action="append", required=True, help="Label=path pair. Can be repeated.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/pareto_frontier.html"), help="Output HTML path.")
    return parser.parse_args()


def parse_frontiers(values: list[str]) -> dict[str, Path]:
    frontiers: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Frontier must be Label=path, got {value!r}")
        label, path = value.split("=", 1)
        if not label:
            raise SystemExit("Frontier label cannot be empty")
        frontiers[label] = Path(path)
    return frontiers


def main() -> None:
    args = parse_args()
    output = write_pareto_html(parse_frontiers(args.frontier), args.output)
    print(json.dumps({"output": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
