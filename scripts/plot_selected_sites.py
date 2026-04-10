#!/usr/bin/env python3
"""Plot selected candidate sites to HTML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.coverage_maps import write_selected_site_map_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write selected-site map HTML.")
    parser.add_argument("--metadata", type=Path, required=True, help="Visibility metadata JSON.")
    parser.add_argument("--result", type=Path, required=True, help="Optimization result JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path.")
    parser.add_argument("--title", default="Selected Ground Station Sites", help="Map title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_selected_site_map_html(args.metadata, args.result, args.output, title=args.title)
    print(json.dumps({"output": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
