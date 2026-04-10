#!/usr/bin/env python3
"""Generate proxy internet backbone hub points."""

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

from src.constraints.backhaul import proxy_backhaul_hubs, write_backhaul_points_csv


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build proxy IXP/fiber hub points for development runs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--output", type=Path, default=None, help="Output backhaul point CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)
    output = args.output or PROJECT_ROOT / config["paths"]["backhaul_points"]
    points = proxy_backhaul_hubs()
    out = write_backhaul_points_csv(points, output)
    print(
        json.dumps(
            {
                "output": str(out),
                "num_points": len(points),
                "source": "approximate_proxy_internet_hubs",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
