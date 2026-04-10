#!/usr/bin/env python3
"""Build sparse latency service-cost matrices from slant-range tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from scipy.sparse import load_npz, save_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.milp import propagation_latency_cost


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sparse service-cost matrix from slant range.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--range-npz", type=Path, default=None, help="Sparse slant range .npz path.")
    parser.add_argument("--output", type=Path, default=None, help="Sparse service-cost .npz output path.")
    parser.add_argument("--alpha", type=float, default=None, help="Propagation delay weight.")
    parser.add_argument("--speed-of-light-km-s", type=float, default=None, help="Signal speed in km/s.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)
    paths = config["paths"]
    latency_cfg = config["latency"]

    range_npz = args.range_npz or PROJECT_ROOT / paths["range_npz"]
    output = args.output or PROJECT_ROOT / paths["service_cost_npz"]
    alpha = args.alpha if args.alpha is not None else latency_cfg["alpha"]
    speed = args.speed_of_light_km_s if args.speed_of_light_km_s is not None else latency_cfg["speed_of_light_km_s"]

    cost = propagation_latency_cost(load_npz(range_npz), speed_of_light_km_s=float(speed))
    cost.data = cost.data * float(alpha)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output, cost, compressed=True)
    print(
        json.dumps(
            {
                "range_npz": str(range_npz),
                "output": str(output),
                "nnz": int(cost.nnz),
                "alpha": float(alpha),
                "speed_of_light_km_s": float(speed),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
