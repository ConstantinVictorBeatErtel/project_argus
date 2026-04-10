#!/usr/bin/env python3
"""Create a tiny deterministic instance for end-to-end pipeline checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.visibility import geodetic_to_ecef_km, local_up_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic candidates, backhaul points, and ECEF positions.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for synthetic files.")
    parser.add_argument("--num-satellites", type=int, default=3, help="Number of synthetic satellites.")
    parser.add_argument("--num-times", type=int, default=4, help="Number of synthetic time steps.")
    parser.add_argument("--num-sites", type=int, default=5, help="Number of candidate sites.")
    parser.add_argument("--altitude-km", type=float, default=550.0, help="Synthetic satellite altitude above site radial up.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_satellites <= 0 or args.num_times <= 0 or args.num_sites <= 0:
        raise SystemExit("num-satellites, num-times, and num-sites must be positive")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    candidates_csv = out / "synthetic_candidates.csv"
    backhaul_csv = out / "synthetic_backhaul.csv"
    positions_npy = out / "synthetic_positions.npy"
    manifest_json = out / "synthetic_manifest.json"

    latitudes = np.linspace(-20.0, 20.0, args.num_sites)
    longitudes = np.linspace(-60.0, 60.0, args.num_sites)
    candidate_lines = ["site_id,latitude_deg,longitude_deg,altitude_m"]
    for idx, (lat, lon) in enumerate(zip(latitudes, longitudes, strict=True)):
        candidate_lines.append(f"site-{idx},{lat:.6f},{lon:.6f},0.0")
    candidates_csv.write_text("\n".join(candidate_lines) + "\n", encoding="utf-8")

    backhaul_lines = ["id,lat,lon,type"]
    for idx in range(0, args.num_sites, 2):
        backhaul_lines.append(f"ixp-{idx},{latitudes[idx]:.6f},{longitudes[idx]:.6f},ixp")
    backhaul_csv.write_text("\n".join(backhaul_lines) + "\n", encoding="utf-8")

    site_ecef = geodetic_to_ecef_km(latitudes, longitudes, 0.0)
    site_up = local_up_vectors(latitudes, longitudes)
    positions = np.empty((args.num_satellites, args.num_times, 3), dtype=np.float32)
    for sat_idx in range(args.num_satellites):
        for time_idx in range(args.num_times):
            site_idx = (sat_idx + time_idx) % args.num_sites
            positions[sat_idx, time_idx, :] = site_ecef[site_idx] + args.altitude_km * site_up[site_idx]
    np.save(positions_npy, positions)

    manifest = {
        "candidates_csv": str(candidates_csv),
        "backhaul_csv": str(backhaul_csv),
        "positions_npy": str(positions_npy),
        "num_satellites": args.num_satellites,
        "num_times": args.num_times,
        "num_sites": args.num_sites,
        "altitude_km": args.altitude_km,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
