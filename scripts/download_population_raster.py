#!/usr/bin/env python3
"""Download a global WorldPop population raster for Phase 2 demand modeling."""

from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

import certifi


DEFAULT_URL = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2024/0_Mosaicked/v1/1km/constrained/global_pop_2024_CN_1km_R2024B_v1.tif"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "population" / "worldpop_2024_1km_constrained.tif"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "raw" / "population" / "worldpop_2024_1km_constrained.metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an official WorldPop global 1km raster.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Official WorldPop raster URL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output GeoTIFF path.")
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA, help="Metadata JSON path.")
    return parser.parse_args()


def download_file(url: str, output_path: Path) -> tuple[Path, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "satellite-gs-optimizer/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=300, context=context) as response:
        size = int(response.headers.get("content-length") or 0)
        with output_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return output_path, size


def main() -> None:
    args = parse_args()
    output_path, expected_size = download_file(args.url, args.output)
    actual_size = output_path.stat().st_size
    metadata = {
        "source": "WorldPop global 1km constrained population raster",
        "url": args.url,
        "output": str(output_path),
        "expected_size_bytes": expected_size,
        "downloaded_size_bytes": actual_size,
        "note": "Used as the Phase 2 raster demand source in place of GPW because it is official, global, directly downloadable, and much smaller than the unconstrained mosaic.",
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
