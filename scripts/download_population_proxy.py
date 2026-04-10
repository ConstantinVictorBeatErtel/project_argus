#!/usr/bin/env python3
"""Download and normalize Natural Earth populated places as a demand proxy."""

from __future__ import annotations

import argparse
import json
import ssl
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import certifi
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_populated_places.geojson"
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "population" / "ne_10m_populated_places.geojson"
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "population" / "populated_places.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "raw" / "population" / "populated_places.metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Natural Earth populated places for demand proxying.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Source GeoJSON URL.")
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW, help="Raw GeoJSON output path.")
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV, help="Normalized CSV output path.")
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA, help="Metadata JSON output path.")
    parser.add_argument("--min-population", type=float, default=50000.0, help="Minimum population retained in CSV.")
    return parser.parse_args()


def download_url(url: str, output_path: Path) -> Path:
    """Download a URL with certifi-backed TLS verification."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "satellite-gs-optimizer/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=120, context=context) as response:
        output_path.write_bytes(response.read())
    return output_path


def normalize_geojson(raw_geojson: Path, csv_output: Path, *, min_population: float) -> pd.DataFrame:
    """Extract lon/lat and population fields from Natural Earth GeoJSON."""

    payload = json.loads(raw_geojson.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        properties = feature.get("properties") or {}
        population = float(properties.get("POP_MAX") or properties.get("pop_max") or 0.0)
        if population < min_population:
            continue
        rows.append(
            {
                "place_id": str(properties.get("scalerank", "")) + "_" + str(properties.get("NAME", properties.get("name", ""))),
                "name": properties.get("NAME") or properties.get("name"),
                "adm0name": properties.get("ADM0NAME") or properties.get("adm0name"),
                "latitude_deg": float(coordinates[1]),
                "longitude_deg": float(coordinates[0]),
                "population": population,
                "rank_max": properties.get("RANK_MAX") or properties.get("rank_max"),
                "source": "natural_earth_populated_places",
            }
        )

    frame = pd.DataFrame(rows).sort_values("population", ascending=False).reset_index(drop=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_output, index=False)
    return frame


def main() -> None:
    args = parse_args()
    raw_path = download_url(args.url, args.raw_output)
    frame = normalize_geojson(raw_path, args.csv_output, min_population=args.min_population)
    metadata = {
        "source": "Natural Earth 10m populated places GeoJSON",
        "url": args.url,
        "raw_output": str(raw_path),
        "csv_output": str(args.csv_output),
        "min_population": float(args.min_population),
        "num_places": int(frame.shape[0]),
        "population_sum": float(frame["population"].sum()) if not frame.empty else 0.0,
        "note": "Population fields are used as a lightweight proxy for GPW/WorldPop raster demand.",
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
