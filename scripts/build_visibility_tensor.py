#!/usr/bin/env python3
"""Build canonical sparse visibility tensors from positions and site candidates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.propagator import PropagationResult, propagate_tle_file, write_ecef_hdf5
from src.simulation.visibility import (
    GroundStationCandidate,
    VisibilityMetadata,
    build_visibility_and_range_csr,
    candidates_to_arrays,
    save_visibility_npz,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"


def _first_present(row: pd.Series, names: tuple[str, ...], *, default: Any = None) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _parse_bool(value: Any, *, default: bool = True) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value) != 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def load_candidates_csv(path: str | Path, *, limit: int | None = None) -> list[GroundStationCandidate]:
    """Load candidate ground-station sites from CSV.

    Accepted columns:
        site id: ``site_id``, ``id``, ``name``
        latitude: ``latitude_deg``, ``lat``, ``latitude``
        longitude: ``longitude_deg``, ``lon``, ``lng``, ``longitude``
        altitude: ``altitude_m``, ``alt_m``, ``elevation_m``
        feasibility: ``backhaul_feasible``, ``b_i``, ``feasible``
        regulatory: ``regulatory_allowed``, ``itu_allowed``, ``allowed``
    """

    candidate_path = Path(path)
    frame = pd.read_csv(candidate_path)
    if limit is not None:
        if limit <= 0:
            return []
        frame = frame.head(limit)

    candidates: list[GroundStationCandidate] = []
    for row_idx, row in frame.iterrows():
        latitude = _first_present(row, ("latitude_deg", "lat", "latitude"))
        longitude = _first_present(row, ("longitude_deg", "lon", "lng", "longitude"))
        if latitude is None or longitude is None:
            raise ValueError(
                "Candidate CSV must include latitude/longitude columns. "
                f"Failed at row {row_idx} in {candidate_path}."
            )

        site_id = _first_present(row, ("site_id", "id", "name"), default=f"site_{row_idx:05d}")
        altitude_m = float(_first_present(row, ("altitude_m", "alt_m", "elevation_m"), default=0.0))
        backhaul_feasible = _parse_bool(_first_present(row, ("backhaul_feasible", "b_i", "feasible"), default=True))
        regulatory_allowed = _parse_bool(_first_present(row, ("regulatory_allowed", "itu_allowed", "allowed"), default=True))

        candidates.append(
            GroundStationCandidate(
                site_id=str(site_id),
                latitude_deg=float(latitude),
                longitude_deg=float(longitude),
                altitude_m=altitude_m,
                backhaul_feasible=backhaul_feasible,
                regulatory_allowed=regulatory_allowed,
            )
        )

    if not candidates:
        raise ValueError(f"No candidates loaded from {candidate_path}")
    return candidates


def apply_backhaul_mask_csv(
    candidates: list[GroundStationCandidate],
    mask_path: str | Path,
) -> list[GroundStationCandidate]:
    """Apply a ``backhaul_mask.csv`` file to candidate feasibility flags."""

    frame = pd.read_csv(mask_path)
    if "b_i" not in frame.columns and "backhaul_feasible" not in frame.columns:
        raise ValueError("backhaul mask must include `b_i` or `backhaul_feasible`")

    if "site_id" in frame.columns:
        feasible_by_site = {
            str(row["site_id"]): _parse_bool(row["b_i"] if "b_i" in frame.columns else row["backhaul_feasible"])
            for _, row in frame.iterrows()
        }
        missing = [candidate.site_id for candidate in candidates if candidate.site_id not in feasible_by_site]
        if missing:
            raise ValueError(f"backhaul mask missing site_ids: {missing[:5]}")
        flags = [feasible_by_site[candidate.site_id] for candidate in candidates]
    else:
        if frame.shape[0] != len(candidates):
            raise ValueError("row-order backhaul mask length must match candidates")
        column = "b_i" if "b_i" in frame.columns else "backhaul_feasible"
        flags = [_parse_bool(value) for value in frame[column].tolist()]

    return [
        GroundStationCandidate(
            site_id=candidate.site_id,
            latitude_deg=candidate.latitude_deg,
            longitude_deg=candidate.longitude_deg,
            altitude_m=candidate.altitude_m,
            backhaul_feasible=candidate.backhaul_feasible and bool(flag),
            regulatory_allowed=candidate.regulatory_allowed,
        )
        for candidate, flag in zip(candidates, flags, strict=True)
    ]


def load_positions_npy(path: str | Path) -> np.ndarray:
    """Load ECEF positions from a NumPy file."""

    positions = np.load(Path(path))
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions .npy must have shape (num_satellites, num_times, 3)")
    return np.asarray(positions, dtype=np.float32)


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _parse_start_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _metadata_payload(
    metadata: VisibilityMetadata,
    *,
    candidates: list[GroundStationCandidate],
    satellite_ids: tuple[str, ...] | None,
    epochs_utc: tuple[datetime, ...] | None,
    source: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = asdict(metadata)
    payload["source"] = source
    payload["site_ids"] = [candidate.site_id for candidate in candidates]
    payload["candidate_sites"] = [asdict(candidate) for candidate in candidates]
    if satellite_ids is not None:
        payload["satellite_ids"] = list(satellite_ids)
    if epochs_utc is not None:
        payload["epochs_utc"] = [epoch.isoformat().replace("+00:00", "Z") for epoch in epochs_utc]
    return payload


def build_visibility_outputs(
    *,
    positions_ecef_km: np.ndarray,
    candidates: list[GroundStationCandidate],
    min_elevation_deg: float,
    site_chunk_size: int,
    time_chunk_size: int,
    visibility_output: str | Path,
    metadata_output: str | Path,
    range_output: str | Path | None = None,
    satellite_ids: tuple[str, ...] | None = None,
    epochs_utc: tuple[datetime, ...] | None = None,
    source: str = "precomputed_positions",
) -> tuple[Path, Path | None, Path]:
    """Build and persist visibility, optional slant range, and metadata."""

    site_ids, latitudes, longitudes, altitudes, feasible = candidates_to_arrays(candidates)
    visibility, ranges = build_visibility_and_range_csr(
        positions_ecef_km,
        latitudes,
        longitudes,
        altitudes,
        feasible_mask=feasible,
        min_elevation_deg=min_elevation_deg,
        site_chunk_size=site_chunk_size,
        time_chunk_size=time_chunk_size,
    )

    metadata = VisibilityMetadata(
        num_sites=len(site_ids),
        num_satellites=int(positions_ecef_km.shape[0]),
        num_times=int(positions_ecef_km.shape[1]),
        min_elevation_deg=float(min_elevation_deg),
    )
    visibility_path = save_visibility_npz(visibility, visibility_output, metadata=metadata, metadata_path=metadata_output)

    metadata_payload = _metadata_payload(
        metadata,
        candidates=candidates,
        satellite_ids=satellite_ids,
        epochs_utc=epochs_utc,
        source=source,
    )
    metadata_path = Path(metadata_output)
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")

    range_path: Path | None = None
    if range_output is not None:
        range_path = save_visibility_npz(ranges, range_output, compressed=True)

    return visibility_path, range_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sparse visibility.npz from satellite positions and candidate sites.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--positions-npy", type=Path, default=None, help="Precomputed ECEF positions, shape sats x times x 3.")
    parser.add_argument("--tle-file", type=Path, default=None, help="TLE file used when positions are not supplied.")
    parser.add_argument("--candidates-csv", type=Path, default=None, help="Ground-station candidate CSV.")
    parser.add_argument("--backhaul-mask-csv", type=Path, default=None, help="Optional precomputed backhaul mask CSV.")
    parser.add_argument("--visibility-output", type=Path, default=None, help="Output visibility .npz path.")
    parser.add_argument("--range-output", type=Path, default=None, help="Optional output slant-range .npz path.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Output metadata JSON path.")
    parser.add_argument("--positions-h5-output", type=Path, default=None, help="Optional propagated ECEF HDF5 output.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Optional candidate row limit.")
    parser.add_argument("--satellite-limit", type=int, default=None, help="Optional TLE satellite limit.")
    parser.add_argument("--start-utc", type=str, default=None, help="Propagation start time, ISO-8601 UTC.")
    parser.add_argument("--duration-hours", type=float, default=None, help="Propagation duration in hours.")
    parser.add_argument("--step-seconds", type=int, default=None, help="Propagation step size in seconds.")
    parser.add_argument("--include-endpoint", action="store_true", help="Include propagation endpoint if aligned to the grid.")
    parser.add_argument("--min-elevation-deg", type=float, default=None, help="Minimum elevation angle for visibility.")
    parser.add_argument("--site-chunk-size", type=int, default=None, help="Vectorized site chunk size.")
    parser.add_argument("--time-chunk-size", type=int, default=None, help="Vectorized time chunk size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)

    paths = config["paths"]
    scenario = config["scenario"]
    time_grid = config["time_grid"]
    visibility_cfg = config["visibility"]
    propagation_cfg = config["propagation"]

    candidates_csv = args.candidates_csv or PROJECT_ROOT / paths["candidates_file"]
    visibility_output = args.visibility_output or PROJECT_ROOT / paths["visibility_npz"]
    metadata_output = args.metadata_output or PROJECT_ROOT / paths["visibility_metadata"]
    candidate_limit = args.candidate_limit if args.candidate_limit is not None else scenario.get("candidate_limit")
    satellite_limit = args.satellite_limit if args.satellite_limit is not None else scenario.get("satellite_limit")
    min_elevation_deg = args.min_elevation_deg if args.min_elevation_deg is not None else visibility_cfg["min_elevation_deg"]
    site_chunk_size = args.site_chunk_size if args.site_chunk_size is not None else visibility_cfg["site_chunk_size"]
    time_chunk_size = args.time_chunk_size if args.time_chunk_size is not None else visibility_cfg["time_chunk_size"]

    candidates = load_candidates_csv(candidates_csv, limit=candidate_limit)
    if args.backhaul_mask_csv is not None:
        candidates = apply_backhaul_mask_csv(candidates, args.backhaul_mask_csv)

    satellite_ids: tuple[str, ...] | None = None
    epochs_utc: tuple[datetime, ...] | None = None
    source = "precomputed_positions"
    if args.positions_npy is not None:
        positions = load_positions_npy(args.positions_npy)
    else:
        tle_file = args.tle_file or PROJECT_ROOT / paths["tle_file"]
        start_utc = _parse_start_utc(args.start_utc or time_grid["start_utc"])
        duration_hours = args.duration_hours if args.duration_hours is not None else time_grid["duration_hours"]
        step_seconds = args.step_seconds if args.step_seconds is not None else time_grid["step_seconds"]
        include_endpoint = bool(args.include_endpoint or time_grid.get("include_endpoint", False))
        result: PropagationResult = propagate_tle_file(
            tle_file,
            start_utc=start_utc,
            duration=timedelta(hours=float(duration_hours)),
            step_seconds=int(step_seconds),
            satellite_limit=satellite_limit,
            include_endpoint=include_endpoint,
            dtype=np.dtype(propagation_cfg.get("position_dtype", "float32")),
        )
        positions = result.ecef_km
        satellite_ids = result.satellite_ids
        epochs_utc = result.epochs_utc
        source = "tle_sgp4"

        if args.positions_h5_output is not None:
            write_ecef_hdf5(
                result,
                args.positions_h5_output,
                compression=propagation_cfg.get("hdf5_compression", "gzip"),
                compression_opts=propagation_cfg.get("hdf5_compression_level", 4),
            )

    visibility_path, range_path, metadata_path = build_visibility_outputs(
        positions_ecef_km=positions,
        candidates=candidates,
        min_elevation_deg=float(min_elevation_deg),
        site_chunk_size=int(site_chunk_size),
        time_chunk_size=int(time_chunk_size),
        visibility_output=visibility_output,
        metadata_output=metadata_output,
        range_output=args.range_output,
        satellite_ids=satellite_ids,
        epochs_utc=epochs_utc,
        source=source,
    )

    print(
        json.dumps(
            {
                "visibility_output": str(visibility_path),
                "range_output": None if range_path is None else str(range_path),
                "metadata_output": str(metadata_path),
                "num_satellites": int(positions.shape[0]),
                "num_times": int(positions.shape[1]),
                "num_sites": len(candidates),
                "source": source,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
