"""Vectorized visibility tensor construction.

The full ``a_ijt`` tensor is stored as a CSR matrix with shape
``(num_satellites * num_times, num_sites)``. A nonzero at row
``j * num_times + t`` and column ``i`` means satellite ``j`` is visible from
candidate site ``i`` at time index ``t``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, save_npz

WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class GroundStationCandidate:
    """Ground-station candidate metadata used by the visibility engine."""

    site_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    backhaul_feasible: bool = True
    regulatory_allowed: bool = True


@dataclass(frozen=True)
class VisibilityMetadata:
    """Metadata required to invert flattened sparse tensor rows."""

    num_sites: int
    num_satellites: int
    num_times: int
    min_elevation_deg: float
    row_order: str = "row = satellite_index * num_times + time_index"
    matrix_layout: str = "CSR rows=satellite_time columns=sites"
    units: str = "binary visibility"


def geodetic_to_ecef_km(
    latitude_deg: Sequence[float] | NDArray[np.floating],
    longitude_deg: Sequence[float] | NDArray[np.floating],
    altitude_m: Sequence[float] | NDArray[np.floating] | float = 0.0,
) -> NDArray[np.float64]:
    """Convert WGS84 geodetic coordinates to ECEF kilometers."""

    lat = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    alt_km = np.asarray(altitude_m, dtype=np.float64) / 1000.0
    alt_km = np.broadcast_to(alt_km, lat.shape)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    prime_vertical_radius = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (prime_vertical_radius + alt_km) * cos_lat * np.cos(lon)
    y = (prime_vertical_radius + alt_km) * cos_lat * np.sin(lon)
    z = (prime_vertical_radius * (1.0 - WGS84_E2) + alt_km) * sin_lat

    return np.stack((x, y, z), axis=-1)


def local_up_vectors(
    latitude_deg: Sequence[float] | NDArray[np.floating],
    longitude_deg: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return local geodetic up unit vectors for WGS84 latitude/longitude."""

    lat = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def candidates_to_arrays(
    candidates: Sequence[GroundStationCandidate],
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Convert station candidate objects to vectorized NumPy arrays."""

    if not candidates:
        raise ValueError("At least one ground-station candidate is required")

    site_ids = [candidate.site_id for candidate in candidates]
    latitudes = np.asarray([candidate.latitude_deg for candidate in candidates], dtype=np.float64)
    longitudes = np.asarray([candidate.longitude_deg for candidate in candidates], dtype=np.float64)
    altitudes = np.asarray([candidate.altitude_m for candidate in candidates], dtype=np.float64)
    feasible = np.asarray(
        [candidate.backhaul_feasible and candidate.regulatory_allowed for candidate in candidates],
        dtype=np.bool_,
    )
    return site_ids, latitudes, longitudes, altitudes, feasible


def build_visibility_csr_from_candidates(
    satellite_ecef_km: NDArray[np.floating],
    candidates: Sequence[GroundStationCandidate],
    *,
    min_elevation_deg: float = 25.0,
    site_chunk_size: int = 16,
    time_chunk_size: int = 12,
    dtype: np.dtype[np.uint8] = np.dtype("uint8"),
) -> csr_matrix:
    """Build a binary sparse visibility matrix from candidate objects."""

    _, latitudes, longitudes, altitudes, feasible = candidates_to_arrays(candidates)
    return build_visibility_csr(
        satellite_ecef_km=satellite_ecef_km,
        station_latitude_deg=latitudes,
        station_longitude_deg=longitudes,
        station_altitude_m=altitudes,
        feasible_mask=feasible,
        min_elevation_deg=min_elevation_deg,
        site_chunk_size=site_chunk_size,
        time_chunk_size=time_chunk_size,
        dtype=dtype,
    )


def build_visibility_csr(
    satellite_ecef_km: NDArray[np.floating],
    station_latitude_deg: Sequence[float] | NDArray[np.floating],
    station_longitude_deg: Sequence[float] | NDArray[np.floating],
    station_altitude_m: Sequence[float] | NDArray[np.floating] | float = 0.0,
    *,
    feasible_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
    min_elevation_deg: float = 25.0,
    site_chunk_size: int = 16,
    time_chunk_size: int = 12,
    dtype: np.dtype[np.uint8] = np.dtype("uint8"),
) -> csr_matrix:
    """Construct the sparse binary visibility tensor.

    Args:
        satellite_ecef_km: Satellite ECEF tensor with shape
            ``(num_satellites, num_times, 3)`` in kilometers.
        station_latitude_deg: Candidate site latitudes.
        station_longitude_deg: Candidate site longitudes.
        station_altitude_m: Candidate site altitudes in meters.
        feasible_mask: Optional preprocessing mask. Columns with ``False``
            remain all-zero, enforcing constraints such as ``y_i <= b_i``
            upstream.
        min_elevation_deg: Minimum elevation angle for visibility.
        site_chunk_size: Number of sites processed per vectorized chunk.
        time_chunk_size: Number of time steps processed per vectorized chunk.
        dtype: Data dtype for the binary nonzero entries.

    Returns:
        CSR matrix with shape ``(num_satellites * num_times, num_sites)``.
    """

    visibility, _ = _build_sparse_visibility(
        satellite_ecef_km=satellite_ecef_km,
        station_latitude_deg=station_latitude_deg,
        station_longitude_deg=station_longitude_deg,
        station_altitude_m=station_altitude_m,
        feasible_mask=feasible_mask,
        min_elevation_deg=min_elevation_deg,
        site_chunk_size=site_chunk_size,
        time_chunk_size=time_chunk_size,
        visibility_dtype=dtype,
        include_range_km=False,
    )
    return visibility


def build_visibility_and_range_csr(
    satellite_ecef_km: NDArray[np.floating],
    station_latitude_deg: Sequence[float] | NDArray[np.floating],
    station_longitude_deg: Sequence[float] | NDArray[np.floating],
    station_altitude_m: Sequence[float] | NDArray[np.floating] | float = 0.0,
    *,
    feasible_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
    min_elevation_deg: float = 25.0,
    site_chunk_size: int = 16,
    time_chunk_size: int = 12,
    visibility_dtype: np.dtype[np.uint8] = np.dtype("uint8"),
    range_dtype: np.dtype[np.float32] = np.dtype("float32"),
) -> tuple[csr_matrix, csr_matrix]:
    """Construct binary visibility and matching slant-range sparse matrices."""

    visibility, ranges = _build_sparse_visibility(
        satellite_ecef_km=satellite_ecef_km,
        station_latitude_deg=station_latitude_deg,
        station_longitude_deg=station_longitude_deg,
        station_altitude_m=station_altitude_m,
        feasible_mask=feasible_mask,
        min_elevation_deg=min_elevation_deg,
        site_chunk_size=site_chunk_size,
        time_chunk_size=time_chunk_size,
        visibility_dtype=visibility_dtype,
        include_range_km=True,
        range_dtype=range_dtype,
    )
    if ranges is None:  # Defensive only; include_range_km=True guarantees it.
        raise RuntimeError("Range matrix was not generated")
    return visibility, ranges


def _build_sparse_visibility(
    satellite_ecef_km: NDArray[np.floating],
    station_latitude_deg: Sequence[float] | NDArray[np.floating],
    station_longitude_deg: Sequence[float] | NDArray[np.floating],
    station_altitude_m: Sequence[float] | NDArray[np.floating] | float,
    *,
    feasible_mask: Sequence[bool] | NDArray[np.bool_] | None,
    min_elevation_deg: float,
    site_chunk_size: int,
    time_chunk_size: int,
    visibility_dtype: np.dtype[np.uint8],
    include_range_km: bool,
    range_dtype: np.dtype[np.float32] = np.dtype("float32"),
) -> tuple[csr_matrix, csr_matrix | None]:
    if site_chunk_size <= 0:
        raise ValueError("site_chunk_size must be positive")
    if time_chunk_size <= 0:
        raise ValueError("time_chunk_size must be positive")

    sat = np.asarray(satellite_ecef_km, dtype=np.float32)
    if sat.ndim != 3 or sat.shape[2] != 3:
        raise ValueError("satellite_ecef_km must have shape (num_satellites, num_times, 3)")
    num_satellites, num_times, _ = sat.shape

    lat = np.asarray(station_latitude_deg, dtype=np.float64)
    lon = np.asarray(station_longitude_deg, dtype=np.float64)
    if lat.shape != lon.shape:
        raise ValueError("station_latitude_deg and station_longitude_deg must have matching shapes")
    if lat.ndim != 1:
        raise ValueError("station coordinates must be one-dimensional")
    num_sites = lat.size
    if num_sites == 0:
        raise ValueError("At least one station candidate is required")

    alt = np.asarray(station_altitude_m, dtype=np.float64)
    alt = np.broadcast_to(alt, lat.shape)

    if feasible_mask is None:
        feasible = np.ones(num_sites, dtype=np.bool_)
    else:
        feasible = np.asarray(feasible_mask, dtype=np.bool_)
        if feasible.shape != (num_sites,):
            raise ValueError("feasible_mask must have shape (num_sites,)")

    site_ecef = geodetic_to_ecef_km(lat, lon, alt).astype(np.float32, copy=False)
    site_up = local_up_vectors(lat, lon).astype(np.float32, copy=False)
    min_elevation_sin = np.float32(np.sin(np.deg2rad(min_elevation_deg)))

    row_parts: list[NDArray[np.int32]] = []
    col_parts: list[NDArray[np.int32]] = []
    visibility_parts: list[NDArray[np.uint8]] = []
    range_parts: list[NDArray[np.float32]] = []

    for site_start in range(0, num_sites, site_chunk_size):
        site_end = min(site_start + site_chunk_size, num_sites)
        chunk_size = site_end - site_start
        site_positions = site_ecef[site_start:site_end]
        up_vectors = site_up[site_start:site_end]
        chunk_feasible = feasible[site_start:site_end]

        feasible_rows = np.flatnonzero(chunk_feasible)
        if feasible_rows.size > 0:
            for time_start in range(0, num_times, time_chunk_size):
                time_end = min(time_start + time_chunk_size, num_times)
                time_count = time_end - time_start

                sat_block = sat[:, time_start:time_end, :]
                rho = sat_block[None, :, :, :] - site_positions[:, None, None, :]
                slant_range = np.linalg.norm(rho, axis=-1)
                vertical_projection = np.einsum("cstk,ck->cst", rho, up_vectors, optimize=True)
                visible = (slant_range > 0.0) & (vertical_projection >= min_elevation_sin * slant_range)

                for local_row in feasible_rows:
                    flat_visible = np.flatnonzero(visible[local_row])
                    if flat_visible.size == 0:
                        continue

                    satellite_index = flat_visible // time_count
                    local_time_index = flat_visible % time_count
                    rows = satellite_index * num_times + time_start + local_time_index
                    cols = np.full(rows.size, site_start + local_row, dtype=np.int32)
                    row_parts.append(rows.astype(np.int32, copy=False))
                    col_parts.append(cols)
                    visibility_parts.append(np.ones(rows.size, dtype=visibility_dtype))

                    if include_range_km:
                        range_parts.append(
                            slant_range[local_row].reshape(-1)[flat_visible].astype(range_dtype, copy=False)
                        )

    if row_parts:
        rows = np.concatenate(row_parts).astype(np.int32, copy=False)
        cols = np.concatenate(col_parts).astype(np.int32, copy=False)
        visibility_data = np.concatenate(visibility_parts).astype(visibility_dtype, copy=False)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        visibility_data = np.empty(0, dtype=visibility_dtype)

    shape = (num_satellites * num_times, num_sites)
    visibility_csr = csr_matrix((visibility_data, (rows, cols)), shape=shape)
    visibility_csr.sum_duplicates()

    if not include_range_km:
        return visibility_csr, None

    if range_parts:
        range_data = np.concatenate(range_parts).astype(range_dtype, copy=False)
    else:
        range_data = np.empty(0, dtype=range_dtype)
    range_csr = csr_matrix((range_data, (rows.copy(), cols.copy())), shape=shape)
    range_csr.sum_duplicates()
    return visibility_csr, range_csr


def sparse_row_to_satellite_time(row: int, num_times: int) -> tuple[int, int]:
    """Invert the flattened visibility row index into ``(satellite, time)``."""

    if row < 0:
        raise ValueError("row must be non-negative")
    if num_times <= 0:
        raise ValueError("num_times must be positive")
    return divmod(row, num_times)


def sparse_column_to_satellite_time(column: int, num_times: int) -> tuple[int, int]:
    """Backward-compatible alias for older site-row sparse layouts."""

    return sparse_row_to_satellite_time(column, num_times)


def save_visibility_npz(
    matrix: csr_matrix,
    output_path: str | Path,
    *,
    metadata: VisibilityMetadata | None = None,
    metadata_path: str | Path | None = None,
    compressed: bool = True,
) -> Path:
    """Persist a sparse visibility matrix and optional JSON metadata."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, matrix, compressed=compressed)

    if metadata is not None:
        meta_out = Path(metadata_path) if metadata_path is not None else out.with_suffix(".json")
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        meta_out.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8")

    return out


__all__ = [
    "GroundStationCandidate",
    "VisibilityMetadata",
    "build_visibility_and_range_csr",
    "build_visibility_csr",
    "build_visibility_csr_from_candidates",
    "candidates_to_arrays",
    "geodetic_to_ecef_km",
    "local_up_vectors",
    "save_visibility_npz",
    "sparse_column_to_satellite_time",
    "sparse_row_to_satellite_time",
]
