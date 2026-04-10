"""Demand models for satellite-time relay requirements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)
WGS84_E2 = 1.0 - (WGS84_B_KM**2 / WGS84_A_KM**2)
WGS84_EP2 = (WGS84_A_KM**2 - WGS84_B_KM**2) / WGS84_B_KM**2
EARTH_MEAN_RADIUS_KM = 6371.0088


def satellite_time_rows(num_satellites: int, num_times: int) -> NDArray[np.int64]:
    """Return flattened row ids using ``row = satellite_index * num_times + time_index``."""

    if num_satellites <= 0:
        raise ValueError("num_satellites must be positive")
    if num_times <= 0:
        raise ValueError("num_times must be positive")
    return np.arange(num_satellites * num_times, dtype=np.int64)


def uniform_demand_vector(
    num_satellites: int,
    num_times: int,
    *,
    weight: float = 1.0,
    normalize: bool = False,
) -> NDArray[np.float64]:
    """Build a uniform demand vector over flattened satellite-time rows."""

    if weight < 0.0:
        raise ValueError("weight must be nonnegative")
    rows = satellite_time_rows(num_satellites, num_times)
    demand = np.full(rows.shape, float(weight), dtype=np.float64)
    if normalize:
        total = float(demand.sum())
        if total <= 0.0:
            raise ValueError("Cannot normalize zero total demand")
        demand = demand / total
    return demand


def demand_frame_from_vector(
    demand: NDArray[np.floating],
    *,
    num_satellites: int,
    num_times: int,
    model: str,
) -> pd.DataFrame:
    """Convert a flattened demand vector to a tabular artifact."""

    values = np.asarray(demand, dtype=np.float64)
    expected_rows = num_satellites * num_times
    if values.shape != (expected_rows,):
        raise ValueError(f"demand vector must have shape ({expected_rows},)")

    rows = satellite_time_rows(num_satellites, num_times)
    return pd.DataFrame(
        {
            "satellite_time_row": rows,
            "satellite_index": rows // num_times,
            "time_index": rows % num_times,
            "demand": values,
            "model": model,
        }
    )


def ecef_to_geodetic_lat_lon(
    ecef_km: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ECEF positions to WGS84 geodetic latitude and longitude.

    The returned arrays have the leading shape of ``ecef_km`` without the
    coordinate axis. Heights are not needed for demand modeling, so this helper
    returns only latitude and longitude in degrees.
    """

    positions = np.asarray(ecef_km, dtype=np.float64)
    if positions.shape[-1] != 3:
        raise ValueError("ecef_km must have a final coordinate axis of length 3")

    x = positions[..., 0]
    y = positions[..., 1]
    z = positions[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A_KM, p * WGS84_B_KM)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B_KM * sin_theta**3,
        p - WGS84_E2 * WGS84_A_KM * cos_theta**3,
    )
    return np.degrees(lat).astype(np.float64), np.degrees(lon).astype(np.float64)


def load_population_points_csv(
    path: str | Path,
    *,
    min_population: float = 0.0,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Load population proxy points from CSV.

    Required columns are ``latitude_deg``, ``longitude_deg``, and
    ``population``. Extra source columns are preserved.
    """

    frame = pd.read_csv(path)
    required = {"latitude_deg", "longitude_deg", "population"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"population points missing required columns: {sorted(missing)}")

    points = frame.copy()
    points["population"] = pd.to_numeric(points["population"], errors="coerce").fillna(0.0)
    points = points[points["population"] >= float(min_population)]
    points = points[np.isfinite(points["latitude_deg"]) & np.isfinite(points["longitude_deg"])]
    points = points[points["population"] > 0.0]
    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n must be positive when provided")
        points = points.sort_values("population", ascending=False).head(top_n)
    if points.empty:
        raise ValueError("No population points remain after filtering")
    return points.reset_index(drop=True)


def _haversine_distance_matrix_km(
    row_latitude_deg: NDArray[np.float64],
    row_longitude_deg: NDArray[np.float64],
    point_latitude_deg: NDArray[np.float64],
    point_longitude_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    row_lat = np.radians(row_latitude_deg)[:, None]
    row_lon = np.radians(row_longitude_deg)[:, None]
    point_lat = np.radians(point_latitude_deg)[None, :]
    point_lon = np.radians(point_longitude_deg)[None, :]
    dlat = point_lat - row_lat
    dlon = point_lon - row_lon
    a = np.sin(dlat / 2.0) ** 2 + np.cos(row_lat) * np.cos(point_lat) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_MEAN_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def population_weighted_demand_vector(
    positions_ecef_km: NDArray[np.floating],
    population_points: pd.DataFrame,
    *,
    kernel_radius_km: float = 500.0,
    population_exponent: float = 0.75,
    floor_weight: float = 0.05,
    normalize_total: float | None = None,
    row_chunk_size: int = 1024,
) -> NDArray[np.float64]:
    """Build demand from a Gaussian population kernel around ground tracks.

    Args:
        positions_ecef_km: Satellite ECEF tensor with shape
            ``(num_satellites, num_times, 3)``.
        population_points: DataFrame with ``latitude_deg``, ``longitude_deg``,
            and ``population`` columns.
        kernel_radius_km: Gaussian kernel radius around each satellite subpoint.
        population_exponent: Dampens the dominance of megacities while keeping
            dense areas more important than sparse areas.
        floor_weight: Baseline demand added to every row before normalization.
        normalize_total: Optional total demand after weighting. Using
            ``num_satellites * num_times`` makes comparisons to the uniform
            baseline direct.
        row_chunk_size: Number of satellite-time rows per vectorized distance
            block.
    """

    if kernel_radius_km <= 0.0:
        raise ValueError("kernel_radius_km must be positive")
    if population_exponent <= 0.0:
        raise ValueError("population_exponent must be positive")
    if floor_weight < 0.0:
        raise ValueError("floor_weight must be nonnegative")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")

    positions = np.asarray(positions_ecef_km, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions_ecef_km must have shape (num_satellites, num_times, 3)")

    latitudes, longitudes = ecef_to_geodetic_lat_lon(positions)
    row_latitudes = latitudes.reshape(-1)
    row_longitudes = longitudes.reshape(-1)
    point_latitudes = population_points["latitude_deg"].to_numpy(dtype=np.float64)
    point_longitudes = population_points["longitude_deg"].to_numpy(dtype=np.float64)
    point_weights = population_points["population"].to_numpy(dtype=np.float64) ** float(population_exponent)

    demand = np.full(row_latitudes.shape, float(floor_weight), dtype=np.float64)
    for start in range(0, row_latitudes.size, row_chunk_size):
        end = min(start + row_chunk_size, row_latitudes.size)
        distances = _haversine_distance_matrix_km(
            row_latitudes[start:end],
            row_longitudes[start:end],
            point_latitudes,
            point_longitudes,
        )
        scaled_distances = distances / float(kernel_radius_km)
        kernel = np.where(
            scaled_distances <= 4.0,
            np.exp(-0.5 * scaled_distances**2),
            0.0,
        )
        demand[start:end] += np.sum(kernel * point_weights[None, :], axis=1)

    if normalize_total is not None:
        if normalize_total <= 0.0:
            raise ValueError("normalize_total must be positive when provided")
        total = float(demand.sum())
        if total <= 0.0:
            raise ValueError("Cannot normalize zero total demand")
        demand = demand * (float(normalize_total) / total)
    return demand.astype(np.float64)


def build_population_weighted_demand_frame(
    positions_ecef_km: NDArray[np.floating],
    population_points: pd.DataFrame,
    *,
    kernel_radius_km: float = 500.0,
    population_exponent: float = 0.75,
    floor_weight: float = 0.05,
    normalize_to_rows: bool = True,
    row_chunk_size: int = 1024,
) -> pd.DataFrame:
    """Build a population-weighted satellite-time demand frame."""

    positions = np.asarray(positions_ecef_km)
    if positions.ndim != 3:
        raise ValueError("positions_ecef_km must have shape (num_satellites, num_times, 3)")
    num_satellites, num_times = int(positions.shape[0]), int(positions.shape[1])
    normalize_total = float(num_satellites * num_times) if normalize_to_rows else None
    demand = population_weighted_demand_vector(
        positions,
        population_points,
        kernel_radius_km=kernel_radius_km,
        population_exponent=population_exponent,
        floor_weight=floor_weight,
        normalize_total=normalize_total,
        row_chunk_size=row_chunk_size,
    )
    return demand_frame_from_vector(
        demand,
        num_satellites=num_satellites,
        num_times=num_times,
        model="population_proxy",
    )


def build_uniform_demand_frame(
    num_satellites: int,
    num_times: int,
    *,
    weight: float = 1.0,
    normalize: bool = False,
) -> pd.DataFrame:
    """Build a uniform-demand DataFrame for Phase 1 baseline comparisons."""

    demand = uniform_demand_vector(num_satellites, num_times, weight=weight, normalize=normalize)
    model = "uniform_normalized" if normalize else "uniform"
    return demand_frame_from_vector(demand, num_satellites=num_satellites, num_times=num_times, model=model)


def demand_vector_from_frame(frame: pd.DataFrame) -> NDArray[np.float64]:
    """Extract a solver-ready demand vector ordered by ``satellite_time_row``."""

    required_columns = {"satellite_time_row", "demand"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"demand frame missing required columns: {sorted(missing)}")
    ordered = frame.sort_values("satellite_time_row")
    rows = ordered["satellite_time_row"].to_numpy(dtype=np.int64)
    expected = np.arange(rows.size, dtype=np.int64)
    if not np.array_equal(rows, expected):
        raise ValueError("demand rows must be contiguous from 0 to n-1")
    return ordered["demand"].to_numpy(dtype=np.float64)


def load_visibility_metadata(path: str | Path) -> tuple[int, int]:
    """Read ``(num_satellites, num_times)`` from visibility metadata JSON."""

    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return int(payload["num_satellites"]), int(payload["num_times"])


def write_demand_outputs(
    frame: pd.DataFrame,
    *,
    parquet_path: str | Path,
    npy_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Persist demand as Parquet and optionally as a solver-ready NumPy vector."""

    parquet_out = Path(parquet_path)
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_out, index=False)

    npy_out: Path | None = None
    if npy_path is not None:
        npy_out = Path(npy_path)
        npy_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_out, demand_vector_from_frame(frame))

    return parquet_out, npy_out


__all__ = [
    "build_population_weighted_demand_frame",
    "build_uniform_demand_frame",
    "demand_frame_from_vector",
    "demand_vector_from_frame",
    "ecef_to_geodetic_lat_lon",
    "load_population_points_csv",
    "load_visibility_metadata",
    "population_weighted_demand_vector",
    "satellite_time_rows",
    "uniform_demand_vector",
    "write_demand_outputs",
]
