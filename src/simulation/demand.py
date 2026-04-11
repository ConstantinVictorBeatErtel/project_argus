"""Demand models for satellite-time relay requirements."""

from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PopulationRasterMetadata:
    """Regular lat/lon population raster metadata."""

    path: Path
    width: int
    height: int
    nodata: float | None
    x_origin_deg: float
    y_origin_deg: float
    x_resolution_deg: float
    y_resolution_deg: float
    crs: str | None


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


def load_population_raster_metadata(path: str | Path) -> PopulationRasterMetadata:
    """Load metadata for a regular EPSG:4326 population raster."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rasterio is required for raster demand. Install it with `pip install rasterio`.") from exc

    raster_path = Path(path)
    with rasterio.open(raster_path) as dataset:
        transform = dataset.transform
        if transform.b != 0.0 or transform.d != 0.0:
            raise ValueError("Only north-up rasters without rotation are supported")
        crs = None if dataset.crs is None else str(dataset.crs)
        return PopulationRasterMetadata(
            path=raster_path,
            width=int(dataset.width),
            height=int(dataset.height),
            nodata=None if dataset.nodata is None else float(dataset.nodata),
            x_origin_deg=float(transform.c),
            y_origin_deg=float(transform.f),
            x_resolution_deg=float(transform.a),
            y_resolution_deg=float(abs(transform.e)),
            crs=crs,
        )


def _wrap_longitude_deg(longitude_deg: float) -> float:
    wrapped = ((float(longitude_deg) + 180.0) % 360.0) - 180.0
    return 180.0 if np.isclose(wrapped, -180.0) else wrapped


def _adjust_longitudes_for_distance(
    longitude_deg: NDArray[np.float64],
    *,
    reference_longitude_deg: float,
) -> NDArray[np.float64]:
    adjusted = longitude_deg.astype(np.float64, copy=True)
    delta = adjusted - float(reference_longitude_deg)
    adjusted[delta > 180.0] -= 360.0
    adjusted[delta < -180.0] += 360.0
    return adjusted


def _raster_row_centers(metadata: PopulationRasterMetadata, rows: NDArray[np.int64]) -> NDArray[np.float64]:
    return metadata.y_origin_deg - metadata.y_resolution_deg * (rows.astype(np.float64) + 0.5)


def _raster_col_centers(metadata: PopulationRasterMetadata, cols: NDArray[np.int64]) -> NDArray[np.float64]:
    return metadata.x_origin_deg + metadata.x_resolution_deg * (cols.astype(np.float64) + 0.5)


def _split_wrapped_col_ranges(start_col: int, stop_col: int, *, width: int) -> list[tuple[int, int]]:
    if width <= 0:
        raise ValueError("width must be positive")
    if stop_col <= start_col:
        return []
    if stop_col - start_col >= width:
        return [(0, width)]

    ranges: list[tuple[int, int]] = []
    current = start_col
    while current < stop_col:
        wrapped = current % width
        length = min(stop_col - current, width - wrapped)
        ranges.append((wrapped, wrapped + length))
        current += length
    return ranges


def _read_population_window(
    dataset: Any,
    metadata: PopulationRasterMetadata,
    *,
    latitude_deg: float,
    longitude_deg: float,
    support_radius_km: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if support_radius_km <= 0.0:
        raise ValueError("support_radius_km must be positive")

    lat_radius_deg = support_radius_km / 111.32
    min_row = max(0, int(np.floor((metadata.y_origin_deg - (latitude_deg + lat_radius_deg)) / metadata.y_resolution_deg - 0.5)))
    max_row = min(
        metadata.height - 1,
        int(np.ceil((metadata.y_origin_deg - (latitude_deg - lat_radius_deg)) / metadata.y_resolution_deg - 0.5)),
    )
    if min_row > max_row:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )

    cos_lat = max(abs(np.cos(np.deg2rad(latitude_deg))), 1e-3)
    lon_radius_deg = min(180.0, support_radius_km / (111.32 * cos_lat))
    center_col = int(np.floor((_wrap_longitude_deg(longitude_deg) - metadata.x_origin_deg) / metadata.x_resolution_deg - 0.5))
    radius_cols = int(np.ceil(lon_radius_deg / metadata.x_resolution_deg))
    col_ranges = _split_wrapped_col_ranges(center_col - radius_cols, center_col + radius_cols + 1, width=metadata.width)
    if not col_ranges:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )

    row_slice = slice(min_row, max_row + 1)
    row_indices = np.arange(min_row, max_row + 1, dtype=np.int64)
    data_blocks: list[NDArray[np.float64]] = []
    col_index_blocks: list[NDArray[np.int64]] = []
    for start_col, stop_col in col_ranges:
        window_data = dataset.read(1, window=((row_slice.start, row_slice.stop), (start_col, stop_col)))
        data_blocks.append(np.asarray(window_data, dtype=np.float64))
        col_index_blocks.append(np.arange(start_col, stop_col, dtype=np.int64))

    data = np.concatenate(data_blocks, axis=1)
    col_indices = np.concatenate(col_index_blocks)
    latitudes = _raster_row_centers(metadata, row_indices)
    longitudes = _adjust_longitudes_for_distance(
        _raster_col_centers(metadata, col_indices),
        reference_longitude_deg=longitude_deg,
    )
    if metadata.nodata is not None:
        data = np.where(np.isclose(data, metadata.nodata), 0.0, data)
    data = np.where(np.isfinite(data) & (data > 0.0), data, 0.0)
    return latitudes, longitudes, data


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


def population_raster_weighted_demand_vector(
    positions_ecef_km: NDArray[np.floating],
    population_raster: str | Path | PopulationRasterMetadata,
    *,
    kernel_radius_km: float = 250.0,
    support_multiplier: float = 3.0,
    floor_weight: float = 0.05,
    population_exponent: float = 1.0,
    normalize_total: float | None = None,
) -> NDArray[np.float64]:
    """Build demand from a gridded population raster around satellite subpoints."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rasterio is required for raster demand. Install it with `pip install rasterio`.") from exc

    if kernel_radius_km <= 0.0:
        raise ValueError("kernel_radius_km must be positive")
    if support_multiplier <= 0.0:
        raise ValueError("support_multiplier must be positive")
    if floor_weight < 0.0:
        raise ValueError("floor_weight must be nonnegative")
    if population_exponent <= 0.0:
        raise ValueError("population_exponent must be positive")

    metadata = (
        population_raster
        if isinstance(population_raster, PopulationRasterMetadata)
        else load_population_raster_metadata(population_raster)
    )
    positions = np.asarray(positions_ecef_km, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions_ecef_km must have shape (num_satellites, num_times, 3)")

    latitudes, longitudes = ecef_to_geodetic_lat_lon(positions)
    row_latitudes = latitudes.reshape(-1)
    row_longitudes = longitudes.reshape(-1)
    demand = np.full(row_latitudes.shape, float(floor_weight), dtype=np.float64)
    support_radius_km = float(kernel_radius_km) * float(support_multiplier)

    with rasterio.open(metadata.path) as dataset:
        for row_index, (latitude_deg, longitude_deg) in enumerate(zip(row_latitudes, row_longitudes, strict=True)):
            window_latitudes, window_longitudes, window_population = _read_population_window(
                dataset,
                metadata,
                latitude_deg=float(latitude_deg),
                longitude_deg=float(longitude_deg),
                support_radius_km=support_radius_km,
            )
            if window_population.size == 0:
                continue

            lat_rad = np.radians(window_latitudes)[:, None]
            lon_rad = np.radians(_adjust_longitudes_for_distance(window_longitudes, reference_longitude_deg=float(longitude_deg)))[
                None, :
            ]
            row_lat_rad = np.deg2rad(float(latitude_deg))
            row_lon_rad = np.deg2rad(float(longitude_deg))
            dlat = lat_rad - row_lat_rad
            dlon = lon_rad - row_lon_rad
            hav = np.sin(dlat / 2.0) ** 2 + np.cos(row_lat_rad) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
            distances = 2.0 * EARTH_MEAN_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
            kernel = np.where(
                distances <= support_radius_km,
                np.exp(-0.5 * (distances / float(kernel_radius_km)) ** 2),
                0.0,
            )
            if not np.any(kernel):
                continue
            demand[row_index] += float(np.sum(kernel * np.power(window_population, float(population_exponent))))

    if normalize_total is not None:
        if normalize_total <= 0.0:
            raise ValueError("normalize_total must be positive when provided")
        total = float(demand.sum())
        if total <= 0.0:
            raise ValueError("Cannot normalize zero total demand")
        demand = demand * (float(normalize_total) / total)
    return demand.astype(np.float64)


def build_population_raster_demand_frame(
    positions_ecef_km: NDArray[np.floating],
    population_raster: str | Path | PopulationRasterMetadata,
    *,
    kernel_radius_km: float = 250.0,
    support_multiplier: float = 3.0,
    population_exponent: float = 1.0,
    floor_weight: float = 0.05,
    normalize_to_rows: bool = True,
) -> pd.DataFrame:
    """Build a raster-backed satellite-time demand frame."""

    positions = np.asarray(positions_ecef_km)
    if positions.ndim != 3:
        raise ValueError("positions_ecef_km must have shape (num_satellites, num_times, 3)")
    num_satellites, num_times = int(positions.shape[0]), int(positions.shape[1])
    normalize_total = float(num_satellites * num_times) if normalize_to_rows else None
    demand = population_raster_weighted_demand_vector(
        positions,
        population_raster,
        kernel_radius_km=kernel_radius_km,
        support_multiplier=support_multiplier,
        population_exponent=population_exponent,
        floor_weight=floor_weight,
        normalize_total=normalize_total,
    )
    return demand_frame_from_vector(
        demand,
        num_satellites=num_satellites,
        num_times=num_times,
        model="population_raster",
    )


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
    "build_population_raster_demand_frame",
    "build_uniform_demand_frame",
    "demand_frame_from_vector",
    "demand_vector_from_frame",
    "ecef_to_geodetic_lat_lon",
    "load_population_points_csv",
    "load_population_raster_metadata",
    "load_visibility_metadata",
    "population_raster_weighted_demand_vector",
    "population_weighted_demand_vector",
    "PopulationRasterMetadata",
    "satellite_time_rows",
    "uniform_demand_vector",
    "write_demand_outputs",
]
