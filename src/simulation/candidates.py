"""Generated candidate-site inputs for dataset-free Phase 1 runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class BoundingBox:
    """Latitude/longitude rectangle used as a rough land proxy."""

    name: str
    min_latitude_deg: float
    max_latitude_deg: float
    min_longitude_deg: float
    max_longitude_deg: float


ROUGH_LAND_BOXES: tuple[BoundingBox, ...] = (
    BoundingBox("north_america", 15.0, 72.0, -170.0, -50.0),
    BoundingBox("south_america", -56.0, 13.0, -82.0, -34.0),
    BoundingBox("europe", 35.0, 72.0, -12.0, 45.0),
    BoundingBox("africa", -35.0, 37.0, -18.0, 52.0),
    BoundingBox("asia", 5.0, 77.0, 45.0, 180.0),
    BoundingBox("australia", -45.0, -10.0, 110.0, 155.0),
)


def points_in_boxes(
    latitude_deg: NDArray[np.floating],
    longitude_deg: NDArray[np.floating],
    boxes: Sequence[BoundingBox] = ROUGH_LAND_BOXES,
) -> NDArray[np.bool_]:
    """Return a mask for points inside any rough land bounding box."""

    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    if lat.shape != lon.shape:
        raise ValueError("latitude and longitude arrays must have matching shapes")

    mask = np.zeros(lat.shape, dtype=np.bool_)
    for box in boxes:
        mask |= (
            (lat >= box.min_latitude_deg)
            & (lat <= box.max_latitude_deg)
            & (lon >= box.min_longitude_deg)
            & (lon <= box.max_longitude_deg)
        )
    return mask


def _latlon_unit_vectors(latitude_deg: NDArray[np.floating], longitude_deg: NDArray[np.floating]) -> NDArray[np.float64]:
    lat = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def farthest_point_sample_latlon(
    latitude_deg: NDArray[np.floating],
    longitude_deg: NDArray[np.floating],
    *,
    max_sites: int,
) -> NDArray[np.int64]:
    """Select geographically diverse lat/lon points with deterministic farthest-point sampling."""

    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    if lat.shape != lon.shape or lat.ndim != 1:
        raise ValueError("latitude/longitude must be matching one-dimensional arrays")
    if max_sites <= 0:
        raise ValueError("max_sites must be positive")
    if lat.size <= max_sites:
        return np.arange(lat.size, dtype=np.int64)

    vectors = _latlon_unit_vectors(lat, lon)
    selected = np.empty(max_sites, dtype=np.int64)
    selected[0] = int(np.argmin(lat))
    min_chord_distance = np.full(lat.size, np.inf, dtype=np.float64)

    for out_idx in range(1, max_sites):
        previous = vectors[selected[out_idx - 1]]
        distance = np.sum((vectors - previous) ** 2, axis=1)
        min_chord_distance = np.minimum(min_chord_distance, distance)
        selected[out_idx] = int(np.argmax(min_chord_distance))

    return selected


def generate_candidate_grid(
    *,
    latitude_step_deg: float = 10.0,
    longitude_step_deg: float = 10.0,
    min_latitude_deg: float = -60.0,
    max_latitude_deg: float = 75.0,
    min_longitude_deg: float = -180.0,
    max_longitude_deg: float = 180.0,
    rough_land_only: bool = True,
    max_sites: int | None = None,
    downsample_method: str = "farthest",
) -> pd.DataFrame:
    """Generate a deterministic coarse candidate grid.

    This is a proxy input generator, not a real site inventory. It exists so the
    optimizer can run without external candidate datasets.
    """

    if latitude_step_deg <= 0.0 or longitude_step_deg <= 0.0:
        raise ValueError("grid steps must be positive")
    if min_latitude_deg > max_latitude_deg:
        raise ValueError("min_latitude_deg cannot exceed max_latitude_deg")
    if min_longitude_deg > max_longitude_deg:
        raise ValueError("min_longitude_deg cannot exceed max_longitude_deg")

    latitudes = np.arange(min_latitude_deg, max_latitude_deg + 0.5 * latitude_step_deg, latitude_step_deg)
    longitudes = np.arange(min_longitude_deg, max_longitude_deg + 0.5 * longitude_step_deg, longitude_step_deg)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    flat_lat = lat_grid.ravel()
    flat_lon = lon_grid.ravel()

    if rough_land_only:
        mask = points_in_boxes(flat_lat, flat_lon)
        flat_lat = flat_lat[mask]
        flat_lon = flat_lon[mask]

    if max_sites is not None:
        if max_sites <= 0:
            raise ValueError("max_sites must be positive when provided")
        if flat_lat.size > max_sites:
            if downsample_method == "farthest":
                selected = farthest_point_sample_latlon(flat_lat, flat_lon, max_sites=max_sites)
            elif downsample_method == "linspace":
                selected = np.linspace(0, flat_lat.size - 1, max_sites, dtype=np.int64)
            else:
                raise ValueError("downsample_method must be 'farthest' or 'linspace'")
            flat_lat = flat_lat[selected]
            flat_lon = flat_lon[selected]

    site_ids = [f"grid-{idx:05d}" for idx in range(flat_lat.size)]
    return pd.DataFrame(
        {
            "site_id": site_ids,
            "latitude_deg": flat_lat.astype(float),
            "longitude_deg": flat_lon.astype(float),
            "altitude_m": np.zeros(flat_lat.size, dtype=float),
            "candidate_source": "rough_land_grid_proxy" if rough_land_only else "global_grid_proxy",
            "downsample_method": downsample_method,
        }
    )


def write_candidate_grid_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist generated candidate sites."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


__all__ = [
    "BoundingBox",
    "ROUGH_LAND_BOXES",
    "farthest_point_sample_latlon",
    "generate_candidate_grid",
    "points_in_boxes",
    "write_candidate_grid_csv",
]
