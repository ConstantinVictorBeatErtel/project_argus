"""Backhaul feasibility checks for candidate ground-station sites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

EARTH_RADIUS_KM = 6371.0088

PROXY_BACKHAUL_HUBS: tuple["BackhaulPoint", ...]


@dataclass(frozen=True)
class BackhaulPoint:
    """Internet backbone access point used for feasibility filtering."""

    point_id: str
    latitude_deg: float
    longitude_deg: float
    point_type: str = "unknown"


@dataclass(frozen=True)
class BackhaulMask:
    """Nearest-backhaul result for candidate sites."""

    feasible: NDArray[np.bool_]
    nearest_distance_km: NDArray[np.float64]
    nearest_point_index: NDArray[np.int64]


def _first_present(row: pd.Series, names: tuple[str, ...], *, default: Any = None) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def load_backhaul_points_csv(path: str | Path, *, limit: int | None = None) -> list[BackhaulPoint]:
    """Load IXP/fiber landing locations from a CSV file.

    Accepted columns:
        point id: ``point_id``, ``id``, ``name``
        latitude: ``latitude_deg``, ``lat``, ``latitude``
        longitude: ``longitude_deg``, ``lon``, ``lng``, ``longitude``
        type: ``point_type``, ``type``, ``kind``
    """

    point_path = Path(path)
    frame = pd.read_csv(point_path)
    if limit is not None:
        if limit <= 0:
            return []
        frame = frame.head(limit)

    points: list[BackhaulPoint] = []
    for row_idx, row in frame.iterrows():
        latitude = _first_present(row, ("latitude_deg", "lat", "latitude"))
        longitude = _first_present(row, ("longitude_deg", "lon", "lng", "longitude"))
        if latitude is None or longitude is None:
            raise ValueError(
                "Backhaul CSV must include latitude/longitude columns. "
                f"Failed at row {row_idx} in {point_path}."
            )

        point_id = _first_present(row, ("point_id", "id", "name"), default=f"backhaul_{row_idx:05d}")
        point_type = _first_present(row, ("point_type", "type", "kind"), default="unknown")
        points.append(
            BackhaulPoint(
                point_id=str(point_id),
                latitude_deg=float(latitude),
                longitude_deg=float(longitude),
                point_type=str(point_type),
            )
        )

    if not points:
        raise ValueError(f"No backhaul points loaded from {point_path}")
    return points


def proxy_backhaul_hubs() -> list[BackhaulPoint]:
    """Return approximate global internet/backbone hub locations.

    These are proxy points for development runs, not authoritative fiber or IXP
    locations. Replace with PeeringDB/fiber landing data for real analysis.
    """

    return list(PROXY_BACKHAUL_HUBS)


def haversine_distance_km(
    lat_a_deg: Sequence[float] | NDArray[np.floating],
    lon_a_deg: Sequence[float] | NDArray[np.floating],
    lat_b_deg: Sequence[float] | NDArray[np.floating],
    lon_b_deg: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Pairwise great-circle distances in kilometers via NumPy broadcasting."""

    lat_a = np.deg2rad(np.asarray(lat_a_deg, dtype=np.float64))
    lon_a = np.deg2rad(np.asarray(lon_a_deg, dtype=np.float64))
    lat_b = np.deg2rad(np.asarray(lat_b_deg, dtype=np.float64))
    lon_b = np.deg2rad(np.asarray(lon_b_deg, dtype=np.float64))

    dlat = lat_b[None, :] - lat_a[:, None]
    dlon = lon_b[None, :] - lon_a[:, None]
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    hav = sin_dlat * sin_dlat + np.cos(lat_a[:, None]) * np.cos(lat_b[None, :]) * sin_dlon * sin_dlon
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def compute_backhaul_mask(
    candidate_latitude_deg: Sequence[float] | NDArray[np.floating],
    candidate_longitude_deg: Sequence[float] | NDArray[np.floating],
    backhaul_latitude_deg: Sequence[float] | NDArray[np.floating],
    backhaul_longitude_deg: Sequence[float] | NDArray[np.floating],
    *,
    max_distance_km: float,
    candidate_chunk_size: int = 2048,
) -> BackhaulMask:
    """Compute ``b_i`` by nearest distance to any backbone access point."""

    if max_distance_km < 0.0:
        raise ValueError("max_distance_km must be nonnegative")
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")

    cand_lat = np.asarray(candidate_latitude_deg, dtype=np.float64)
    cand_lon = np.asarray(candidate_longitude_deg, dtype=np.float64)
    back_lat = np.asarray(backhaul_latitude_deg, dtype=np.float64)
    back_lon = np.asarray(backhaul_longitude_deg, dtype=np.float64)
    if cand_lat.shape != cand_lon.shape or cand_lat.ndim != 1:
        raise ValueError("candidate latitude/longitude must be matching one-dimensional arrays")
    if back_lat.shape != back_lon.shape or back_lat.ndim != 1:
        raise ValueError("backhaul latitude/longitude must be matching one-dimensional arrays")
    if cand_lat.size == 0:
        raise ValueError("At least one candidate site is required")
    if back_lat.size == 0:
        raise ValueError("At least one backhaul point is required")

    nearest_distance = np.empty(cand_lat.size, dtype=np.float64)
    nearest_index = np.empty(cand_lat.size, dtype=np.int64)

    for start in range(0, cand_lat.size, candidate_chunk_size):
        end = min(start + candidate_chunk_size, cand_lat.size)
        distances = haversine_distance_km(cand_lat[start:end], cand_lon[start:end], back_lat, back_lon)
        local_nearest = np.argmin(distances, axis=1)
        nearest_distance[start:end] = distances[np.arange(end - start), local_nearest]
        nearest_index[start:end] = local_nearest

    return BackhaulMask(
        feasible=nearest_distance <= max_distance_km,
        nearest_distance_km=nearest_distance,
        nearest_point_index=nearest_index,
    )


def compute_backhaul_mask_from_points(
    candidate_latitude_deg: Sequence[float] | NDArray[np.floating],
    candidate_longitude_deg: Sequence[float] | NDArray[np.floating],
    backhaul_points: Sequence[BackhaulPoint],
    *,
    max_distance_km: float,
    candidate_chunk_size: int = 2048,
) -> BackhaulMask:
    """Compute backhaul mask from loaded ``BackhaulPoint`` objects."""

    if not backhaul_points:
        raise ValueError("At least one backhaul point is required")
    back_lat = np.asarray([point.latitude_deg for point in backhaul_points], dtype=np.float64)
    back_lon = np.asarray([point.longitude_deg for point in backhaul_points], dtype=np.float64)
    return compute_backhaul_mask(
        candidate_latitude_deg,
        candidate_longitude_deg,
        back_lat,
        back_lon,
        max_distance_km=max_distance_km,
        candidate_chunk_size=candidate_chunk_size,
    )


def write_backhaul_mask_csv(
    output_path: str | Path,
    *,
    site_ids: Sequence[str],
    mask: BackhaulMask,
    backhaul_points: Sequence[BackhaulPoint] | None = None,
) -> Path:
    """Write backhaul feasibility results for candidate sites."""

    if len(site_ids) != mask.feasible.size:
        raise ValueError("site_ids length must match mask length")

    nearest_ids: list[str] | None = None
    nearest_types: list[str] | None = None
    if backhaul_points is not None:
        nearest_ids = [backhaul_points[int(idx)].point_id for idx in mask.nearest_point_index]
        nearest_types = [backhaul_points[int(idx)].point_type for idx in mask.nearest_point_index]

    frame = pd.DataFrame(
        {
            "site_id": list(site_ids),
            "b_i": mask.feasible.astype(np.uint8),
            "backhaul_feasible": mask.feasible,
            "nearest_backhaul_distance_km": mask.nearest_distance_km,
            "nearest_backhaul_index": mask.nearest_point_index,
        }
    )
    if nearest_ids is not None and nearest_types is not None:
        frame["nearest_backhaul_id"] = nearest_ids
        frame["nearest_backhaul_type"] = nearest_types

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def write_backhaul_points_csv(points: Sequence[BackhaulPoint], output_path: str | Path) -> Path:
    """Persist backhaul points to the common CSV schema."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": [point.point_id for point in points],
            "lat": [point.latitude_deg for point in points],
            "lon": [point.longitude_deg for point in points],
            "type": [point.point_type for point in points],
        }
    ).to_csv(out, index=False)
    return out


PROXY_BACKHAUL_HUBS = (
    BackhaulPoint("ashburn", 39.0438, -77.4874, "proxy_internet_hub"),
    BackhaulPoint("new_york", 40.7128, -74.0060, "proxy_internet_hub"),
    BackhaulPoint("los_angeles", 34.0522, -118.2437, "proxy_internet_hub"),
    BackhaulPoint("seattle", 47.6062, -122.3321, "proxy_internet_hub"),
    BackhaulPoint("dallas", 32.7767, -96.7970, "proxy_internet_hub"),
    BackhaulPoint("miami", 25.7617, -80.1918, "proxy_internet_hub"),
    BackhaulPoint("london", 51.5074, -0.1278, "proxy_internet_hub"),
    BackhaulPoint("amsterdam", 52.3676, 4.9041, "proxy_internet_hub"),
    BackhaulPoint("frankfurt", 50.1109, 8.6821, "proxy_internet_hub"),
    BackhaulPoint("paris", 48.8566, 2.3522, "proxy_internet_hub"),
    BackhaulPoint("stockholm", 59.3293, 18.0686, "proxy_internet_hub"),
    BackhaulPoint("madrid", 40.4168, -3.7038, "proxy_internet_hub"),
    BackhaulPoint("dubai", 25.2048, 55.2708, "proxy_internet_hub"),
    BackhaulPoint("mumbai", 19.0760, 72.8777, "proxy_internet_hub"),
    BackhaulPoint("singapore", 1.3521, 103.8198, "proxy_internet_hub"),
    BackhaulPoint("tokyo", 35.6762, 139.6503, "proxy_internet_hub"),
    BackhaulPoint("seoul", 37.5665, 126.9780, "proxy_internet_hub"),
    BackhaulPoint("sydney", -33.8688, 151.2093, "proxy_internet_hub"),
    BackhaulPoint("sao_paulo", -23.5505, -46.6333, "proxy_internet_hub"),
    BackhaulPoint("johannesburg", -26.2041, 28.0473, "proxy_internet_hub"),
)


__all__ = [
    "BackhaulMask",
    "BackhaulPoint",
    "compute_backhaul_mask",
    "compute_backhaul_mask_from_points",
    "haversine_distance_km",
    "load_backhaul_points_csv",
    "proxy_backhaul_hubs",
    "write_backhaul_mask_csv",
    "write_backhaul_points_csv",
]
