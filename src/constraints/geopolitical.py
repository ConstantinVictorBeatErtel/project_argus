"""Geopolitical site-eligibility masks for proxy ground-station candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExclusionZone:
    """Coarse lat/lon zone excluded from candidate siting."""

    name: str
    reason: str
    min_latitude_deg: float
    max_latitude_deg: float
    min_longitude_deg: float
    max_longitude_deg: float


DEFAULT_EXCLUSION_ZONES: tuple[ExclusionZone, ...] = (
    ExclusionZone("russia_belarus", "sanctions_and_operating_approval_risk", 50.0, 77.0, 20.0, 180.0),
    ExclusionZone("china", "market_access_and_licensing_risk", 18.0, 54.0, 73.0, 135.0),
    ExclusionZone("iran", "sanctions_and_operating_approval_risk", 24.0, 40.0, 44.0, 64.0),
    ExclusionZone("north_korea", "sanctions_and_operating_approval_risk", 37.0, 43.0, 124.0, 131.0),
    ExclusionZone("syria", "sanctions_and_operating_approval_risk", 32.0, 38.0, 35.0, 43.0),
    ExclusionZone("cuba", "sanctions_and_operating_approval_risk", 19.0, 24.0, -85.0, -74.0),
    ExclusionZone("venezuela", "sanctions_and_operating_approval_risk", 0.0, 13.0, -73.0, -59.0),
)


def assign_exclusion_zone(
    latitude_deg: Sequence[float] | NDArray[np.floating],
    longitude_deg: Sequence[float] | NDArray[np.floating],
    zones: Sequence[ExclusionZone] = DEFAULT_EXCLUSION_ZONES,
) -> list[str]:
    """Return the first matching exclusion-zone name for each coordinate."""

    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    if lat.shape != lon.shape:
        raise ValueError("latitude and longitude arrays must have matching shapes")

    labels = np.full(lat.shape, "", dtype=object)
    for zone in zones:
        mask = (
            (labels == "")
            & (lat >= zone.min_latitude_deg)
            & (lat <= zone.max_latitude_deg)
            & (lon >= zone.min_longitude_deg)
            & (lon <= zone.max_longitude_deg)
        )
        labels[mask] = zone.name
    return [str(value) for value in labels.tolist()]


def build_geopolitical_mask(
    candidates: pd.DataFrame,
    zones: Sequence[ExclusionZone] = DEFAULT_EXCLUSION_ZONES,
) -> pd.DataFrame:
    """Build a site-level geopolitical feasibility mask.

    This is intentionally an explicit policy scenario, not a statement that all
    sites in a coarse bounding box are impossible forever. The mask represents
    a conservative U.S.-operator siting assumption where major sanctions,
    licensing, and market-access risks are treated as hard exclusions.
    """

    required = {"site_id", "latitude_deg", "longitude_deg"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate frame missing required columns: {sorted(missing)}")

    labels = assign_exclusion_zone(
        candidates["latitude_deg"].to_numpy(dtype=float),
        candidates["longitude_deg"].to_numpy(dtype=float),
        zones=zones,
    )
    allowed = [label == "" for label in labels]
    reason_by_name = {zone.name: zone.reason for zone in zones}
    return pd.DataFrame(
        {
            "site_id": candidates["site_id"].astype(str),
            "latitude_deg": candidates["latitude_deg"].astype(float),
            "longitude_deg": candidates["longitude_deg"].astype(float),
            "g_i": [int(value) for value in allowed],
            "geopolitical_allowed": allowed,
            "excluded_zone": labels,
            "exclusion_reason": [reason_by_name.get(label, "") for label in labels],
        }
    )


__all__ = [
    "DEFAULT_EXCLUSION_ZONES",
    "ExclusionZone",
    "assign_exclusion_zone",
    "build_geopolitical_mask",
]
