"""Alternative site-selection heuristics for the 200-site proxy network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


def _haversine_distance_matrix(candidate_coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return pairwise great-circle distances in kilometers."""

    coords = np.asarray(candidate_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("candidate_coords must have shape (n_sites, 2) for latitude/longitude pairs")

    lat = np.deg2rad(coords[:, 0])[:, None]
    lon = np.deg2rad(coords[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * (2.0 * np.arcsin(np.clip(np.sqrt(hav), 0.0, 1.0)))


def geographic_spread_selection(candidate_coords: NDArray[np.float64], k: int = 20) -> list[int]:
    """Greedy farthest-point sampling on latitude/longitude coordinates."""

    coords = np.asarray(candidate_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("candidate_coords must have shape (n_sites, 2)")
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= coords.shape[0]:
        return list(range(coords.shape[0]))

    distances = _haversine_distance_matrix(coords)
    seed = int(np.lexsort((coords[:, 1], coords[:, 0]))[0])
    selected = [seed]
    min_distance = distances[seed].copy()
    min_distance[seed] = -np.inf

    while len(selected) < k:
        next_idx = int(np.argmax(min_distance))
        selected.append(next_idx)
        min_distance = np.minimum(min_distance, distances[next_idx])
        min_distance[selected] = -np.inf

    return selected


def greedy_max_arcs_selection(visibility_matrix: sparse.spmatrix, k: int = 20) -> list[int]:
    """Select the ``k`` sites with the highest visible-row counts."""

    if k <= 0:
        raise ValueError("k must be positive")
    matrix = visibility_matrix.tocsc()
    column_sums = np.asarray(matrix.getnnz(axis=0)).ravel()
    order = np.argsort(-column_sums, kind="stable")
    return [int(idx) for idx in order[: min(k, matrix.shape[1])]]


def coverage_fraction(
    visibility_matrix: sparse.spmatrix,
    selected_sites: list[int] | NDArray[np.int64],
    demand_weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute covered-row fraction for a selected site portfolio."""

    matrix = visibility_matrix.tocsr()
    selected = np.asarray(selected_sites, dtype=np.int64)
    if selected.size == 0:
        return 0.0

    covered_rows = np.asarray(matrix[:, selected].getnnz(axis=1)).ravel() > 0
    if demand_weights is None:
        return float(np.mean(covered_rows))

    weights = np.asarray(demand_weights, dtype=np.float64).ravel()
    if weights.shape[0] != matrix.shape[0]:
        raise ValueError("demand_weights length must match the number of visibility rows")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("demand_weights must sum to a positive value")
    return float(weights[covered_rows].sum() / total)


def load_candidate_coords(metadata_path: str | Path) -> NDArray[np.float64]:
    """Load candidate latitude/longitude pairs from a visibility metadata file."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return np.asarray(
        [
            [float(site["latitude_deg"]), float(site["longitude_deg"])]
            for site in metadata["candidate_sites"]
        ],
        dtype=np.float64,
    )


__all__ = [
    "coverage_fraction",
    "geographic_spread_selection",
    "greedy_max_arcs_selection",
    "load_candidate_coords",
]
