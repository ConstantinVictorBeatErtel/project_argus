"""Visibility geometry and sparse-layout tests."""

from __future__ import annotations

import numpy as np

from src.simulation.visibility import (
    GroundStationCandidate,
    build_visibility_and_range_csr,
    build_visibility_csr_from_candidates,
    geodetic_to_ecef_km,
    sparse_row_to_satellite_time,
)


def test_visibility_uses_satellite_time_rows_and_site_columns() -> None:
    site = geodetic_to_ecef_km([0.0], [0.0], [0.0])[0]
    satellite_ecef_km = np.array(
        [
            [
                site + np.array([500.0, 0.0, 0.0]),
                site + np.array([600.0, 0.0, 0.0]),
            ],
            [
                -site,
                -site,
            ],
        ],
        dtype=np.float32,
    )

    visibility, ranges = build_visibility_and_range_csr(
        satellite_ecef_km,
        [0.0],
        [0.0],
        min_elevation_deg=10.0,
        site_chunk_size=1,
        time_chunk_size=1,
    )

    assert visibility.shape == (4, 1)
    assert visibility.nnz == 2
    assert visibility.nonzero()[0].tolist() == [0, 1]
    assert visibility.nonzero()[1].tolist() == [0, 0]
    assert [sparse_row_to_satellite_time(int(row), num_times=2) for row in visibility.nonzero()[0]] == [(0, 0), (0, 1)]
    assert ranges.data.round(1).tolist() == [500.0, 600.0]


def test_infeasible_candidate_column_is_zeroed() -> None:
    site = geodetic_to_ecef_km([0.0], [0.0], [0.0])[0]
    satellite_ecef_km = np.array([[[*(site + np.array([500.0, 0.0, 0.0]))]]], dtype=np.float32)

    visibility = build_visibility_csr_from_candidates(
        satellite_ecef_km,
        [GroundStationCandidate("site-0", 0.0, 0.0, backhaul_feasible=False)],
    )

    assert visibility.shape == (1, 1)
    assert visibility.nnz == 0
