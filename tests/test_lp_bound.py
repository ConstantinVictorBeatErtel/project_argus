"""Small LP/MIP regression placeholder for Phase 1."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.optimization.milp import solve_ground_station_milp


def test_small_exact_milp_reaches_zero_gap_optimum() -> None:
    visibility = csr_matrix(np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8))
    result = solve_ground_station_milp(
        visibility,
        coverage_requirement=1.0,
        station_cost=[2.0, 3.0],
        mip_gap=0.0,
        time_limit_seconds=30,
        msg=False,
    )

    assert result.status == "Optimal"
    assert result.objective_value == 5.0
    assert result.coverage_fraction == 1.0
