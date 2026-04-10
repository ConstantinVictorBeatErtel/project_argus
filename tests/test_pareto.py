"""Pareto sweep utilities."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.optimization.pareto import coverage_grid, pareto_points_to_frame, solve_pareto_sweep


def test_coverage_grid_is_inclusive() -> None:
    grid = coverage_grid(0.1, 0.3, 0.1)

    assert np.allclose(grid, [0.1, 0.2, 0.3])


def test_solve_pareto_sweep_records_statuses() -> None:
    visibility = csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.uint8))
    points = solve_pareto_sweep(
        visibility,
        coverage_targets=[0.5, 1.0],
        max_ground_stations=1,
        time_limit_seconds=30,
        msg=False,
    )
    frame = pareto_points_to_frame(points)

    assert frame["coverage_target"].tolist() == [0.5, 1.0]
    assert frame["status"].tolist() == ["Optimal", "Infeasible"]
    assert frame.loc[0, "selected_site_count"] == 1
