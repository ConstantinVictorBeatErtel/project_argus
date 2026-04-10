"""Sensitivity analysis utilities."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.evaluation.sensitivity import (
    row_visibility_upper_bound,
    sensitivity_results_to_frame,
    solve_budget_sensitivity,
    visibility_upper_bounds,
)
from src.optimization.milp import solve_max_coverage_milp


def test_solve_max_coverage_milp_respects_station_budget() -> None:
    visibility = csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
            ],
            dtype=np.uint8,
        )
    )

    result = solve_max_coverage_milp(visibility, max_ground_stations=1, time_limit_seconds=30, msg=False)

    assert result.status == "Optimal"
    assert result.selected_sites == (0,)
    assert result.coverage_fraction == 2 / 3


def test_row_visibility_upper_bound_applies_site_mask() -> None:
    visibility = csr_matrix(np.array([[1, 0], [0, 1], [0, 0]], dtype=np.uint8))

    arc_count, bound = row_visibility_upper_bound(visibility, site_feasible=[True, False])

    assert arc_count == 1
    assert bound == 1 / 3


def test_visibility_upper_bounds_include_demand_weighting() -> None:
    visibility = csr_matrix(np.array([[1, 0], [0, 0], [0, 1]], dtype=np.uint8))

    arc_count, row_bound, demand_bound = visibility_upper_bounds(
        visibility,
        demand=[10.0, 100.0, 10.0],
        site_feasible=[True, False],
    )

    assert arc_count == 1
    assert row_bound == 1 / 3
    assert demand_bound == 10.0 / 120.0


def test_solve_budget_sensitivity_returns_csv_ready_rows() -> None:
    visibility = csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.uint8))
    results = solve_budget_sensitivity(
        visibility,
        elevation_deg=10.0,
        budgets=[1, 2],
        time_limit_seconds=30,
        msg=False,
    )
    frame = sensitivity_results_to_frame(results)

    assert frame["max_ground_stations"].tolist() == [1, 2]
    assert frame["status"].tolist() == ["Optimal", "Optimal"]
    assert frame["achieved_coverage"].tolist() == [0.5, 1.0]
    assert frame["demand_visibility_upper_bound"].tolist() == [1.0, 1.0]
