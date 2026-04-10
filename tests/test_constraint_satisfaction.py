"""MILP solution feasibility checks."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix

from scripts.run_optimization import load_site_feasible
from src.optimization.milp import solve_ground_station_milp


def test_milp_assigns_only_visible_arcs_and_open_feasible_sites() -> None:
    visibility = csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=np.uint8,
        )
    )

    result = solve_ground_station_milp(
        visibility,
        coverage_requirement=1.0,
        station_cost=[1.0, 1.0],
        time_limit_seconds=30,
        msg=False,
    )

    assert result.status == "Optimal"
    assert result.coverage_fraction == 1.0
    assert set(result.selected_sites) == {0, 1}

    for assignment in result.assignments:
        assert visibility[assignment.satellite_time_row, assignment.site_index] == 1
        assert assignment.site_index in result.selected_sites


def test_milp_respects_backhaul_feasibility_mask() -> None:
    visibility = csr_matrix(
        np.array(
            [
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=np.uint8,
        )
    )

    result = solve_ground_station_milp(
        visibility,
        coverage_requirement=0.5,
        station_cost=[1.0, 1.0],
        site_feasible=[True, False],
        time_limit_seconds=30,
        msg=False,
    )

    assert result.status == "Optimal"
    assert result.selected_sites == (0,)
    assert result.coverage_fraction >= 0.5
    assert all(assignment.site_index == 0 for assignment in result.assignments)


def test_milp_does_not_report_stale_values_for_infeasible_solution() -> None:
    visibility = csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.uint8))

    result = solve_ground_station_milp(
        visibility,
        coverage_requirement=1.0,
        max_ground_stations=1,
        time_limit_seconds=30,
        msg=False,
    )

    assert result.status == "Infeasible"
    assert result.selected_sites == ()
    assert result.assignments == ()
    assert result.objective_value is None


def test_load_site_feasible_aligns_mask_by_metadata_site_ids() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mask_csv = root / "mask.csv"
        metadata_json = root / "metadata.json"
        mask_csv.write_text("site_id,b_i\nsite-b,0\nsite-a,1\n", encoding="utf-8")
        metadata_json.write_text('{"site_ids": ["site-a", "site-b"]}', encoding="utf-8")

        feasible = load_site_feasible(mask_csv, metadata_json=metadata_json)

    assert feasible.tolist() == [True, False]
