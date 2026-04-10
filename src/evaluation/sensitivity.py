"""Scenario sensitivity utilities for ground-station placement experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from src.optimization.milp import solve_max_coverage_milp


@dataclass(frozen=True)
class SensitivityResult:
    """One max-coverage solve under a budget/elevation scenario."""

    elevation_deg: float
    max_ground_stations: int
    status: str
    achieved_coverage: float
    covered_demand: float
    total_demand: float
    selected_site_count: int
    assignment_count: int
    objective_value: float | None
    runtime_seconds: float
    visible_arc_count: int
    row_visibility_upper_bound: float
    demand_visibility_upper_bound: float
    selected_sites: tuple[int, ...]


def row_visibility_upper_bound(
    visibility: csr_matrix,
    *,
    site_feasible: Sequence[bool] | NDArray[np.bool_] | None = None,
) -> tuple[int, float]:
    """Return visible arc count and fraction of rows with at least one feasible arc."""

    vis = visibility.tocsr()
    if site_feasible is not None:
        feasible = np.asarray(site_feasible, dtype=np.bool_)
        if feasible.shape != (vis.shape[1],):
            raise ValueError(f"site_feasible must have shape ({vis.shape[1]},)")
        vis = vis[:, feasible]

    rows_with_visibility = np.asarray(vis.getnnz(axis=1) > 0, dtype=np.bool_)
    return int(vis.nnz), float(rows_with_visibility.mean()) if vis.shape[0] else 0.0


def visibility_upper_bounds(
    visibility: csr_matrix,
    *,
    demand: Sequence[float] | NDArray[np.floating] | None = None,
    site_feasible: Sequence[bool] | NDArray[np.bool_] | None = None,
) -> tuple[int, float, float]:
    """Return arc count, row bound, and demand-weighted visibility bound."""

    vis = visibility.tocsr()
    if site_feasible is not None:
        feasible = np.asarray(site_feasible, dtype=np.bool_)
        if feasible.shape != (vis.shape[1],):
            raise ValueError(f"site_feasible must have shape ({vis.shape[1]},)")
        vis = vis[:, feasible]

    rows_with_visibility = np.asarray(vis.getnnz(axis=1) > 0, dtype=np.bool_)
    if demand is None:
        weights = np.ones(vis.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(demand, dtype=np.float64)
        if weights.shape != (vis.shape[0],):
            raise ValueError(f"demand must have shape ({vis.shape[0]},)")
    total_demand = float(weights.sum())
    demand_bound = 0.0 if total_demand <= 0.0 else float(weights[rows_with_visibility].sum() / total_demand)
    row_bound = float(rows_with_visibility.mean()) if vis.shape[0] else 0.0
    return int(vis.nnz), row_bound, demand_bound


def solve_budget_sensitivity(
    visibility: csr_matrix,
    *,
    elevation_deg: float,
    budgets: Sequence[int],
    service_cost: csr_matrix | None = None,
    demand: Sequence[float] | NDArray[np.floating] | None = None,
    station_cost: float | Sequence[float] | NDArray[np.floating] = 1.0,
    site_feasible: Sequence[bool] | NDArray[np.bool_] | None = None,
    time_limit_seconds: int | None = None,
    mip_gap: float | None = None,
    msg: bool = False,
) -> list[SensitivityResult]:
    """Run max-coverage MILPs over a list of station budgets."""

    arc_count, row_bound, demand_bound = visibility_upper_bounds(
        visibility,
        demand=demand,
        site_feasible=site_feasible,
    )
    results: list[SensitivityResult] = []
    for budget in budgets:
        started = perf_counter()
        solution = solve_max_coverage_milp(
            visibility,
            service_cost=service_cost,
            demand=demand,
            station_cost=station_cost,
            max_ground_stations=int(budget),
            site_feasible=site_feasible,
            time_limit_seconds=time_limit_seconds,
            mip_gap=mip_gap,
            msg=msg,
        )
        results.append(
            SensitivityResult(
                elevation_deg=float(elevation_deg),
                max_ground_stations=int(budget),
                status=solution.status,
                achieved_coverage=solution.coverage_fraction,
                covered_demand=solution.covered_demand,
                total_demand=solution.total_demand,
                selected_site_count=len(solution.selected_sites),
                assignment_count=len(solution.assignments),
                objective_value=solution.objective_value,
                runtime_seconds=perf_counter() - started,
                visible_arc_count=arc_count,
                row_visibility_upper_bound=row_bound,
                demand_visibility_upper_bound=demand_bound,
                selected_sites=solution.selected_sites,
            )
        )
    return results


def sensitivity_results_to_frame(results: Sequence[SensitivityResult]) -> pd.DataFrame:
    """Convert sensitivity results to a CSV-friendly DataFrame."""

    rows = []
    for result in results:
        row = asdict(result)
        row["selected_sites"] = ",".join(str(site) for site in result.selected_sites)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "SensitivityResult",
    "row_visibility_upper_bound",
    "sensitivity_results_to_frame",
    "solve_budget_sensitivity",
    "visibility_upper_bounds",
]
