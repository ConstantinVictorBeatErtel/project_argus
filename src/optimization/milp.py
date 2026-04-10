"""Core MILP formulation for satellite ground-station placement.

The Phase 1 model solves a sparse maximum-coverage facility-location problem:

* ``y_i`` opens candidate site ``i``.
* ``x_ri`` assigns flattened satellite-time row ``r`` to an open visible site.
* each row can be assigned at most once.
* weighted coverage must exceed an epsilon-style requirement.
* the objective minimizes site fixed cost plus optional sparse service cost.

The expected visibility matrix layout is CSR with rows ``satellite_time`` and
columns ``site``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse

try:  # pragma: no cover - import itself is environment-dependent.
    import pulp
except ImportError:  # pragma: no cover
    pulp = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Assignment:
    """Selected assignment of one satellite-time row to one site."""

    satellite_time_row: int
    site_index: int
    demand_weight: float
    service_cost: float


@dataclass(frozen=True)
class OptimizationResult:
    """MILP solution summary."""

    status: str
    objective_value: float | None
    selected_sites: tuple[int, ...]
    coverage_fraction: float
    covered_demand: float
    total_demand: float
    assignments: tuple[Assignment, ...]


def _require_pulp() -> None:
    if pulp is None:
        raise ImportError("PuLP is required for the exact MILP solver. Install it with `pip install pulp`.")


def _as_csr(matrix: csr_matrix, *, name: str) -> csr_matrix:
    if not issparse(matrix):
        raise TypeError(f"{name} must be a SciPy sparse matrix")
    return matrix.tocsr()


def _demand_vector(demand: Sequence[float] | NDArray[np.floating] | None, num_rows: int) -> NDArray[np.float64]:
    if demand is None:
        return np.ones(num_rows, dtype=np.float64)
    weights = np.asarray(demand, dtype=np.float64)
    if weights.shape != (num_rows,):
        raise ValueError(f"demand must have shape ({num_rows},)")
    if np.any(weights < 0.0):
        raise ValueError("demand weights must be nonnegative")
    return weights


def _station_cost_vector(
    station_cost: float | Sequence[float] | NDArray[np.floating],
    num_sites: int,
) -> NDArray[np.float64]:
    if np.isscalar(station_cost):
        return np.full(num_sites, float(station_cost), dtype=np.float64)
    costs = np.asarray(station_cost, dtype=np.float64)
    if costs.shape != (num_sites,):
        raise ValueError(f"station_cost must be a scalar or have shape ({num_sites},)")
    if np.any(costs < 0.0):
        raise ValueError("station costs must be nonnegative")
    return costs


def _service_cost_lookup(service_cost: csr_matrix | None, visibility: csr_matrix) -> Mapping[tuple[int, int], float]:
    if service_cost is None:
        return {}
    costs = _as_csr(service_cost, name="service_cost")
    if costs.shape != visibility.shape:
        raise ValueError("service_cost must have the same shape as visibility")
    sparse_costs = costs.multiply(visibility).tocoo()
    return {
        (int(row), int(col)): float(value)
        for row, col, value in zip(sparse_costs.row, sparse_costs.col, sparse_costs.data, strict=True)
    }


def solve_ground_station_milp(
    visibility: csr_matrix,
    *,
    service_cost: csr_matrix | None = None,
    demand: Sequence[float] | NDArray[np.floating] | None = None,
    station_cost: float | Sequence[float] | NDArray[np.floating] = 1.0,
    coverage_requirement: float = 0.9,
    max_ground_stations: int | None = None,
    site_feasible: Sequence[bool] | NDArray[np.bool_] | None = None,
    time_limit_seconds: int | None = None,
    mip_gap: float | None = None,
    msg: bool = False,
) -> OptimizationResult:
    """Solve the sparse ground-station placement MILP.

    Args:
        visibility: CSR matrix with shape ``(num_satellite_time_rows, num_sites)``.
        service_cost: Optional sparse cost matrix over visible arcs. This is
            where latency costs such as ``alpha * distance / c`` enter.
        demand: Optional nonnegative weight per satellite-time row. Defaults to
            uniform demand.
        station_cost: Scalar or vector fixed cost for opening sites.
        coverage_requirement: Required fraction of total positive demand covered.
        max_ground_stations: Optional hard cardinality limit.
        site_feasible: Optional binary feasibility mask. Infeasible sites are
            forced closed via ``y_i <= b_i``.
        time_limit_seconds: Optional CBC solver time limit.
        mip_gap: Optional relative MIP gap for CBC.
        msg: Whether PuLP should emit solver logs.

    Returns:
        OptimizationResult with selected sites and assignments.
    """

    _require_pulp()
    if not 0.0 <= coverage_requirement <= 1.0:
        raise ValueError("coverage_requirement must be in [0, 1]")

    vis = _as_csr(visibility, name="visibility")
    vis.eliminate_zeros()
    num_rows, num_sites = vis.shape
    demand_weights = _demand_vector(demand, num_rows)
    fixed_costs = _station_cost_vector(station_cost, num_sites)
    total_demand = float(demand_weights.sum())
    if total_demand <= 0.0:
        raise ValueError("total demand must be positive")

    if max_ground_stations is not None and max_ground_stations < 0:
        raise ValueError("max_ground_stations must be nonnegative")

    if site_feasible is None:
        feasible = np.ones(num_sites, dtype=np.bool_)
    else:
        feasible = np.asarray(site_feasible, dtype=np.bool_)
        if feasible.shape != (num_sites,):
            raise ValueError(f"site_feasible must have shape ({num_sites},)")

    service_costs = _service_cost_lookup(service_cost, vis)

    problem = pulp.LpProblem("satellite_ground_station_placement", pulp.LpMinimize)
    y = {
        site: pulp.LpVariable(f"y_{site}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for site in range(num_sites)
    }

    x: dict[tuple[int, int], pulp.LpVariable] = {}
    row_exprs: dict[int, list[pulp.LpVariable]] = {}
    for row in range(num_rows):
        start = vis.indptr[row]
        end = vis.indptr[row + 1]
        if start == end:
            continue
        for site in vis.indices[start:end]:
            site_idx = int(site)
            if not feasible[site_idx]:
                continue
            variable = pulp.LpVariable(f"x_{row}_{site_idx}", lowBound=0, upBound=1, cat=pulp.LpBinary)
            x[(row, site_idx)] = variable
            row_exprs.setdefault(row, []).append(variable)
            problem += variable <= y[site_idx], f"assign_only_if_open_{row}_{site_idx}"

    for row, variables in row_exprs.items():
        problem += pulp.lpSum(variables) <= 1, f"assign_satellite_time_once_{row}"

    for site, is_feasible in enumerate(feasible):
        problem += y[site] <= int(is_feasible), f"backhaul_feasibility_{site}"

    if max_ground_stations is not None:
        problem += pulp.lpSum(y.values()) <= max_ground_stations, "max_ground_stations"

    covered_demand_expr = pulp.lpSum(demand_weights[row] * variable for (row, _), variable in x.items())
    problem += covered_demand_expr >= coverage_requirement * total_demand, "minimum_weighted_coverage"

    fixed_cost_expr = pulp.lpSum(fixed_costs[site] * variable for site, variable in y.items())
    service_cost_expr = pulp.lpSum(
        service_costs.get((row, site), 0.0) * variable
        for (row, site), variable in x.items()
    )
    problem += fixed_cost_expr + service_cost_expr

    solver_kwargs: dict[str, object] = {"msg": msg}
    if time_limit_seconds is not None:
        solver_kwargs["timeLimit"] = time_limit_seconds
    if mip_gap is not None:
        solver_kwargs["gapRel"] = mip_gap
    solver = pulp.PULP_CBC_CMD(**solver_kwargs)
    problem.solve(solver)

    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        return OptimizationResult(
            status=status,
            objective_value=None,
            selected_sites=(),
            coverage_fraction=0.0,
            covered_demand=0.0,
            total_demand=total_demand,
            assignments=(),
        )

    selected_sites = tuple(site for site, variable in y.items() if (variable.value() or 0.0) >= 0.5)
    assignments = tuple(
        Assignment(
            satellite_time_row=row,
            site_index=site,
            demand_weight=float(demand_weights[row]),
            service_cost=float(service_costs.get((row, site), 0.0)),
        )
        for (row, site), variable in sorted(x.items())
        if (variable.value() or 0.0) >= 0.5
    )
    covered_demand = float(sum(assignment.demand_weight for assignment in assignments))
    objective_value = None if problem.objective is None else float(pulp.value(problem.objective))

    return OptimizationResult(
        status=status,
        objective_value=objective_value,
        selected_sites=selected_sites,
        coverage_fraction=covered_demand / total_demand,
        covered_demand=covered_demand,
        total_demand=total_demand,
        assignments=assignments,
    )


def solve_max_coverage_milp(
    visibility: csr_matrix,
    *,
    service_cost: csr_matrix | None = None,
    demand: Sequence[float] | NDArray[np.floating] | None = None,
    station_cost: float | Sequence[float] | NDArray[np.floating] = 1.0,
    max_ground_stations: int | None = None,
    site_feasible: Sequence[bool] | NDArray[np.bool_] | None = None,
    cost_tiebreak_weight: float = 1e-6,
    time_limit_seconds: int | None = None,
    mip_gap: float | None = None,
    msg: bool = False,
) -> OptimizationResult:
    """Maximize covered demand for a fixed station budget.

    This is useful for scenario sensitivity: instead of asking whether a fixed
    coverage target is feasible, it asks what the best achievable coverage is
    under the current geometry, backhaul mask, and station-count budget.
    A tiny cost tie-breaker discourages opening unused stations among solutions
    with identical coverage.
    """

    _require_pulp()
    if cost_tiebreak_weight < 0.0:
        raise ValueError("cost_tiebreak_weight must be nonnegative")

    vis = _as_csr(visibility, name="visibility")
    vis.eliminate_zeros()
    num_rows, num_sites = vis.shape
    demand_weights = _demand_vector(demand, num_rows)
    fixed_costs = _station_cost_vector(station_cost, num_sites)
    total_demand = float(demand_weights.sum())
    if total_demand <= 0.0:
        raise ValueError("total demand must be positive")

    if max_ground_stations is not None and max_ground_stations < 0:
        raise ValueError("max_ground_stations must be nonnegative")

    if site_feasible is None:
        feasible = np.ones(num_sites, dtype=np.bool_)
    else:
        feasible = np.asarray(site_feasible, dtype=np.bool_)
        if feasible.shape != (num_sites,):
            raise ValueError(f"site_feasible must have shape ({num_sites},)")

    service_costs = _service_cost_lookup(service_cost, vis)

    problem = pulp.LpProblem("satellite_ground_station_max_coverage", pulp.LpMaximize)
    y = {
        site: pulp.LpVariable(f"y_{site}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for site in range(num_sites)
    }

    x: dict[tuple[int, int], pulp.LpVariable] = {}
    row_exprs: dict[int, list[pulp.LpVariable]] = {}
    for row in range(num_rows):
        start = vis.indptr[row]
        end = vis.indptr[row + 1]
        if start == end:
            continue
        for site in vis.indices[start:end]:
            site_idx = int(site)
            if not feasible[site_idx]:
                continue
            variable = pulp.LpVariable(f"x_{row}_{site_idx}", lowBound=0, upBound=1, cat=pulp.LpBinary)
            x[(row, site_idx)] = variable
            row_exprs.setdefault(row, []).append(variable)
            problem += variable <= y[site_idx], f"assign_only_if_open_{row}_{site_idx}"

    for row, variables in row_exprs.items():
        problem += pulp.lpSum(variables) <= 1, f"assign_satellite_time_once_{row}"

    for site, is_feasible in enumerate(feasible):
        problem += y[site] <= int(is_feasible), f"backhaul_feasibility_{site}"

    if max_ground_stations is not None:
        problem += pulp.lpSum(y.values()) <= max_ground_stations, "max_ground_stations"

    covered_demand_expr = pulp.lpSum(demand_weights[row] * variable for (row, _), variable in x.items())
    fixed_cost_expr = pulp.lpSum(fixed_costs[site] * variable for site, variable in y.items())
    service_cost_expr = pulp.lpSum(
        service_costs.get((row, site), 0.0) * variable
        for (row, site), variable in x.items()
    )
    problem += covered_demand_expr - cost_tiebreak_weight * (fixed_cost_expr + service_cost_expr)

    solver_kwargs: dict[str, object] = {"msg": msg}
    if time_limit_seconds is not None:
        solver_kwargs["timeLimit"] = time_limit_seconds
    if mip_gap is not None:
        solver_kwargs["gapRel"] = mip_gap
    solver = pulp.PULP_CBC_CMD(**solver_kwargs)
    problem.solve(solver)

    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        return OptimizationResult(
            status=status,
            objective_value=None,
            selected_sites=(),
            coverage_fraction=0.0,
            covered_demand=0.0,
            total_demand=total_demand,
            assignments=(),
        )

    selected_sites = tuple(site for site, variable in y.items() if (variable.value() or 0.0) >= 0.5)
    assignments = tuple(
        Assignment(
            satellite_time_row=row,
            site_index=site,
            demand_weight=float(demand_weights[row]),
            service_cost=float(service_costs.get((row, site), 0.0)),
        )
        for (row, site), variable in sorted(x.items())
        if (variable.value() or 0.0) >= 0.5
    )
    covered_demand = float(sum(assignment.demand_weight for assignment in assignments))
    objective_value = None if problem.objective is None else float(pulp.value(problem.objective))

    return OptimizationResult(
        status=status,
        objective_value=objective_value,
        selected_sites=selected_sites,
        coverage_fraction=covered_demand / total_demand,
        covered_demand=covered_demand,
        total_demand=total_demand,
        assignments=assignments,
    )


def propagation_latency_cost(range_km: csr_matrix, *, speed_of_light_km_s: float = 299792.458) -> csr_matrix:
    """Convert sparse slant range in kilometers to one-way propagation delay seconds."""

    if speed_of_light_km_s <= 0.0:
        raise ValueError("speed_of_light_km_s must be positive")
    ranges = _as_csr(range_km, name="range_km").astype(np.float64)
    costs = ranges.copy()
    costs.data = costs.data / speed_of_light_km_s
    return costs


__all__ = [
    "Assignment",
    "OptimizationResult",
    "propagation_latency_cost",
    "solve_max_coverage_milp",
    "solve_ground_station_milp",
]
