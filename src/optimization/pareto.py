"""Epsilon-constraint Pareto frontier utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from src.optimization.milp import OptimizationResult, solve_ground_station_milp


@dataclass(frozen=True)
class ParetoPoint:
    """One epsilon-constraint solve on the coverage frontier."""

    coverage_target: float
    status: str
    achieved_coverage: float
    objective_value: float | None
    selected_site_count: int
    assignment_count: int
    covered_demand: float
    total_demand: float
    runtime_seconds: float
    selected_sites: tuple[int, ...]


def coverage_grid(start: float, stop: float, step: float) -> NDArray[np.float64]:
    """Build an inclusive floating coverage grid."""

    if step <= 0.0:
        raise ValueError("step must be positive")
    if start < 0.0 or stop > 1.0 or start > stop:
        raise ValueError("coverage range must satisfy 0 <= start <= stop <= 1")
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    if values[-1] < stop - 1e-12:
        values = np.append(values, stop)
    else:
        values[-1] = min(values[-1], stop)
    return values


def solve_pareto_sweep(
    visibility: csr_matrix,
    *,
    coverage_targets: Sequence[float],
    service_cost: csr_matrix | None = None,
    demand: Sequence[float] | NDArray[np.floating] | None = None,
    station_cost: float | Sequence[float] | NDArray[np.floating] = 1.0,
    max_ground_stations: int | None = None,
    site_feasible: Sequence[bool] | NDArray[np.bool_] | None = None,
    time_limit_seconds: int | None = None,
    mip_gap: float | None = None,
    msg: bool = False,
) -> list[ParetoPoint]:
    """Run epsilon-constraint solves over coverage targets."""

    points: list[ParetoPoint] = []
    for target in coverage_targets:
        started = perf_counter()
        result: OptimizationResult = solve_ground_station_milp(
            visibility,
            service_cost=service_cost,
            demand=demand,
            station_cost=station_cost,
            coverage_requirement=float(target),
            max_ground_stations=max_ground_stations,
            site_feasible=site_feasible,
            time_limit_seconds=time_limit_seconds,
            mip_gap=mip_gap,
            msg=msg,
        )
        runtime = perf_counter() - started
        points.append(
            ParetoPoint(
                coverage_target=float(target),
                status=result.status,
                achieved_coverage=result.coverage_fraction,
                objective_value=result.objective_value,
                selected_site_count=len(result.selected_sites),
                assignment_count=len(result.assignments),
                covered_demand=result.covered_demand,
                total_demand=result.total_demand,
                runtime_seconds=runtime,
                selected_sites=result.selected_sites,
            )
        )
    return points


def pareto_points_to_frame(points: Sequence[ParetoPoint]) -> pd.DataFrame:
    """Convert Pareto points to a CSV-friendly DataFrame."""

    rows = []
    for point in points:
        row = asdict(point)
        row["selected_sites"] = ",".join(str(site) for site in point.selected_sites)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["ParetoPoint", "coverage_grid", "pareto_points_to_frame", "solve_pareto_sweep"]
