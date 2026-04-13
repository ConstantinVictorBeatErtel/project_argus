"""Demand-misspecification cross-evaluation for two fixed 20-station portfolios.

Evaluates each portfolio (uniform-optimal and raster-optimal) under both demand
models (uniform and raster-weighted) without re-running the MILP.  Coverage is
defined as:

    sum(demand_weight[r] for r in rows where any open site is visible)
    -------------------------------------------------------------------
    sum(demand_weight[r] for all rows r)

Results are written to data/processed/demand_misspecification_crosseval.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

VIS_NPZ = DATA / "visibility_200.npz"
RASTER_DEMAND_NPY = DATA / "demand_population_raster.npy"
RESULT_UNIFORM = DATA / "optimization_result_200.json"
RESULT_RASTER = DATA / "optimization_result_population_raster.json"
OUTPUT_JSON = DATA / "demand_misspecification_crosseval.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_selected_sites(path: Path) -> list[int]:
    """Return the list of selected site indices from an optimization result."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(s) for s in payload["selected_sites"]]


def evaluate_coverage(
    visibility: sp.csr_matrix,
    open_sites: list[int],
    demand: np.ndarray,
) -> float:
    """Compute weighted coverage fraction for a fixed portfolio.

    Args:
        visibility: CSR matrix of shape (num_rows, num_sites), binary.
        open_sites: List of site column indices that are open.
        demand: 1-D demand weight vector of length num_rows.

    Returns:
        Weighted coverage fraction in [0, 1].
    """
    if len(open_sites) == 0:
        return 0.0
    # Sum visibility across open columns; any positive entry means covered.
    sub = visibility[:, open_sites]
    covered_mask = np.asarray(sub.sum(axis=1)).ravel() > 0
    total_demand = float(demand.sum())
    if total_demand == 0.0:
        return 0.0
    return float(demand[covered_mask].sum()) / total_demand


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load artefacts
    visibility: sp.csr_matrix = sp.load_npz(VIS_NPZ)
    raster_demand = np.load(RASTER_DEMAND_NPY).astype(np.float64)
    uniform_demand = np.ones(visibility.shape[0], dtype=np.float64)

    sites_uniform = _load_selected_sites(RESULT_UNIFORM)
    sites_raster = _load_selected_sites(RESULT_RASTER)

    num_rows = visibility.shape[0]
    assert raster_demand.shape == (num_rows,), "Demand vector length mismatch"

    # 2×2 cross-evaluation
    cov_uu = evaluate_coverage(visibility, sites_uniform, uniform_demand)
    cov_ur = evaluate_coverage(visibility, sites_uniform, raster_demand)
    cov_ru = evaluate_coverage(visibility, sites_raster, uniform_demand)
    cov_rr = evaluate_coverage(visibility, sites_raster, raster_demand)

    # Penalties (absolute pp drop and % of optimal)
    # Under uniform demand: optimal = cov_uu, penalty = cov_uu - cov_ru
    penalty_uniform_abs = cov_uu - cov_ru          # loss when raster sites used under uniform eval
    penalty_uniform_pct = penalty_uniform_abs / cov_uu * 100.0 if cov_uu > 0 else float("nan")

    # Under raster demand: optimal = cov_rr, penalty = cov_rr - cov_ur
    penalty_raster_abs = cov_rr - cov_ur           # loss when uniform sites used under raster eval
    penalty_raster_pct = penalty_raster_abs / cov_rr * 100.0 if cov_rr > 0 else float("nan")

    result = {
        "description": (
            "2x2 demand-misspecification cross-evaluation. "
            "Rows = portfolio optimized for; columns = demand model used for evaluation."
        ),
        "portfolios": {
            "uniform_optimal": {
                "source": RESULT_UNIFORM.name,
                "num_sites": len(sites_uniform),
                "site_indices": sites_uniform,
            },
            "raster_optimal": {
                "source": RESULT_RASTER.name,
                "num_sites": len(sites_raster),
                "site_indices": sites_raster,
            },
        },
        "cross_evaluation_table": {
            "uniform_portfolio_uniform_eval":  round(cov_uu, 6),
            "uniform_portfolio_raster_eval":   round(cov_ur, 6),
            "raster_portfolio_uniform_eval":   round(cov_ru, 6),
            "raster_portfolio_raster_eval":    round(cov_rr, 6),
        },
        "coverage_penalty": {
            "under_uniform_demand": {
                "description": "Loss incurred when the raster-optimal portfolio is evaluated under uniform demand",
                "optimal_coverage": round(cov_uu, 6),
                "penalty_portfolio_coverage": round(cov_ru, 6),
                "absolute_pp_drop": round(penalty_uniform_abs, 6),
                "pct_of_optimal": round(penalty_uniform_pct, 4),
            },
            "under_raster_demand": {
                "description": "Loss incurred when the uniform-optimal portfolio is evaluated under raster demand",
                "optimal_coverage": round(cov_rr, 6),
                "penalty_portfolio_coverage": round(cov_ur, 6),
                "absolute_pp_drop": round(penalty_raster_abs, 6),
                "pct_of_optimal": round(penalty_raster_pct, 4),
            },
        },
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    sep = "-" * 70
    print(sep)
    print("DEMAND MISSPECIFICATION CROSS-EVALUATION")
    print(sep)
    print(f"Visibility matrix : {visibility.shape[0]} rows × {visibility.shape[1]} sites, "
          f"{visibility.nnz} nonzeros")
    print(f"Uniform portfolio : {len(sites_uniform)} sites  ({RESULT_UNIFORM.name})")
    print(f"Raster portfolio  : {len(sites_raster)} sites  ({RESULT_RASTER.name})")
    print()
    print(f"{'':30s} {'Uniform Coverage':>18s} {'Raster-Weighted Cov':>21s}")
    print(f"{'':30s} {'(fraction)':>18s} {'(fraction)':>21s}")
    print("-" * 70)
    print(f"{'Uniform-optimal portfolio':30s} {cov_uu:18.6f} {cov_ur:21.6f}")
    print(f"{'Raster-optimal portfolio':30s} {cov_ru:18.6f} {cov_rr:21.6f}")
    print()
    print(f"{'':30s} {'Uniform Coverage':>18s} {'Raster-Weighted Cov':>21s}")
    print(f"{'':30s} {'(%)':>18s} {'(%)':>21s}")
    print("-" * 70)
    print(f"{'Uniform-optimal portfolio':30s} {cov_uu*100:17.4f}% {cov_ur*100:20.4f}%")
    print(f"{'Raster-optimal portfolio':30s} {cov_ru*100:17.4f}% {cov_rr*100:20.4f}%")
    print()
    print("PENALTIES (loss from building the wrong portfolio)")
    print(sep)
    print(f"Under UNIFORM demand:  raster-optimal sites achieve {cov_ru*100:.4f}% vs. "
          f"optimal {cov_uu*100:.4f}%")
    print(f"  Absolute penalty : {penalty_uniform_abs*100:+.4f} pp")
    print(f"  Relative penalty : {penalty_uniform_pct:.4f}% of optimal")
    print()
    print(f"Under RASTER demand:   uniform-optimal sites achieve {cov_ur*100:.4f}% vs. "
          f"optimal {cov_rr*100:.4f}%")
    print(f"  Absolute penalty : {penalty_raster_abs*100:+.4f} pp")
    print(f"  Relative penalty : {penalty_raster_pct:.4f}% of optimal")
    print(sep)
    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
