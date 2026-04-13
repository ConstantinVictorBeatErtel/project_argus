"""Pareto frontier marginal analysis at the 25-degree elevation threshold.

Loads the sensitivity results for uniform, city-point proxy, and raster demand
and computes:

  1. Marginal coverage gain per additional station at each budget step.
  2. The "knee" of the frontier: the first station count where the marginal
     gain drops below half its maximum value.
  3. The station count / coverage level where cost per incremental coverage
     point doubles relative to the cheapest segment.

Budgets analysed: 5, 10, 15, 20, 30 stations (elevation = 25 degrees).

Results are written to data/processed/pareto_marginal_analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"

SENS_UNIFORM = DATA / "sensitivity_results.json"
SENS_PROXY = DATA / "sensitivity_results_population_proxy.json"
SENS_RASTER = DATA / "sensitivity_results_population_raster.json"
OUTPUT_JSON = DATA / "pareto_marginal_analysis.json"

ELEVATION_DEG = 25.0
BUDGETS = [5, 10, 15, 20, 30]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_25deg_rows(path: Path) -> dict[int, float]:
    """Return {max_ground_stations: achieved_coverage} for elevation=25 deg."""
    records = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, float] = {}
    for row in records:
        if abs(float(row["elevation_deg"]) - ELEVATION_DEG) < 0.01:
            result[int(row["max_ground_stations"])] = float(row["achieved_coverage"])
    return result


def marginal_analysis(coverages: dict[int, float], label: str) -> dict:
    """Compute marginal coverage statistics for the given budget→coverage map."""
    budgets = BUDGETS
    # Ensure all budgets present
    cov = [coverages[b] for b in budgets]

    segments: list[dict] = []
    for i in range(len(budgets) - 1):
        b0, b1 = budgets[i], budgets[i + 1]
        c0, c1 = cov[i], cov[i + 1]
        delta_stations = b1 - b0
        delta_cov = c1 - c0
        marginal_per_station = delta_cov / delta_stations
        # Stations needed per 1 pp of coverage gain
        stations_per_pp = delta_stations / (delta_cov * 100.0) if delta_cov > 0 else float("inf")
        segments.append(
            {
                "from_stations": b0,
                "to_stations": b1,
                "coverage_from": round(c0, 8),
                "coverage_to": round(c1, 8),
                "delta_coverage": round(delta_cov, 8),
                "delta_stations": delta_stations,
                "marginal_coverage_per_station": round(marginal_per_station, 8),
                "stations_per_1pp_gain": round(stations_per_pp, 4),
            }
        )

    marginals = [s["marginal_coverage_per_station"] for s in segments]
    max_marginal = max(marginals)
    half_max = max_marginal / 2.0

    # Knee: first budget where marginal drops below half of its maximum
    knee_budget: int | None = None
    for i, seg in enumerate(segments):
        if seg["marginal_coverage_per_station"] < half_max:
            knee_budget = seg["from_stations"]
            break
    if knee_budget is None:
        knee_budget = budgets[-1]

    # Find cheapest segment (lowest stations_per_pp)
    finite_costs = [s["stations_per_1pp_gain"] for s in segments if s["stations_per_1pp_gain"] < float("inf")]
    if finite_costs:
        min_cost = min(finite_costs)
        double_cost_threshold = 2.0 * min_cost
        double_segment: dict | None = None
        for seg in segments:
            if seg["stations_per_1pp_gain"] >= double_cost_threshold:
                double_segment = seg
                break
    else:
        double_segment = None
        min_cost = float("inf")
        double_cost_threshold = float("inf")

    return {
        "demand_model": label,
        "elevation_deg": ELEVATION_DEG,
        "budget_coverage_table": {b: round(coverages[b], 8) for b in budgets},
        "segments": segments,
        "max_marginal_coverage_per_station": round(max_marginal, 8),
        "knee_station_budget": knee_budget,
        "knee_description": (
            f"Marginal coverage drops below {half_max*100:.4f}% per station "
            f"(half of peak {max_marginal*100:.4f}%) after {knee_budget} stations"
        ),
        "cheapest_segment_stations_per_1pp": round(min_cost, 4),
        "cost_doubling_threshold_stations_per_1pp": round(double_cost_threshold, 4),
        "cost_doubles_at_segment": double_segment,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cov_uniform = extract_25deg_rows(SENS_UNIFORM)
    cov_proxy = extract_25deg_rows(SENS_PROXY)
    cov_raster = extract_25deg_rows(SENS_RASTER)

    analysis_uniform = marginal_analysis(cov_uniform, "uniform")
    analysis_proxy = marginal_analysis(cov_proxy, "city_point_proxy")
    analysis_raster = marginal_analysis(cov_raster, "population_raster")

    result = {
        "elevation_deg": ELEVATION_DEG,
        "budgets_analysed": BUDGETS,
        "analyses": [analysis_uniform, analysis_proxy, analysis_raster],
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    sep = "=" * 75
    print(sep)
    print("PARETO FRONTIER MARGINAL ANALYSIS  (elevation threshold = 25°)")
    print(sep)

    for analysis in result["analyses"]:
        label = analysis["demand_model"].upper().replace("_", " ")
        print(f"\n── {label} ──")

        # Budget/coverage table
        print(f"  {'Stations':>10s}  {'Coverage (%)':>14s}")
        print(f"  {'-'*10}  {'-'*14}")
        for b, c in analysis["budget_coverage_table"].items():
            print(f"  {b:>10d}  {c*100:>13.4f}%")

        # Marginal analysis table
        print()
        print(f"  {'Segment':>12s}  {'Δ Coverage':>12s}  {'Marg/station':>14s}  {'Sta/1pp':>10s}")
        print(f"  {'-'*12}  {'-'*12}  {'-'*14}  {'-'*10}")
        for seg in analysis["segments"]:
            seg_label = f"{seg['from_stations']}→{seg['to_stations']}"
            print(
                f"  {seg_label:>12s}  "
                f"{seg['delta_coverage']*100:>11.4f}%  "
                f"{seg['marginal_coverage_per_station']*100:>13.4f}%  "
                f"{seg['stations_per_1pp_gain']:>10.4f}"
            )

        # Knee
        print()
        print(f"  Knee: {analysis['knee_description']}")

        # Cost doubling
        ds = analysis["cost_doubles_at_segment"]
        if ds:
            print(
                f"  Cost doubles: segment {ds['from_stations']}→{ds['to_stations']} "
                f"({ds['stations_per_1pp_gain']:.4f} sta/pp vs cheapest "
                f"{analysis['cheapest_segment_stations_per_1pp']:.4f} sta/pp, "
                f"threshold {analysis['cost_doubling_threshold_stations_per_1pp']:.4f})"
            )
        else:
            print("  Cost doubling: not reached within the analysed budget range")

    print(f"\n{sep}")
    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
