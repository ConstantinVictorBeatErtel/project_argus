# satellite-gs-optimizer Summary

## What The Project Does

`satellite-gs-optimizer` finds cost-effective ground-station placements for satellite constellations. It combines orbital propagation, WGS84 visibility geometry, sparse tensor storage, backhaul feasibility, demand weighting, and MILP optimization.

For the class project, the strongest framing is supply-chain network design: candidate ground-station sites are facilities, satellite-time pairs are demand nodes, and orbital visibility defines the feasible assignment arcs. The main artifact is the cost-coverage frontier under station-count, visibility, latency, and backhaul assumptions.

The core artifact is a sparse visibility tensor `a_ijt`, stored as CSR with rows `satellite_index * num_times + time_index` and columns `site_index`. A nonzero means a satellite can see a candidate site at that time. The optimizer then chooses which sites to open and which visible satellite-time demands to serve.

## Architecture

Current intended layout:

```text
data/
  raw/
    tle/
    candidates/
    population/
    backhaul/
    ground_truth/
  processed/
    positions.h5
    visibility.npz
    slant_range.npz
    service_cost.npz
    demand.parquet
    demand.npy
    backhaul_mask.csv
src/
  simulation/
    propagator.py
    visibility.py
    demand.py
  optimization/
    milp.py
    rolling_horizon.py
    heuristics.py
    pareto.py
  constraints/
  evaluation/
  visualization/
tests/
scripts/
config/parameters.yaml
notebooks/
```

## Where We Are

Phase 1 foundation is underway. The overarching plan remains valid, but the project narrative has been updated: the proxy experiment should not be judged against a fixed 90% coverage promise. Coverage is an output of the scenario assumptions, and the central question is the tradeoff between cost, coverage, latency, and feasible infrastructure.

Completed:

- Created the project scaffold and canonical data directories.
- Added `config/parameters.yaml` for paths, scenario sizing, propagation, visibility, latency, optimization, rolling-horizon, Pareto, and validation settings.
- Implemented `src/simulation/propagator.py`:
  - TLE parsing
  - UTC time-grid generation
  - Skyfield SGP4 propagation hook to ITRS/ECEF
  - optional HDF5 export
- Implemented `src/simulation/visibility.py`:
  - WGS84 geodetic-to-ECEF conversion
  - local-up elevation geometry
  - backhaul/regulatory feasibility masking
  - sparse CSR visibility and slant-range generation
- Standardized the visibility matrix layout to `(satellite_time) x site`.
- Implemented `src/optimization/milp.py`:
  - binary site-open variables `y_i`
  - binary visible assignment variables `x_ri`
  - one-assignment-per-satellite-time-row constraints
  - `y_i <= b_i` feasibility enforcement
  - weighted coverage requirement
  - fixed-cost plus optional sparse service-cost objective
- Added `scripts/run_optimization.py` for solving from a precomputed `visibility.npz`.
- Added `scripts/build_visibility_tensor.py` for producing `visibility.npz` and optional slant-range tensors from ECEF positions or TLE propagation.
- Added `scripts/download_tle.py` for fetching, validating, normalizing, and recording metadata for CelesTrak TLE files.
- Added `src/constraints/backhaul.py` and `scripts/build_backhaul_mask.py` to compute `b_i` feasibility masks from IXP/fiber proximity.
- Added `src/simulation/demand.py` and `scripts/build_demand.py` for a uniform Phase 1 demand baseline in Parquet plus solver-ready `.npy`.
- Added population proxy demand tooling:
  - `scripts/download_population_proxy.py`
  - Natural Earth populated-place CSV normalization to `data/raw/population/populated_places.csv`
  - ECEF-to-WGS84 subpoint conversion
  - Gaussian city-population kernel demand weighting
  - normalized solver vector at `data/processed/demand_population_proxy.npy`
- Added `scripts/build_service_cost.py` to convert sparse slant range into sparse propagation-delay service costs.
- Wired optional `backhaul_mask.csv` support into visibility precomputation and the MILP CLI.
- Added `scripts/create_synthetic_instance.py` to generate deterministic toy candidates, backhaul points, and ECEF positions for full pipeline checks.
- Added `src/simulation/candidates.py`, `scripts/build_candidate_grid.py`, and `scripts/build_proxy_backhaul.py` so Phase 1 can proceed without external candidate or backhaul datasets. These are proxy inputs, not authoritative site/fiber data.
- Installed/runtime-verified `skyfield`, `h5py`, and `pytest`.
- Improved generated proxy candidate downsampling with deterministic farthest-point sampling for broader geographic coverage.
- Added `src/evaluation/run_manifest.py` and `scripts/write_run_manifest.py`; current run metadata is saved to `data/processed/run_manifest.json`.
- Added `src/optimization/pareto.py` and `scripts/run_pareto.py` for epsilon-constraint coverage/cost sweeps.
- Added `src/visualization/pareto_plot.py`, `src/visualization/coverage_maps.py`, `scripts/plot_pareto.py`, and `scripts/plot_selected_sites.py`.
- Added max-coverage sensitivity tooling:
  - `src/optimization/milp.py` now includes `solve_max_coverage_milp`
  - `src/evaluation/sensitivity.py`
  - `scripts/run_sensitivity.py`
  - `src/visualization/sensitivity_plot.py`
  - `scripts/plot_sensitivity.py`
- Added class-facing documentation:
  - `README.md`
  - `docs/project_framing.md`
  - `docs/related_work.md`
  - `docs/report_outline.md`
- Ran a real TLE-backed Phase 1 proxy instance:
  - 20 Starlink TLE records from CelesTrak
  - 50 generated rough-land proxy candidate sites, farthest-sampled from a 6-degree source grid
  - 48 time steps over the configured 4-hour window
  - 50 proxy-feasible sites after using a relaxed development backhaul radius
  - sparse visibility matrix shape `(960, 50)`
  - 198 nonzero visibility arcs at the 25-degree elevation threshold
  - sparse service-cost matrix from slant range propagation delay
  - uniform demand vector with total demand `960`
  - PuLP/CBC solve with `max_ground_stations=20`, coverage target `0.14`
  - optimal result saved to `data/processed/optimization_result.json`
  - achieved coverage `0.140625` with 18 selected sites
- Ran a Pareto sweep on the 50-site proxy instance:
  - frontier saved to `data/processed/pareto_frontier.csv`
  - optimal targets through `0.14`
  - infeasible targets at `0.16`, `0.18`, and `0.20`
- Ran a 200-site proxy comparison:
  - 200 generated rough-land proxy candidate sites, farthest-sampled from a 4-degree source grid
  - sparse visibility matrix shape `(960, 200)`
  - 732 nonzero visibility arcs at the 25-degree elevation threshold
  - row visibility upper bound `0.596875`
  - frontier saved to `data/processed/pareto_frontier_200.csv`
  - optimal targets through `0.20`; infeasible at `0.30` and above under `max_ground_stations=20`
  - comparison result saved to `data/processed/optimization_result_200.json`
  - achieved coverage `0.2` with 19 selected sites
- Generated visualization HTML artifacts:
  - `data/processed/pareto_frontier.html` compares the 50-site and 200-site Pareto frontiers
  - `data/processed/selected_sites_50.html` maps selected sites for the 50-site proxy run
  - `data/processed/selected_sites_200.html` maps selected sites for the 200-site proxy run
- Ran station-budget and elevation sensitivity on the 200-site proxy candidate set:
  - budgets: 5, 10, 15, 20, 30 ground stations
  - elevation thresholds: 0, 10, 25 degrees
  - results saved to `data/processed/sensitivity_results.csv` and `data/processed/sensitivity_results.json`
  - chart saved to `data/processed/sensitivity_coverage.html`
  - best diagnostic case: 0-degree threshold and 30 stations achieved 0.897917 coverage, near the 0-degree row visibility bound of 0.909375
  - at 10 degrees and 30 stations, achieved coverage was 0.595833
  - at 25 degrees and 30 stations, achieved coverage was 0.272917
  - at the original 25-degree, 20-station setting, max coverage was 0.209375
- Ran the same sensitivity grid using population-weighted proxy demand:
  - population source: Natural Earth 10m populated places GeoJSON, normalized to 4,215 city/town points with population at least 50,000
  - demand vector saved to `data/processed/demand_population_proxy.npy` and `data/processed/demand_population_proxy.parquet`
  - weighted sensitivity saved to `data/processed/sensitivity_results_population_proxy.csv` and `.json`
  - chart saved to `data/processed/sensitivity_coverage_population_proxy.html`
  - at 0 degrees and 30 stations, weighted demand coverage reached 0.999300
  - at 10 degrees and 30 stations, weighted demand coverage reached 0.959469
  - at 25 degrees and 20 stations, weighted demand coverage reached 0.517199
  - at 25 degrees and 30 stations, weighted demand coverage reached 0.633632
  - demand-weighted visibility upper bound at 25 degrees was 0.887413, compared with the row visibility upper bound of 0.596875
- Solved a population-weighted 25-degree exact optimization target:
  - target coverage: 0.50
  - max stations: 20
  - status: optimal
  - achieved weighted demand coverage: 0.500338
  - selected 19 sites
  - result saved to `data/processed/optimization_result_population_proxy.json`
  - map saved to `data/processed/selected_sites_population_proxy.html`
- Compared selected-site geography for the 25-degree, 20-station max-coverage runs:
  - uniform selected sites and population-weighted selected sites overlap at 4 sites
  - 16 sites are uniform-only and 16 sites are population-only
  - Jaccard similarity is 0.111111
  - comparison table saved to `data/processed/selected_site_comparison_uniform_vs_population.csv`
  - comparison summary saved to `data/processed/selected_site_comparison_uniform_vs_population.json`
  - comparison map saved to `data/processed/selected_site_comparison_uniform_vs_population.html`
- Added focused tests for visibility geometry, feasibility masking, and small exact MILP feasibility.

Important Phase 1 caveat:

- The generated proxy geometry cannot support the original 90% coverage target under the original 25-degree threshold and 20-station budget. With 50 proxy sites and a 25-degree elevation threshold, about 20.6% of satellite-time rows have any visible site, and the `max_ground_stations=20` cap makes the feasible frontier lower. A 200-site proxy comparison raises the row visibility upper bound to about 59.7%, but the 20-site cap still makes targets above 20% infeasible. A sensitivity run shows that relaxing the threshold to 0 degrees and allowing 30 stations achieves about 89.8% coverage, confirming that the model can reach high coverage when the physical visibility assumption and station budget allow it.
- This caveat should be used as a project finding, not treated as a failure. The model is exposing how the feasible frontier changes when candidate density, elevation threshold, and station budget change.

## Class Project Framing

Recommended research question:

> How do candidate-site density, ground-station budget, backhaul feasibility, and satellite visibility constraints shape the cost-coverage frontier for a LEO ground-station network?

Recommended report claim:

> This project develops and evaluates a mixed-integer facility-location model for LEO ground-station placement. A sparse orbital visibility graph defines feasible service arcs between satellite-time demand nodes and candidate ground-station facilities. The optimization chooses a budgeted set of stations and assignments to trace the cost-coverage frontier under visibility, latency, and backhaul feasibility constraints.

Direct research anchors are now recorded in `docs/related_work.md`. The key bridge is that recent satellite gateway-placement and ground-station-selection papers justify the domain model, while the maximal covering location problem literature justifies the class/supply-chain modeling frame.

## Dependency Notes

Available locally:

- `numpy`
- `scipy`
- `pandas`
- `pulp`
- `PyYAML`
- `skyfield`
- `h5py`
- `pytest`

The code still imports `skyfield` and `h5py` lazily so lightweight geometry and MILP tests can run even if a future environment lacks those packages.

## What Is Next

Immediate next steps:

1. Use `docs/report_outline.md` as the skeleton for the class report.
2. Turn `docs/related_work.md` into 1-2 polished report paragraphs that explicitly connect facility location, maximum coverage, and satellite ground-station optimization.
3. Add static image export for the visualizations if needed, via `kaleido` or a matplotlib fallback.
4. Add Heavens-Above ISS regression data if reliable pass-window data becomes available.

Later phases:

- Phase 2: GPW demand, backhaul filtering, uniform-vs-population demand comparison.
- Phase 3: 24-hour sparse scaling, rolling-horizon decomposition, runtime and MIP-gap comparison.
- Phase 4: Pareto sweep, Starlink gateway clustering validation, sensitivity/disruption scenarios.
