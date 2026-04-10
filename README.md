# satellite-gs-optimizer

This project models LEO satellite ground-station placement as a supply-chain network design problem.

The optimizer chooses which candidate ground-station facilities to open and which satellite-time demand nodes to serve. The feasible assignment arcs come from orbital propagation and WGS84 line-of-sight visibility, then are stored as sparse CSR matrices so the model can scale beyond small dense tensors.

## Current Framing

The class-facing framing is:

> A fixed-charge facility-location and maximum-coverage model for satellite ground-segment infrastructure, with time-varying orbital visibility as the feasibility graph.

The project is not built around a fixed promise of 90% proxy coverage. Instead, it estimates the feasible cost-coverage frontier under the chosen station budget, candidate-site set, elevation threshold, and backhaul feasibility assumptions.

## Current Status

Implemented:

- TLE loading and Skyfield/SGP4 propagation.
- WGS84 candidate-site geometry.
- Sparse visibility and slant-range matrix construction.
- Proxy candidate-site and proxy backhaul generation.
- Uniform Phase 1 demand and Natural Earth populated-place proxy demand.
- PuLP/CBC MILP for site opening and satellite-time assignment.
- Epsilon-constraint Pareto sweep.
- Max-coverage sensitivity analysis across station budgets and elevation thresholds.
- Plotly HTML visualizations for Pareto frontier, selected sites, and sensitivity results.
- A focused pytest suite for geometry, masks, MILP feasibility, sensitivity, and visualization generation.

Current generated artifacts live in `data/processed/`.

Current sensitivity artifacts:

- `data/processed/sensitivity_results.csv`
- `data/processed/sensitivity_results.json`
- `data/processed/sensitivity_coverage.html`
- `data/processed/sensitivity_results_population_proxy.csv`
- `data/processed/sensitivity_results_population_proxy.json`
- `data/processed/sensitivity_coverage_population_proxy.html`

Current population proxy artifacts:

- `data/raw/population/populated_places.csv`
- `data/processed/demand_population_proxy.parquet`
- `data/processed/demand_population_proxy.npy`
- `data/processed/optimization_result_population_proxy.json`
- `data/processed/selected_sites_population_proxy.html`
- `data/processed/selected_site_comparison_uniform_vs_population.csv`
- `data/processed/selected_site_comparison_uniform_vs_population.json`
- `data/processed/selected_site_comparison_uniform_vs_population.html`

## Documentation

- `summary.md`: project ledger with what has been built and what is next.
- `docs/project_framing.md`: class-facing model framing and updated plan.
- `docs/related_work.md`: research papers and direct mapping to this project.
- `docs/report_outline.md`: draft structure for the class report.

## Important Data Caveat

Phase 1 uses generated proxy candidate sites and proxy backhaul hubs. These are enough to test the model and demonstrate the network-design tradeoff, but they should be described as proxies in the report. Real GPW demand, authoritative gateway locations, and validated backhaul data can be added later without changing the core architecture.

The current population-weighted demand is a city-population proxy, not a full population-density raster model. GPW or WorldPop raster convolution remains a higher-fidelity upgrade.
