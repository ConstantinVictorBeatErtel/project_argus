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
- Uniform demand, Natural Earth population proxy demand, and WorldPop raster-backed demand.
- PuLP/CBC MILP for site opening and satellite-time assignment.
- Epsilon-constraint Pareto sweep.
- Max-coverage sensitivity analysis across station budgets and elevation thresholds.
- Plotly HTML visualizations for Pareto frontier, selected sites, and sensitivity results.
- A focused pytest suite for geometry, masks, demand models, MILP feasibility, sensitivity, and visualization generation.

Current generated artifacts live in `data/processed/`.

## Streamlit App

You can explore the 200-site selected-site map as a Streamlit app instead of opening the raw HTML export directly:

```bash
streamlit run streamlit_app.py
```

The app reproduces the `data/processed/selected_sites_200.html` visualization from:

- `data/processed/visibility_metadata_200.json`
- `data/processed/optimization_result_200.json`

and adds run metrics plus a table of the selected sites.

Current sensitivity artifacts:

- `data/processed/sensitivity_results.csv`
- `data/processed/sensitivity_results.json`
- `data/processed/sensitivity_coverage.html`
- `data/processed/sensitivity_results_population_proxy.csv`
- `data/processed/sensitivity_results_population_proxy.json`
- `data/processed/sensitivity_coverage_population_proxy.html`
- `data/processed/sensitivity_results_population_raster.csv`
- `data/processed/sensitivity_results_population_raster.json`
- `data/processed/sensitivity_coverage_population_raster.html`

Current population proxy artifacts:

- `data/raw/population/populated_places.csv`
- `data/processed/demand_population_proxy.parquet`
- `data/processed/demand_population_proxy.npy`
- `data/processed/optimization_result_population_proxy.json`
- `data/processed/selected_sites_population_proxy.html`
- `data/processed/selected_site_comparison_uniform_vs_population.csv`
- `data/processed/selected_site_comparison_uniform_vs_population.json`
- `data/processed/selected_site_comparison_uniform_vs_population.html`

Current raster Phase 2 artifacts:

- `data/raw/population/worldpop_2024_1km_constrained.tif`
- `data/raw/population/worldpop_2024_1km_constrained.metadata.json`
- `data/processed/demand_population_raster.parquet`
- `data/processed/demand_population_raster.npy`
- `data/processed/optimization_result_population_raster.json`
- `data/processed/selected_sites_population_raster.html`
- `data/processed/selected_site_comparison_uniform_vs_raster.csv`
- `data/processed/selected_site_comparison_uniform_vs_raster.json`
- `data/processed/selected_site_comparison_uniform_vs_raster.html`
- `data/processed/phase2_comparison.csv`
- `data/processed/phase2_comparison.json`
- `data/processed/phase2_comparison.md`
- `data/processed/phase2_run_manifest.json`

## Documentation

- `summary.md`: project ledger with what has been built and what is next.
- `docs/project_framing.md`: class-facing model framing and updated plan.
- `docs/related_work.md`: research papers and direct mapping to this project.
- `docs/report_outline.md`: draft structure for the class report.

## Important Data Caveat

Candidate sites and backhaul hubs are still proxy inputs and should be described that way in the report. The demand side is stronger now: Phase 2 uses an official WorldPop global 1 km constrained raster as the population-weighted demand source.

The remaining data upgrade is GPW-specific replication or real gateway validation, not the basic ability to run a raster-backed demand model.
