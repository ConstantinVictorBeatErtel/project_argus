# Report Outline

## Working Title

Cost-Coverage Tradeoffs in LEO Ground-Station Network Design

## One-Sentence Pitch

We model satellite ground-station placement as a fixed-charge facility-location problem where time-varying orbital visibility determines which satellite demand nodes can be served by each candidate facility.

## 1. Motivation

LEO satellite networks need terrestrial gateway infrastructure to relay data. Opening more ground stations can improve coverage and reduce latency, but each station carries fixed infrastructure cost and must satisfy feasibility constraints such as line-of-sight visibility and backhaul access. This is a supply-chain network design problem with a dynamic spatial feasibility graph.

## 2. Literature

Use the related-work file as the source:

- Abe et al. for satellite gateway placement and latency-aware routing.
- Eddy, Ho, and Kochenderfer for LEO ground-station integer programming and reduced-time-domain scaling.
- del Portillo, Cameron, and Crawley for large LEO ground-segment architecture tradeoffs.
- Petelin, Antoniou, and Papa for Pareto/multi-objective ground-station scheduling.
- Cheung and Lee for MIP/heuristic space communication scheduling.
- Church and ReVelle plus Murray for maximum coverage and facility-location foundations.

## 3. Model

Decision variables:

- `y_i = 1` if candidate ground station `i` is opened.
- `x_ri = 1` if satellite-time row `r = (j,t)` is assigned to open site `i`.

Core parameters:

- `a_ri`: sparse visibility matrix from satellite-time row `r` to site `i`.
- `d_r`: demand weight for satellite-time row `r`.
- `f_i`: fixed site-opening cost.
- `c_ri`: latency-weighted service cost.
- `b_i`: backhaul feasibility flag.

Main constraints:

- Assign only through visible arcs.
- Assign only to open stations.
- Open at most the station budget.
- Exclude infeasible backhaul sites with `y_i <= b_i`.
- Achieve an epsilon coverage target for each Pareto solve.

Objective:

Minimize fixed station-opening cost plus latency-weighted service cost for a chosen coverage target.

## 4. Computational Pipeline

1. Download or load TLEs.
2. Propagate satellites with Skyfield/SGP4.
3. Generate candidate sites for the proxy experiment.
4. Compute WGS84 line-of-sight visibility and slant range.
5. Store visibility and range as sparse CSR matrices.
6. Build uniform demand for Phase 1.
7. Solve the MILP with PuLP/CBC.
8. Sweep coverage targets to build a Pareto frontier.
9. Visualize selected sites and cost-coverage tradeoffs.
10. Run sensitivity analysis across station budgets and elevation thresholds.

## 5. Current Results

Phase 1 proxy results:

- 20 Starlink satellites.
- 4-hour time horizon with 48 time steps.
- 25-degree elevation threshold.
- 50-site and 200-site generated candidate experiments.
- Optimal 50-site run achieved 14.0625% coverage with 18 selected sites.
- Optimal 200-site comparison achieved 20% coverage with 19 selected sites.
- Higher coverage targets became infeasible under the station budget and elevation threshold.

Station-budget and elevation sensitivity, using the 200-site proxy candidate set:

| Elevation threshold | 5 sites | 10 sites | 15 sites | 20 sites | 30 sites | Row visibility upper bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 35.2% | 56.8% | 71.7% | 81.5% | 89.8% | 90.9% |
| 10 deg | 19.2% | 31.7% | 40.7% | 48.1% | 59.6% | 79.3% |
| 25 deg | 7.4% | 12.8% | 17.1% | 20.9% | 27.3% | 59.7% |

Artifacts:

- `data/processed/sensitivity_results.csv`
- `data/processed/sensitivity_results.json`
- `data/processed/sensitivity_coverage.html`
- `data/processed/sensitivity/elev_0/visibility.npz`
- `data/processed/sensitivity/elev_10/visibility.npz`
- `data/processed/sensitivity/elev_25/visibility.npz`

Interpretation:

The experiment shows that coverage is constrained by candidate geography, elevation threshold, and station budget. The sensitivity grid is especially important: at a permissive 0-degree elevation threshold, 30 stations cover 89.8% of satellite-time demand, near the 90.9% row visibility upper bound. At the stricter 25-degree threshold, even 30 stations cover only 27.3%, despite 59.7% of satellite-time rows having at least one visible candidate. This separates two effects: physical visibility limits and station-budget limits.

The useful result is the cost/coverage frontier, not a single universal coverage target.

Population-weighted proxy demand:

To move beyond uniform satellite-time demand, we added a lightweight population proxy based on Natural Earth populated-place points. For each satellite-time row, we convert the satellite ECEF position to an approximate WGS84 subpoint and compute a Gaussian kernel against city population points within the global populated-places dataset. The resulting demand vector is normalized to total demand 960, matching the uniform baseline, so coverage percentages remain comparable.

This is not yet a true GPW/WorldPop raster convolution. It is a smaller proxy that captures the key effect needed for the class analysis: demand is concentrated where people are, not uniformly distributed over all satellite-time rows.

Population-weighted sensitivity, using the same 200-site proxy candidate set:

| Elevation threshold | 5 sites | 10 sites | 15 sites | 20 sites | 30 sites | Demand visibility upper bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 67.7% | 91.2% | 98.2% | 99.7% | 99.9% | 99.9% |
| 10 deg | 48.9% | 68.1% | 79.1% | 86.6% | 95.9% | 99.3% |
| 25 deg | 22.1% | 34.6% | 44.2% | 51.7% | 63.4% | 88.7% |

Artifacts:

- `data/raw/population/populated_places.csv`
- `data/raw/population/populated_places.metadata.json`
- `data/processed/demand_population_proxy.parquet`
- `data/processed/demand_population_proxy.npy`
- `data/processed/sensitivity_results_population_proxy.csv`
- `data/processed/sensitivity_results_population_proxy.json`
- `data/processed/sensitivity_coverage_population_proxy.html`
- `data/processed/optimization_result_population_proxy.json`
- `data/processed/selected_sites_population_proxy.html`

Interpretation:

The population-weighted run changes the meaning of coverage. At the original 25-degree threshold and 20-station budget, uniform row coverage is 20.9%, but population-weighted demand coverage is 51.7%. This does not mean the geometry improved; it means the optimizer is covering the high-demand satellite-time rows first. The demand-weighted visibility upper bound at 25 degrees is 88.7%, much higher than the row visibility upper bound of 59.7%, because high-population ground-track rows are disproportionately visible in the proxy scenario.

Selected-site geography comparison:

We compared the 25-degree, 20-station max-coverage portfolios under uniform demand and population-weighted demand. The two portfolios overlap at only 4 selected sites out of 36 unique selected sites, for a Jaccard similarity of 0.111. This is a strong signal that the demand model changes the infrastructure decision, not just the reported objective value.

The overlapping selected sites are indices 48, 121, 132, and 137. The uniform model has 16 sites selected only by uniform demand, while the population-weighted model has 16 sites selected only by population demand. In report language: once demand is concentrated around populated ground-track regions, the optimizer reallocates most of the station portfolio.

Artifacts:

- `data/processed/selected_site_comparison_uniform_vs_population.csv`
- `data/processed/selected_site_comparison_uniform_vs_population.json`
- `data/processed/selected_site_comparison_uniform_vs_population.html`

Phase 2 raster demand:

To complete Phase 2, we replaced the city-point proxy with an official WorldPop global 1 km constrained raster. For each satellite-time row, we compute the satellite subpoint, read only a local raster window around that point, and apply a Gaussian kernel over the raster population values. This keeps the demand model tied to gridded population density while avoiding a full in-memory raster load.

We use WorldPop rather than GPW here because it is official, global, directly downloadable, and small enough to support a reproducible course-project workflow. This is a pragmatic substitution, not a change in the core modeling idea.

Raster-backed sensitivity, using the same 200-site proxy candidate set:

| Elevation threshold | 5 sites | 10 sites | 15 sites | 20 sites | 30 sites | Demand visibility upper bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 73.8% | 94.2% | 99.1% | 99.9% | 100.0% | 100.0% |
| 10 deg | 53.7% | 72.3% | 85.2% | 91.6% | 98.4% | 99.7% |
| 25 deg | 28.3% | 44.0% | 54.1% | 61.4% | 71.5% | 88.4% |

Artifacts:

- `data/raw/population/worldpop_2024_1km_constrained.tif`
- `data/raw/population/worldpop_2024_1km_constrained.metadata.json`
- `data/processed/demand_population_raster.parquet`
- `data/processed/demand_population_raster.npy`
- `data/processed/sensitivity_results_population_raster.csv`
- `data/processed/sensitivity_results_population_raster.json`
- `data/processed/sensitivity_coverage_population_raster.html`
- `data/processed/optimization_result_population_raster.json`
- `data/processed/selected_sites_population_raster.html`
- `data/processed/phase2_comparison.csv`
- `data/processed/phase2_comparison.json`
- `data/processed/phase2_comparison.md`

Interpretation:

At the original 25-degree threshold and 20-station budget, uniform demand achieves 20.9% coverage, while the raster-backed demand model achieves 61.4% coverage. The demand-weighted visibility upper bound rises from 59.7% under uniform row demand to 88.4% under raster demand, which shows that high-population ground-track opportunities are much easier to cover than a uniform row baseline suggests.

This is exactly the managerial point Phase 2 needed to establish: demand assumptions materially change both the measured frontier and the optimal infrastructure portfolio.

Selected-site geography comparison:

We compared the 25-degree, 20-station max-coverage portfolios under uniform demand and raster-backed demand. The overlap is 0 sites out of 40 selected sites, with Jaccard similarity 0.0. In other words, once demand is modeled on a real population raster, the optimizer reallocates the entire station portfolio.

Artifacts:

- `data/processed/selected_site_comparison_uniform_vs_raster.csv`
- `data/processed/selected_site_comparison_uniform_vs_raster.json`
- `data/processed/selected_site_comparison_uniform_vs_raster.html`

## 6. Limitations

- Candidate sites are generated proxy sites, not authoritative ground-station candidates.
- Backhaul hubs are proxy feasibility anchors, not measured fiber/IXP infrastructure.
- Candidate and backhaul inputs are still proxies even though demand is now raster-backed.
- Demand uses a WorldPop raster rather than GPW specifically.
- Validation against known gateway clusters is not yet implemented.
- The 0-degree elevation case is a diagnostic lower-barrier case, not an operational recommendation.

## 7. Next Extensions

- Add real known-gateway validation if reliable coordinates are available.
- Extend the horizon and use rolling-horizon decomposition if full-horizon MILP solves become slow.

## Core Takeaway

The project is strongest when presented as a decision model: given a candidate network and physical feasibility graph, what station portfolio achieves the desired coverage at minimum cost, and how expensive is the next increment of coverage?
