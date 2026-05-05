# Project Argus Final Report Review and Glossary

Generated after syncing `main` from GitHub on 2026-05-05. The repo was already up to date.

## Current-Version Note

The attached report at `/Users/ConstiX/Downloads/Project Argus Final Report VFF Consti.docx` appears to be a shortened/current working copy. It has 86 text paragraphs and 6 tables, but its headings are stored as `normal` style paragraphs rather than Word heading styles. The repo copy `argus_final_report_v3.docx` is longer, includes a title/cover block, and has more front matter. Before submission, reconcile which file is final.

## High-Priority Feedback

1. The report's central quantitative result is good: demand weighting changes both coverage percentages and the selected physical portfolio. The strongest numbers are the 25-degree, 20-station comparison: uniform demand covers 20.9%, while WorldPop weighted demand covers 61.4%; the selected uniform and raster portfolios share zero sites.

2. The real-world validation section currently overstates the evidence. It says the WorldPop raster portfolio most closely matches real operators. But the FCC Kuiper gateway sample I checked is much closer to the uniform 20-site portfolio than to the WorldPop 20-site portfolio. The mean nearest-site distance from the sample is about 1,300 km for the uniform portfolio versus about 7,254 km for the WorldPop portfolio. That does not invalidate the WorldPop result; it means the report should distinguish global population-demand optimization from U.S.-licensed gateway deployment.

3. Section numbering has a visible gap: `2.2 Multi-Objective Scheduling` jumps to `2.4 Differentiation from Prior Work`. Add `2.3 Facility-Location Foundations`, or renumber.

4. The report defines many technical objects, but it defines them after using some of them rhetorically. Add a short "Model Objects and Units" table before Section 3, defining candidate sites, satellite-time rows, visibility arcs, demand weights, coverage, weighted coverage, and visibility upper bound.

5. Figure/table numbering needs cleanup. Table 2 is used both as a sensitivity table label and a caption sentence, Table 5 appears before Table 6 in the document flow after Table 4, and some captions appear without enough interpretation immediately after them. Make every figure/table caption unique and follow each one with one "what this means" paragraph.

6. The abstract says the WorldPop raster "most closely replicates real-world operator siting patterns." That claim should be softened unless the new operator comparison is added. Suggested rewrite: "The WorldPop raster model is the most defensible demand specification for population-service questions, while operator gateway deployments additionally reflect licensing, fiber, national rollout, and peering constraints."

7. The report should explicitly say what "coverage" means in each result. Uniform coverage is row coverage over satellite-time pairs. Population coverage is demand-weighted coverage. They are not the same unit, so the text should avoid comparing them as if they were identical physical service percentages.

8. The conclusion is logically close, but it repeats the overclaim that all three model predictions are confirmed by Starlink, Amazon Leo, and OneWeb. Replace "confirmed" with "consistent with" where the evidence is qualitative, and use "supported by our FCC gateway comparison" only where the data was actually measured.

## Suggested Logical Order

1. Motivation: why gateways are expensive fixed facilities and why visibility creates dynamic feasibility.
2. Glossary / model object table: define terms before equations.
3. Literature: satellite ground segment, facility location, multi-objective/Pareto framing.
4. Model: sets, variables, objective, constraints.
5. Data pipeline: TLEs, candidate sites, visibility tensor, demand models, backhaul mask.
6. Baseline results: uniform demand and sensitivity grid.
7. Demand-model results: city proxy and WorldPop raster.
8. Portfolio divergence and demand misspecification.
9. Real-world comparison: clearly separated as external validation or benchmarking, with source limitations.
10. Limitations and conclusion.

## Actual Results Meaning

The model is not saying "20 stations cover 61.4% of the Earth." It is saying that, within the 20-satellite, 4-hour prototype, the chosen stations can serve 61.4% of the WorldPop-weighted visible satellite-time demand under a 25-degree elevation floor and 20-station budget.

The elevation threshold is the harshest physical lever. At 0 degrees, WorldPop coverage reaches almost 100% with 20-30 stations. At 25 degrees, even 30 stations reach only 71.5%, despite an 88.4% visibility upper bound. That means stricter link geometry removes opportunities that station count alone cannot recover.

The demand model is the harshest managerial lever. Uniform demand spreads value across all satellite-time rows. WorldPop demand concentrates value over populated areas. That is why early stations are far more valuable under WorldPop: at 25 degrees, the first 5 stations cover 28.3% of weighted demand, compared with 7.4% under uniform demand.

The zero-overlap result means the project is about infrastructure allocation, not just metric choice. Changing the demand model changes where money gets spent.

The Kuiper comparison shows a useful tension. Public FCC Kuiper gateway sites are U.S.-heavy and licensing/fiber constrained. The Argus WorldPop solution is global and population-weighted, so it does not naturally land near the U.S. Kuiper sample. This should be framed as a limitation and a future-work opportunity: use verified operator coordinates as candidate sites or as a validation set.

## Glossary

- Backhaul: Terrestrial network connectivity from a gateway into fiber, internet exchange, cloud, or core network infrastructure.
- Backhaul mask: Binary parameter indicating whether a candidate site is allowed by the project's proxy backhaul rule.
- Budget `K`: Maximum number of ground stations the optimizer may open.
- Candidate site: A possible ground-station location considered by the optimizer.
- CAPEX: Capital expenditure required to build or acquire infrastructure.
- City-point proxy demand: Population-weighted demand model based on populated-place points and a spatial kernel.
- Coverage: Fraction of demand served by the selected stations; in this report it can mean row coverage or demand-weighted coverage.
- Coverage target epsilon: Minimum required coverage level in the epsilon-constraint Pareto sweep.
- Coverage upper bound / visibility UB: Best possible coverage if every visible demand row could be served, before station-budget limits bind.
- CSR matrix: Compressed Sparse Row matrix format used to store large sparse visibility and cost matrices efficiently.
- Demand node / row: A satellite-time pair `(j, t)` that may need service.
- Demand vector: Weights assigned to demand rows.
- Demand-weighted coverage: Covered demand divided by total demand after applying population or other weights.
- ECEF: Earth-centered, Earth-fixed coordinate frame used for satellite and ground positions.
- Elevation threshold: Minimum angle above the local horizon required for a valid satellite-ground link.
- Epsilon-constraint method: Pareto method that optimizes cost while imposing a required minimum coverage level.
- Facility-location problem: Optimization problem selecting which facilities to open and which demand to assign to them.
- Farthest-point sampling: Method for spreading candidate sites by iteratively choosing locations far from already selected ones.
- Fixed-charge facility location: Facility-location model with a fixed cost for opening each facility.
- Fixed opening cost `f_i`: Cost paid when candidate site `i` is opened.
- Gateway / ground station: Terrestrial antenna site that links satellites to the ground network.
- Gaussian kernel: Smooth distance-decay function used to spread city-point population influence over nearby demand rows.
- IXP: Internet exchange point where networks interconnect.
- Jaccard index: Overlap metric equal to shared selected sites divided by total unique selected sites.
- Latency-weighted service cost: Assignment cost based on slant range or propagation delay.
- LEO: Low-Earth Orbit, typically satellites orbiting much closer to Earth than geostationary satellites.
- Line of sight: Geometric visibility between a satellite and a ground station.
- Maximal Covering Location Problem: Classical model selecting limited facilities to maximize covered demand.
- MILP / MIP: Mixed-integer linear program / mixed-integer program with continuous and integer decision variables.
- NGSO: Non-geostationary orbit.
- OPEX: Operating expenditure required to run infrastructure.
- Pareto frontier: Set of efficient cost-coverage tradeoffs where improving coverage requires higher cost.
- Population proxy: Approximation of user demand based on population distribution.
- Portfolio: The set of selected ground-station sites.
- Row coverage: Fraction of satellite-time rows served, treating each row equally.
- SGP4: Standard orbit propagation model used with TLE data.
- Skyfield: Python astronomy/orbital mechanics library used to propagate satellite positions.
- Slant range: Direct line distance between a satellite and a ground station.
- Sparse visibility graph: Graph whose arcs exist only for feasible satellite-time/site visibility pairs.
- TLE: Two-line element record describing a satellite orbit.
- Uniform demand: Demand model where every satellite-time row receives equal weight.
- Visibility arc: Feasible link between a demand row and candidate site.
- Visibility tensor: Time-indexed satellite-site visibility structure, stored sparsely in the project.
- WGS84: Standard Earth ellipsoid/geodetic coordinate reference model.
- WorldPop raster demand: Demand model using gridded WorldPop population estimates to weight satellite-time rows.
