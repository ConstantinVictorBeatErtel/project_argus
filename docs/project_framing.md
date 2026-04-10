# Project Framing

## Current Decision

Keep the existing architecture and codebase. The project does not need a technical restart.

The main update is narrative: this should be presented as a supply-chain network design and facility-location project for satellite ground-segment infrastructure, not as a promise to build a complete operational aerospace planning system.

## Class-Facing Research Question

How do candidate-site density, ground-station budget, backhaul feasibility, and satellite visibility constraints shape the cost-coverage frontier for a LEO ground-station network?

## Supply-Chain Interpretation

This project maps cleanly onto classic supply-chain network design:

- Facilities: candidate ground-station sites.
- Demand nodes: satellite-time pairs that require service.
- Assignment arcs: feasible satellite-time-to-site visibility links.
- Facility-opening cost: fixed CAPEX/OPEX proxy for opening a ground station.
- Service cost: latency-weighted propagation cost on visible assignment arcs.
- Feasibility constraints: line-of-sight visibility, elevation threshold, backhaul availability, station-count budget.
- Output: selected ground-station sites and a Pareto frontier showing the cost of higher coverage.

The spatial/orbital pieces create the feasibility graph, but the main decision problem is still a fixed-charge covering and assignment model.

## What Changes

- We stop treating 90% coverage as the headline requirement for the proxy experiment.
- We describe coverage as an endogenous result of the scenario assumptions.
- We report the feasible frontier under each candidate set, elevation threshold, and station budget.
- We clearly label generated candidate sites and proxy backhaul hubs as Phase 1 proxy data.
- We tie the project directly to the facility-location and satellite ground-station optimization literature.

## What Does Not Change

- The sparse visibility tensor remains the core simulation artifact.
- The MILP remains the core optimization model.
- The Pareto sweep remains the central decision artifact.
- The proxy pipeline remains useful for Phase 1.
- Real GPW demand, authoritative candidate locations, and known-gateway validation remain extensions rather than blockers.

## Current Evidence From The Repo

The current Phase 1 proxy runs already support the updated story:

- A 50-site proxy run found an optimal solution with 18 selected sites and 14.0625% coverage under a 25-degree elevation threshold.
- A 200-site proxy comparison increased available visibility and found an optimal 20% coverage solution with 19 selected sites.
- Higher coverage targets became infeasible under the current station-count budget and visibility threshold, which is exactly the kind of tradeoff the class project should analyze.

These results are not a failure of the model. They are evidence that the optimizer is exposing the cost and feasibility structure of the chosen network design assumptions.

## Updated Milestones

1. Phase 1: Proxy network design experiment
   Build sparse visibility, solve small MILP instances, run Pareto sweeps, and generate visualizations.

2. Phase 2: Stronger project narrative
   Add related-work support, report outline, and a clean explanation of assumptions, variables, constraints, and tradeoffs.

3. Phase 3: Demand and validation upgrades
   Add GPW or a simpler population proxy, compare against uniform demand, and benchmark geography against known gateway clusters if an authoritative dataset is available.

4. Phase 4: Scaling and robustness
   Extend the time horizon, add rolling horizon if necessary, and run sensitivity cases for station failures, elevation thresholds, and station budgets.

## Recommended Report Claim

This project develops and evaluates a mixed-integer facility-location model for LEO ground-station placement. A sparse orbital visibility graph defines feasible service arcs between satellite-time demand nodes and candidate ground-station facilities. The optimization chooses a budgeted set of stations and assignments to trace the cost-coverage frontier under visibility, latency, and backhaul feasibility constraints.
