# Related Work

This file keeps the research backbone for the class report. The goal is not to cite everything about satellites. The goal is to cite the papers that directly justify our modeling choices.

## Core Satellite Ground-Segment Papers

### Abe et al., 2024: gateway placement and routing

Link: [Optimizing Satellite Network Infrastructure: A Joint Approach to Gateway Placement and Routing](https://arxiv.org/abs/2405.01149)

What it contributes:

- Models satellite gateway placement as a mixed-integer optimization problem.
- Combines gateway-count cost, traffic allocation, and latency in the objective.
- Shows the tradeoff between fewer gateways and increased network latency.

How it maps to our project:

- Our selected-site variables `y_i` are the gateway/facility-opening decision.
- Our assignment variables `x_ri` are the simplified traffic allocation decision from satellite-time demand rows to ground stations.
- Our latency-weighted service-cost matrix is the same modeling instinct as their cost/latency tradeoff, but without full inter-satellite routing.

### Eddy, Ho, and Kochenderfer, 2025: LEO ground-station selection

Link: [Optimal Ground Station Selection for Low-Earth Orbiting Satellites](https://arxiv.org/abs/2410.16282)

What it contributes:

- Formulates LEO ground-station selection as an integer programming problem.
- Includes mission-level performance variables such as downlink, cost, recurring operational cost, and communications time-gap.
- Addresses scaling by solving over a reduced time domain.

How it maps to our project:

- Our Phase 1 small-horizon experiment follows the same idea of validating on a reduced time domain before scaling.
- Our coverage target is a proxy for mission performance.
- Our future rolling-horizon module is a natural extension of their time-domain reduction idea.

### del Portillo, Cameron, and Crawley, 2018: ground-segment architectures

Link: [Ground Segment Architectures for Large LEO Constellations with Feeder Links in EHF-bands](https://systemarchitect.mit.edu/wp-content/uploads/2025/02/delportillo18a.pdf)

What it contributes:

- Studies large-constellation ground-segment architecture.
- Optimizes ground-station locations subject to capacity, quality-of-service, and link-availability concerns.
- Treats ground-site count and system performance as architectural tradeoffs.

How it maps to our project:

- Our candidate-site selection and station-count budget mirror their architecture tradeoff.
- Our elevation-filtered visibility tensor is a simpler version of link availability.
- Our Pareto frontier is the decision artifact for the same type of ground-segment planning question.

## Multi-Objective Scheduling And Network Optimization

### Petelin, Antoniou, and Papa, 2021/2023: multi-objective ground-station scheduling

Link: [Multi-objective approaches to ground station scheduling for optimization of communication with satellites](https://link.springer.com/article/10.1007/s11081-021-09617-z)

What it contributes:

- Frames ground-station scheduling as a multi-objective optimization problem.
- Emphasizes Pareto-optimal alternatives rather than a single weighted objective.
- Models visibility/access windows as the basic feasibility object for satellite-ground communication.

How it maps to our project:

- Our epsilon-constraint Pareto sweep follows the same decision-maker logic: show tradeoffs instead of hiding them in one weighted score.
- Our visibility matrix is the feasibility layer that scheduling and facility-location decisions both depend on.

### Cheung and Lee, 2012/2013: MIP and heuristic scheduling for space communication networks

Link: [Mixed Integer Programming and Heuristic Scheduling for Space Communication Networks](https://ntrs.nasa.gov/citations/20130001861)

What it contributes:

- Uses mixed-integer programming and heuristic optimization for space communication network scheduling.
- Accounts for dynamic link performance and mission/operations requirements.
- Demonstrates a space-network planning model with 20 spacecraft and 3 ground stations.

How it maps to our project:

- It supports our use of MIP for space-communication network planning.
- It motivates future heuristic fallbacks and rolling-horizon decomposition for larger runs.
- Its dynamic-link-performance framing aligns with our time-indexed visibility tensor.

## Facility-Location Foundation

### Church and ReVelle, 1974: maximal covering location problem

Link: [The Maximal Covering Location Problem](https://doi.org/10.1111/j.1435-5597.1974.tb00902.x)

What it contributes:

- Introduces the classic maximum-coverage location model.
- Provides the operations-research foundation for selecting a limited number of facilities to cover the most demand.

How it maps to our project:

- Our coverage-constrained MILP is a time-expanded, satellite-specific covering problem.
- The ground-station budget `max_ground_stations` plays the role of the facility limit.
- Satellite-time demand rows replace ordinary geographic demand points.

### Murray, 2016: maximal coverage location problem overview

Link: [Maximal Coverage Location Problem: Impacts, Significance, and Evolution](https://journals.sagepub.com/doi/10.1177/0160017615600222)

What it contributes:

- Reviews the evolution and practical significance of the maximal covering location problem.
- Reinforces why coverage models are a standard tool for spatial resource placement.

How it maps to our project:

- This is the class-friendly bridge from the satellite setting back to supply-chain and spatial network design.
- It supports the claim that our model is a facility-location project with a specialized feasibility graph, not merely an orbital simulation.

## Literature Positioning

Our project sits between two literatures:

- Satellite ground-segment optimization: gateway placement, visibility windows, latency, and time-indexed link performance.
- Supply-chain network design: fixed-charge facility location, maximum coverage, budgeted site selection, and Pareto tradeoff analysis.

The differentiator is the visibility tensor. Instead of assuming a static distance-based coverage radius, the feasible assignment arcs are generated from orbital mechanics and stored sparsely.
