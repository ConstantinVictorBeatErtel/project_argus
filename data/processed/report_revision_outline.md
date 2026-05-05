# Chronological Report Revision Outline

## 1. Update The Core Story In The Abstract

Change the abstract so it no longer says WorldPop simply replicates real-world operator siting. The cleaner story is:

> We began with demand-model sensitivity, then incorporated professor feedback by benchmarking against real-world gateway data. That comparison revealed a missing siting constraint: geopolitical and market-access feasibility. When added as a hard site-eligibility constraint, the WorldPop solution became much more consistent with Starlink's observed operational gateway network.

Use the following result sentence:

> Under the 25-degree, 20-station scenario, the geopolitical constraint reduces WorldPop weighted coverage from 61.4% to 43.3%, but also moves the optimized portfolio much closer to Starlink's operational gateway footprint, reducing median nearest-gateway distance from 7,154 km to 881 km.

## 2. Keep The New Section 2 Structure

The revised Section 2 structure is good:

- `2.1 Theoretical Foundations`
- `2.2 Related Work`
- `2.3 Contributions`

Do not reintroduce the old `2.4` differentiation subsection. The new structure defines the theoretical frame first, then explains prior work, then states what this project adds.

## 3. Add A Model Objects And Units Table Before Section 3

Insert a short table before the math formulation. This fixes the main logical-order issue: the report currently uses several terms before readers have a compact definition.

Recommended rows:

- Candidate site: possible gateway location indexed by `i`.
- Satellite-time row: demand node `(j,t)`.
- Visibility arc: feasible assignment from row `r` to site `i`.
- Uniform coverage: fraction of satellite-time rows served.
- Weighted coverage: fraction of demand weight served.
- Visibility upper bound: maximum coverage possible before site-budget limits.
- Backhaul feasibility `b_i`: whether a site can connect to terrestrial network infrastructure.
- Geopolitical feasibility `g_i`: whether a site is allowed under the policy/risk scenario.

## 4. Add The Geopolitical Constraint In The Formulation Section

In the constraints subsection, add:

> To incorporate geopolitical feasibility, we introduce a binary parameter `g_i`, equal to 1 when candidate site `i` lies in an allowed operating jurisdiction and 0 otherwise. Site opening is then constrained by both backhaul and geopolitical eligibility:
>
> `y_i <= b_i`
>
> `y_i <= g_i`
>
> Equivalently, the implementation passes the combined mask `site_feasible_i = b_i AND g_i` to the MILP.

Then state that the geopolitical scenario is deliberately conservative and represents a U.S.-operator siting assumption, not an immutable legal truth.

## 5. Update The Computational Pipeline

Add a pipeline step after the backhaul mask:

> Geopolitical Mask: classify candidate sites by coarse sanctions, licensing, and market-access exclusion zones. The scenario excludes Russia/Belarus, China, Iran, North Korea, Syria, Cuba, and Venezuela, removing 38 of 200 proxy candidate sites.

## 6. Present Baseline Sensitivity Results First

Keep the original three demand-model heat maps:

- Uniform demand
- City-point proxy
- WorldPop raster

The interpretation should be:

> Demand weighting is not just a reporting change. It changes the physical station portfolio and substantially increases measured service efficiency over populated demand.

## 7. Add A New Robustness Section After Demand Misspecification

Suggested section title:

> `5.6 Geopolitical Feasibility Robustness Check`

Explain why this section exists:

> After presentation feedback asked us to check additional real-world constraints and compare against observed operator deployments, we added a geopolitical site-eligibility constraint and reran the sensitivity grid.

Include the key table:

| Demand model | Baseline coverage | With geopolitical constraint | Change |
|---|---:|---:|---:|
| Uniform | 20.9% | 20.8% | -0.1 pp |
| Population proxy | 51.7% | 41.9% | -9.8 pp |
| WorldPop raster | 61.4% | 43.3% | -18.1 pp |

Interpretation:

> Uniform demand barely changes because it spreads value evenly across satellite-time rows. Population-weighted models change sharply because several high-value sites fall inside excluded geopolitical regions. This means geopolitical feasibility is not a minor implementation detail; it changes the capital allocation implied by the demand model.

## 8. Replace The Real-World Operator Section With A Benchmarking Section

Suggested section title:

> `6. Real-World Benchmarking: Starlink and Kuiper`

Do not present real-world data as pure validation. Present it as a benchmark that revealed a missing constraint.

Recommended structure:

1. Data sources and caveats.
2. Kuiper FCC gateway sample.
3. Starlink operational gateway dataset.
4. What the comparison revealed.

For Starlink, use this result:

> The operational Starlink gateway data include 250 operational gateways. The original WorldPop portfolio has median nearest-gateway distance of 7,154 km and only 7.6% of Starlink gateways within 1,000 km of a model-selected site. After imposing the geopolitical constraint, the WorldPop portfolio improves to a median nearest-gateway distance of 881 km, with 57.6% of Starlink gateways within 1,000 km.

For Kuiper, use the more cautious result:

> The FCC Kuiper sample is U.S.-heavy, so it is closer to the uniform model-selected portfolio than to the unconstrained WorldPop portfolio. This reinforces that operator siting reflects jurisdiction, licensing, and rollout strategy in addition to population demand.

## 9. Update The Limitations Section

Add:

> The geopolitical mask uses coarse bounding boxes rather than country polygons or legal review. It should be interpreted as a scenario test, not a definitive regulatory model. Future work should replace the coarse mask with country-level license status, sanctions lists, land ownership restrictions, spectrum rights, and verified gateway candidate parcels.

Also keep:

> The project uses a 20-satellite, 4-hour prototype, so results should be interpreted structurally rather than as a production Starlink prediction.

## 10. Rewrite The Conclusion Around The Stronger Story

The conclusion should say:

> The professor's feedback led to two direct extensions: testing an additional real-world constraint and comparing optimized portfolios to real gateway deployments. The comparison did not merely validate the original model; it revealed that geopolitical feasibility must be included for population-weighted solutions to resemble observed operator siting. With this added constraint, the report's main conclusion becomes stronger and more realistic: demand modeling determines where the model wants to build, while geopolitical and backhaul feasibility determine where an operator can build.

End with:

> The resulting framework is therefore not just a coverage optimizer. It is a constraint-aware capital allocation model for LEO ground infrastructure.

