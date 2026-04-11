# Phase 2 Comparison

- Baseline: `uniform`
- Phase 2 demand model: `raster`
- Target scenario: elevation `25` deg, budget `20`

## Target Scenario

- `uniform` coverage: 0.2094
- `raster` coverage: 0.6144
- absolute delta: 0.4051
- relative multiplier: 2.935
- `uniform` visibility upper bound: 0.5969
- `raster` visibility upper bound: 0.8838
- selected-site overlap count: 0
- Jaccard similarity: 0.0000

## Critical Assessment

- Higher demand-weighted coverage does not mean the physical geometry improved.
- The right comparison is coverage relative to each model's demand-weighted visibility upper bound.
- If the raster demand is highly concentrated, the optimizer can achieve much larger demand coverage than uniform row coverage under the same station budget.

## Raster Coverage Table

| Elevation | 5 | 10 | 15 | 20 | 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 deg | 73.8% | 94.2% | 99.1% | 99.9% | 100.0% |
| 10 deg | 53.7% | 72.3% | 85.2% | 91.6% | 98.4% |
| 25 deg | 28.3% | 44.0% | 54.1% | 61.4% | 71.5% |
