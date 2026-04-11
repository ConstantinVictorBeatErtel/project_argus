"""Coverage comparison utilities for Phase 2 evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ScenarioComparison:
    """Comparison of two demand models for one scenario."""

    elevation_deg: float
    max_ground_stations: int
    left_label: str
    right_label: str
    left_coverage: float
    right_coverage: float
    absolute_delta: float
    relative_multiplier: float | None
    left_visibility_upper_bound: float
    right_visibility_upper_bound: float


def build_coverage_comparison_frame(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    """Join two sensitivity tables and compute coverage deltas."""

    keys = ["elevation_deg", "max_ground_stations"]
    left = left_frame[keys + ["achieved_coverage", "demand_visibility_upper_bound"]].rename(
        columns={
            "achieved_coverage": f"{left_label}_coverage",
            "demand_visibility_upper_bound": f"{left_label}_visibility_upper_bound",
        }
    )
    right = right_frame[keys + ["achieved_coverage", "demand_visibility_upper_bound"]].rename(
        columns={
            "achieved_coverage": f"{right_label}_coverage",
            "demand_visibility_upper_bound": f"{right_label}_visibility_upper_bound",
        }
    )
    comparison = left.merge(right, on=keys, how="inner", validate="one_to_one")
    comparison["absolute_delta"] = comparison[f"{right_label}_coverage"] - comparison[f"{left_label}_coverage"]
    comparison["relative_multiplier"] = comparison[f"{right_label}_coverage"] / comparison[f"{left_label}_coverage"]
    return comparison.sort_values(keys).reset_index(drop=True)


def scenario_comparison_from_frame(
    frame: pd.DataFrame,
    *,
    elevation_deg: float,
    max_ground_stations: int,
    left_label: str,
    right_label: str,
) -> ScenarioComparison:
    """Extract a single scenario comparison from a joined comparison frame."""

    row = frame[
        (frame["elevation_deg"].astype(float) == float(elevation_deg))
        & (frame["max_ground_stations"].astype(int) == int(max_ground_stations))
    ]
    if row.empty:
        raise ValueError(f"No comparison row found for elevation={elevation_deg}, max_ground_stations={max_ground_stations}")
    scenario = row.iloc[0]
    return ScenarioComparison(
        elevation_deg=float(scenario["elevation_deg"]),
        max_ground_stations=int(scenario["max_ground_stations"]),
        left_label=left_label,
        right_label=right_label,
        left_coverage=float(scenario[f"{left_label}_coverage"]),
        right_coverage=float(scenario[f"{right_label}_coverage"]),
        absolute_delta=float(scenario["absolute_delta"]),
        relative_multiplier=None
        if not pd.notna(scenario["relative_multiplier"])
        else float(scenario["relative_multiplier"]),
        left_visibility_upper_bound=float(scenario[f"{left_label}_visibility_upper_bound"]),
        right_visibility_upper_bound=float(scenario[f"{right_label}_visibility_upper_bound"]),
    )


def write_phase2_markdown(
    *,
    output_path: str | Path,
    comparison_frame: pd.DataFrame,
    target_scenario: ScenarioComparison,
    site_comparison_summary: dict[str, Any],
    left_label: str,
    right_label: str,
) -> Path:
    """Write a compact markdown summary for the Phase 2 report."""

    pivot = comparison_frame.pivot(index="elevation_deg", columns="max_ground_stations", values=f"{right_label}_coverage")
    lines = [
        "# Phase 2 Comparison",
        "",
        f"- Baseline: `{left_label}`",
        f"- Phase 2 demand model: `{right_label}`",
        f"- Target scenario: elevation `{target_scenario.elevation_deg:g}` deg, budget `{target_scenario.max_ground_stations}`",
        "",
        "## Target Scenario",
        "",
        f"- `{left_label}` coverage: {target_scenario.left_coverage:.4f}",
        f"- `{right_label}` coverage: {target_scenario.right_coverage:.4f}",
        f"- absolute delta: {target_scenario.absolute_delta:.4f}",
        f"- relative multiplier: {target_scenario.relative_multiplier:.3f}" if target_scenario.relative_multiplier is not None else "- relative multiplier: n/a",
        f"- `{left_label}` visibility upper bound: {target_scenario.left_visibility_upper_bound:.4f}",
        f"- `{right_label}` visibility upper bound: {target_scenario.right_visibility_upper_bound:.4f}",
        f"- selected-site overlap count: {site_comparison_summary['overlap_count']}",
        f"- Jaccard similarity: {site_comparison_summary['jaccard_similarity']:.4f}",
        "",
        "## Critical Assessment",
        "",
        "- Higher demand-weighted coverage does not mean the physical geometry improved.",
        "- The right comparison is coverage relative to each model's demand-weighted visibility upper bound.",
        "- If the raster demand is highly concentrated, the optimizer can achieve much larger demand coverage than uniform row coverage under the same station budget.",
        "",
        "## Raster Coverage Table",
        "",
    ]
    header = "| Elevation | " + " | ".join(str(column) for column in pivot.columns.tolist()) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in pivot.columns.tolist()) + " |"
    lines.extend([header, divider])
    for elevation_deg, values in pivot.iterrows():
        formatted = " | ".join(f"{100.0 * float(value):.1f}%" for value in values.tolist())
        lines.append(f"| {float(elevation_deg):g} deg | {formatted} |")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_phase2_json(
    *,
    output_path: str | Path,
    target_scenario: ScenarioComparison,
    site_comparison_summary: dict[str, Any],
) -> Path:
    """Write the Phase 2 summary JSON."""

    payload = {
        "target_scenario": asdict(target_scenario),
        "site_comparison_summary": site_comparison_summary,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


__all__ = [
    "ScenarioComparison",
    "build_coverage_comparison_frame",
    "scenario_comparison_from_frame",
    "write_phase2_json",
    "write_phase2_markdown",
]
