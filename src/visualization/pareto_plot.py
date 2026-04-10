"""Pareto frontier visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import plotly.graph_objects as go


def load_frontiers(frontier_paths: Mapping[str, str | Path]) -> pd.DataFrame:
    """Load one or more Pareto frontier CSVs into a tagged DataFrame."""

    frames: list[pd.DataFrame] = []
    for label, path in frontier_paths.items():
        frame = pd.read_csv(path)
        frame["frontier"] = label
        frames.append(frame)
    if not frames:
        raise ValueError("At least one frontier CSV is required")
    return pd.concat(frames, ignore_index=True)


def build_pareto_figure(frontiers: pd.DataFrame) -> go.Figure:
    """Build an interactive cost/coverage Pareto chart."""

    required = {"frontier", "coverage_target", "status", "achieved_coverage", "objective_value", "selected_site_count"}
    missing = required - set(frontiers.columns)
    if missing:
        raise ValueError(f"frontier data missing required columns: {sorted(missing)}")

    fig = go.Figure()
    for label, frame in frontiers.groupby("frontier", sort=False):
        optimal = frame[frame["status"] == "Optimal"].copy()
        infeasible = frame[frame["status"] != "Optimal"].copy()
        if not optimal.empty:
            fig.add_trace(
                go.Scatter(
                    x=optimal["achieved_coverage"],
                    y=optimal["objective_value"],
                    mode="lines+markers",
                    name=f"{label} optimal",
                    customdata=optimal[["coverage_target", "selected_site_count", "assignment_count"]].to_numpy(),
                    hovertemplate=(
                        "achieved coverage=%{x:.3f}<br>"
                        "objective=%{y:.3f}<br>"
                        "target=%{customdata[0]:.3f}<br>"
                        "sites=%{customdata[1]}<br>"
                        "assignments=%{customdata[2]}<extra></extra>"
                    ),
                )
            )
        if not infeasible.empty:
            fig.add_trace(
                go.Scatter(
                    x=infeasible["coverage_target"],
                    y=[None] * len(infeasible),
                    mode="markers",
                    name=f"{label} infeasible targets",
                    hovertext=[f"target={target:.3f} infeasible" for target in infeasible["coverage_target"]],
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        title="Coverage vs. Cost Frontier",
        xaxis_title="Achieved coverage fraction",
        yaxis_title="Objective value",
        template="plotly_white",
        legend_title="Run",
    )
    return fig


def write_pareto_html(frontier_paths: Mapping[str, str | Path], output_path: str | Path) -> Path:
    """Write a Pareto frontier HTML chart."""

    fig = build_pareto_figure(load_frontiers(frontier_paths))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return out


__all__ = ["build_pareto_figure", "load_frontiers", "write_pareto_html"]
