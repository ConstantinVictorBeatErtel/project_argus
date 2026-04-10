"""Sensitivity analysis visualization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def load_sensitivity_results(path: str | Path) -> pd.DataFrame:
    """Load sensitivity results from CSV."""

    return pd.read_csv(path)


def build_sensitivity_figure(results: pd.DataFrame) -> go.Figure:
    """Build an interactive station-budget sensitivity chart."""

    required = {
        "elevation_deg",
        "max_ground_stations",
        "status",
        "achieved_coverage",
        "selected_site_count",
        "row_visibility_upper_bound",
        "demand_visibility_upper_bound",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"sensitivity data missing required columns: {sorted(missing)}")

    fig = go.Figure()
    for elevation, frame in results.groupby("elevation_deg", sort=True):
        frame = frame.sort_values("max_ground_stations")
        fig.add_trace(
            go.Scatter(
                x=frame["max_ground_stations"],
                y=frame["achieved_coverage"],
                mode="lines+markers",
                name=f"{elevation:g} deg elevation",
                customdata=frame[
                    [
                        "selected_site_count",
                        "row_visibility_upper_bound",
                        "demand_visibility_upper_bound",
                        "status",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "budget=%{x}<br>"
                    "achieved coverage=%{y:.3f}<br>"
                    "selected sites=%{customdata[0]}<br>"
                    "row visibility bound=%{customdata[1]:.3f}<br>"
                    "demand visibility bound=%{customdata[2]:.3f}<br>"
                    "status=%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Coverage Sensitivity by Station Budget and Elevation Threshold",
        xaxis_title="Maximum ground stations",
        yaxis_title="Achieved coverage fraction",
        template="plotly_white",
        legend_title="Visibility threshold",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def write_sensitivity_html(results_csv: str | Path, output_path: str | Path) -> Path:
    """Write a sensitivity analysis HTML chart."""

    fig = build_sensitivity_figure(load_sensitivity_results(results_csv))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return out


__all__ = ["build_sensitivity_figure", "load_sensitivity_results", "write_sensitivity_html"]
