"""Selected-site map visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.graph_objects as go


def load_selected_site_frame(metadata_path: str | Path, result_path: str | Path) -> pd.DataFrame:
    """Load selected-site coordinates from visibility metadata and optimizer result JSON."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    selected = set(int(site) for site in result.get("selected_sites", []))
    sites = metadata["candidate_sites"]
    rows = []
    for site_index, site in enumerate(sites):
        rows.append(
            {
                "site_index": site_index,
                "site_id": site["site_id"],
                "latitude_deg": float(site["latitude_deg"]),
                "longitude_deg": float(site["longitude_deg"]),
                "selected": site_index in selected,
                "source": site.get("candidate_source", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def build_selected_site_map(site_frame: pd.DataFrame, *, title: str = "Selected Ground Station Sites") -> go.Figure:
    """Build an interactive global selected-site map."""

    required = {"site_index", "site_id", "latitude_deg", "longitude_deg", "selected"}
    missing = required - set(site_frame.columns)
    if missing:
        raise ValueError(f"site frame missing required columns: {sorted(missing)}")

    fig = go.Figure()
    for selected, label, size, opacity in [(False, "candidate", 6, 0.45), (True, "selected", 10, 0.95)]:
        frame = site_frame[site_frame["selected"] == selected]
        fig.add_trace(
            go.Scattergeo(
                lon=frame["longitude_deg"],
                lat=frame["latitude_deg"],
                text=[
                    f"{row.site_id}<br>index={row.site_index}<br>source={row.source}"
                    for row in frame.itertuples(index=False)
                ],
                mode="markers",
                name=label,
                marker={"size": size, "opacity": opacity},
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        geo={
            "projection_type": "natural earth",
            "showland": True,
            "landcolor": "rgb(235, 235, 235)",
            "showocean": True,
            "oceancolor": "rgb(225, 240, 250)",
            "showcountries": True,
        },
        template="plotly_white",
        legend_title="Site status",
    )
    return fig


def load_selection_comparison_frame(
    metadata_path: str | Path,
    *,
    left_selected_sites: Sequence[int],
    right_selected_sites: Sequence[int],
    left_label: str = "left",
    right_label: str = "right",
) -> pd.DataFrame:
    """Build a site-level comparison frame for two selected-site portfolios."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    left = {int(site) for site in left_selected_sites}
    right = {int(site) for site in right_selected_sites}
    rows = []
    for site_index, site in enumerate(metadata["candidate_sites"]):
        in_left = site_index in left
        in_right = site_index in right
        if in_left and in_right:
            status = "both"
        elif in_left:
            status = f"{left_label}_only"
        elif in_right:
            status = f"{right_label}_only"
        else:
            status = "candidate"
        rows.append(
            {
                "site_index": site_index,
                "site_id": site["site_id"],
                "latitude_deg": float(site["latitude_deg"]),
                "longitude_deg": float(site["longitude_deg"]),
                "left_selected": in_left,
                "right_selected": in_right,
                "comparison_status": status,
                "source": site.get("candidate_source", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def build_selection_comparison_map(
    comparison_frame: pd.DataFrame,
    *,
    left_label: str = "left",
    right_label: str = "right",
    title: str = "Selected Site Comparison",
) -> go.Figure:
    """Build an interactive map comparing two selected-site portfolios."""

    required = {"site_index", "site_id", "latitude_deg", "longitude_deg", "comparison_status"}
    missing = required - set(comparison_frame.columns)
    if missing:
        raise ValueError(f"comparison frame missing required columns: {sorted(missing)}")

    labels = {
        "candidate": ("candidate", 5, 0.25),
        f"{left_label}_only": (f"{left_label} only", 10, 0.95),
        f"{right_label}_only": (f"{right_label} only", 10, 0.95),
        "both": ("both", 13, 1.0),
    }
    fig = go.Figure()
    for status, (label, size, opacity) in labels.items():
        frame = comparison_frame[comparison_frame["comparison_status"] == status]
        if frame.empty:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=frame["longitude_deg"],
                lat=frame["latitude_deg"],
                text=[
                    f"{row.site_id}<br>index={row.site_index}<br>status={row.comparison_status}"
                    for row in frame.itertuples(index=False)
                ],
                mode="markers",
                name=label,
                marker={"size": size, "opacity": opacity},
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        geo={
            "projection_type": "natural earth",
            "showland": True,
            "landcolor": "rgb(235, 235, 235)",
            "showocean": True,
            "oceancolor": "rgb(225, 240, 250)",
            "showcountries": True,
        },
        template="plotly_white",
        legend_title="Selection status",
    )
    return fig


def write_selection_comparison_html(
    comparison_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    left_label: str = "left",
    right_label: str = "right",
    title: str = "Selected Site Comparison",
) -> Path:
    """Write a selected-site comparison map to HTML."""

    fig = build_selection_comparison_map(
        comparison_frame,
        left_label=left_label,
        right_label=right_label,
        title=title,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return out


def write_selected_site_map_html(
    metadata_path: str | Path,
    result_path: str | Path,
    output_path: str | Path,
    *,
    title: str = "Selected Ground Station Sites",
) -> Path:
    """Write selected-site map HTML."""

    fig = build_selected_site_map(load_selected_site_frame(metadata_path, result_path), title=title)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return out


__all__ = [
    "build_selected_site_map",
    "build_selection_comparison_map",
    "load_selected_site_frame",
    "load_selection_comparison_frame",
    "write_selected_site_map_html",
    "write_selection_comparison_html",
]
