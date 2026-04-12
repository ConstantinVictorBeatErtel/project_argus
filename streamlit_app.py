"""Streamlit app for the 200-site selected ground-station portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.visualization.coverage_maps import build_selected_site_map, load_selected_site_frame

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_PATH = DATA_DIR / "visibility_metadata_200.json"
RESULT_PATH = DATA_DIR / "optimization_result_200.json"
HTML_EXPORT_PATH = DATA_DIR / "selected_sites_200.html"


@st.cache_data(show_spinner=False)
def load_portfolio_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Load the selected-site portfolio and derived tables for the app."""

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    site_frame = load_selected_site_frame(METADATA_PATH, RESULT_PATH)
    assignments = pd.DataFrame(result.get("assignments", []))
    if assignments.empty:
        selected_frame = site_frame[site_frame["selected"]].copy()
        selected_frame["assignment_count"] = 0
        selected_frame["avg_service_cost"] = pd.NA
    else:
        site_stats = (
            assignments.groupby("site_index", as_index=False)
            .agg(
                assignment_count=("satellite_time_row", "count"),
                avg_service_cost=("service_cost", "mean"),
            )
            .sort_values(["assignment_count", "avg_service_cost"], ascending=[False, True])
        )
        selected_frame = site_frame[site_frame["selected"]].merge(site_stats, on="site_index", how="left")
        selected_frame["assignment_count"] = selected_frame["assignment_count"].fillna(0).astype(int)

    selected_frame = selected_frame.sort_values(
        ["assignment_count", "site_id"], ascending=[False, True]
    ).reset_index(drop=True)
    return site_frame, selected_frame, metadata, result


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    st.set_page_config(page_title="Project Argus Site Map", layout="wide")

    st.title("Project Argus: 200-Site Selection")
    st.caption(
        "Streamlit view of the selected-site portfolio previously exported to "
        "`data/processed/selected_sites_200.html`."
    )

    if not METADATA_PATH.exists() or not RESULT_PATH.exists():
        st.error(
            "Expected processed artifacts are missing. "
            f"Look for `{METADATA_PATH}` and `{RESULT_PATH}`."
        )
        st.stop()

    site_frame, selected_frame, metadata, result = load_portfolio_data()
    figure = build_selected_site_map(site_frame, title="200-site proxy selected sites")
    figure.update_layout(height=650, margin={"l": 0, "r": 0, "t": 60, "b": 0})

    metrics = st.columns(5)
    metrics[0].metric("Candidates", f"{metadata['num_sites']}")
    metrics[1].metric("Selected", f"{len(result['selected_sites'])}")
    metrics[2].metric("Coverage", format_percent(float(result["coverage_fraction"])))
    metrics[3].metric("Assignments", f"{result['num_assignments']}")
    metrics[4].metric("Objective", f"{float(result['objective_value']):.2f}")

    map_col, detail_col = st.columns([2.2, 1.1], gap="large")

    with map_col:
        st.plotly_chart(figure, use_container_width=True)
        if HTML_EXPORT_PATH.exists():
            st.download_button(
                "Download original HTML export",
                data=HTML_EXPORT_PATH.read_text(encoding="utf-8"),
                file_name=HTML_EXPORT_PATH.name,
                mime="text/html",
            )

    with detail_col:
        st.subheader("Run Summary")
        st.write(f"Optimization status: `{result['status']}`")
        st.write(f"Minimum elevation: `{metadata['min_elevation_deg']:.1f} deg`")
        st.write(f"Satellite count: `{metadata['num_satellites']}`")
        st.write(f"Time steps: `{metadata['num_times']}`")
        st.write(f"Covered demand: `{float(result['covered_demand']):.0f} / {float(result['total_demand']):.0f}`")

        st.subheader("Selected Sites")
        display_frame = selected_frame[
            [
                "site_id",
                "site_index",
                "latitude_deg",
                "longitude_deg",
                "assignment_count",
                "avg_service_cost",
            ]
        ].rename(
            columns={
                "site_id": "Site ID",
                "site_index": "Index",
                "latitude_deg": "Latitude",
                "longitude_deg": "Longitude",
                "assignment_count": "Assignments",
                "avg_service_cost": "Avg service cost",
            }
        )
        st.dataframe(
            display_frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Latitude": st.column_config.NumberColumn(format="%.1f deg"),
                "Longitude": st.column_config.NumberColumn(format="%.1f deg"),
                "Avg service cost": st.column_config.NumberColumn(format="%.4f"),
            },
        )


if __name__ == "__main__":
    main()
