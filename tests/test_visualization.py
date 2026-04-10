"""Visualization data loading and figure construction."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.visualization.coverage_maps import (
    build_selected_site_map,
    build_selection_comparison_map,
    load_selected_site_frame,
    load_selection_comparison_frame,
)
from src.visualization.pareto_plot import build_pareto_figure, load_frontiers
from src.visualization.sensitivity_plot import build_sensitivity_figure


def test_build_pareto_figure_from_frontier_csv() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "frontier.csv"
        pd.DataFrame(
            {
                "coverage_target": [0.1, 0.2],
                "status": ["Optimal", "Infeasible"],
                "achieved_coverage": [0.1, 0.0],
                "objective_value": [2.0, None],
                "selected_site_count": [2, 0],
                "assignment_count": [10, 0],
            }
        ).to_csv(path, index=False)
        frame = load_frontiers({"test": path})
        fig = build_pareto_figure(frame)

    assert len(fig.data) == 2


def test_build_selected_site_map_from_metadata_and_result() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        metadata = root / "metadata.json"
        result = root / "result.json"
        metadata.write_text(
            json.dumps(
                {
                    "candidate_sites": [
                        {"site_id": "a", "latitude_deg": 0.0, "longitude_deg": 0.0, "candidate_source": "proxy"},
                        {"site_id": "b", "latitude_deg": 10.0, "longitude_deg": 20.0, "candidate_source": "proxy"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        result.write_text(json.dumps({"selected_sites": [1]}), encoding="utf-8")
        frame = load_selected_site_frame(metadata, result)
        fig = build_selected_site_map(frame)

    assert frame["selected"].tolist() == [False, True]
    assert len(fig.data) == 2


def test_build_selection_comparison_map() -> None:
    with TemporaryDirectory() as tmpdir:
        metadata = Path(tmpdir) / "metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "candidate_sites": [
                        {"site_id": "a", "latitude_deg": 0.0, "longitude_deg": 0.0},
                        {"site_id": "b", "latitude_deg": 10.0, "longitude_deg": 20.0},
                        {"site_id": "c", "latitude_deg": -10.0, "longitude_deg": -20.0},
                    ]
                }
            ),
            encoding="utf-8",
        )
        frame = load_selection_comparison_frame(
            metadata,
            left_selected_sites=[0, 1],
            right_selected_sites=[1, 2],
            left_label="uniform",
            right_label="population",
        )
        fig = build_selection_comparison_map(frame, left_label="uniform", right_label="population")

    assert frame["comparison_status"].tolist() == ["uniform_only", "both", "population_only"]
    assert len(fig.data) == 3


def test_build_sensitivity_figure() -> None:
    frame = pd.DataFrame(
        {
            "elevation_deg": [0.0, 0.0, 25.0, 25.0],
            "max_ground_stations": [5, 10, 5, 10],
            "status": ["Optimal", "Optimal", "Optimal", "Optimal"],
            "achieved_coverage": [0.3, 0.5, 0.1, 0.2],
            "selected_site_count": [5, 10, 5, 10],
            "row_visibility_upper_bound": [0.9, 0.9, 0.6, 0.6],
            "demand_visibility_upper_bound": [0.95, 0.95, 0.7, 0.7],
        }
    )

    fig = build_sensitivity_figure(frame)

    assert len(fig.data) == 2
