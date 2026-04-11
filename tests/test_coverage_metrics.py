"""Coverage comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.evaluation.coverage_metrics import (
    build_coverage_comparison_frame,
    scenario_comparison_from_frame,
    write_phase2_json,
    write_phase2_markdown,
)


def test_build_coverage_comparison_frame() -> None:
    left = pd.DataFrame(
        {
            "elevation_deg": [25.0],
            "max_ground_stations": [20],
            "achieved_coverage": [0.2],
            "demand_visibility_upper_bound": [0.6],
        }
    )
    right = pd.DataFrame(
        {
            "elevation_deg": [25.0],
            "max_ground_stations": [20],
            "achieved_coverage": [0.5],
            "demand_visibility_upper_bound": [0.9],
        }
    )

    frame = build_coverage_comparison_frame(left, right, left_label="uniform", right_label="raster")

    assert frame.loc[0, "absolute_delta"] == 0.3
    assert frame.loc[0, "relative_multiplier"] == 2.5


def test_scenario_comparison_and_writers() -> None:
    frame = pd.DataFrame(
        {
            "elevation_deg": [25.0],
            "max_ground_stations": [20],
            "uniform_coverage": [0.2],
            "uniform_visibility_upper_bound": [0.6],
            "raster_coverage": [0.5],
            "raster_visibility_upper_bound": [0.9],
            "absolute_delta": [0.3],
            "relative_multiplier": [2.5],
        }
    )
    scenario = scenario_comparison_from_frame(
        frame,
        elevation_deg=25.0,
        max_ground_stations=20,
        left_label="uniform",
        right_label="raster",
    )

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        json_path = write_phase2_json(
            output_path=root / "phase2.json",
            target_scenario=scenario,
            site_comparison_summary={"overlap_count": 4, "jaccard_similarity": 0.1},
        )
        markdown_path = write_phase2_markdown(
            output_path=root / "phase2.md",
            comparison_frame=frame,
            target_scenario=scenario,
            site_comparison_summary={"overlap_count": 4, "jaccard_similarity": 0.1},
            left_label="uniform",
            right_label="raster",
        )

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")

    assert loaded["target_scenario"]["absolute_delta"] == 0.3
    assert "Phase 2 Comparison" in markdown
    assert "25 deg" in markdown
