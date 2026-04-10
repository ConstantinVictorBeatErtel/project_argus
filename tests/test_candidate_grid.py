"""Generated candidate-grid proxies."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.constraints.backhaul import proxy_backhaul_hubs, write_backhaul_points_csv
from src.simulation.candidates import (
    farthest_point_sample_latlon,
    generate_candidate_grid,
    points_in_boxes,
    write_candidate_grid_csv,
)


def test_rough_land_filter_excludes_central_pacific_point() -> None:
    mask = points_in_boxes([0.0, 40.0], [-150.0, -100.0])

    assert mask.tolist() == [False, True]


def test_generate_candidate_grid_can_downsample_deterministically() -> None:
    frame = generate_candidate_grid(latitude_step_deg=20.0, longitude_step_deg=20.0, max_sites=10)

    assert frame.shape[0] == 10
    assert {"site_id", "latitude_deg", "longitude_deg", "altitude_m"}.issubset(frame.columns)
    assert frame["site_id"].is_unique


def test_farthest_point_sampling_is_deterministic() -> None:
    selected = farthest_point_sample_latlon(
        latitude_deg=pd.Series([-10.0, 0.0, 10.0, 20.0]).to_numpy(),
        longitude_deg=pd.Series([0.0, 90.0, 180.0, -90.0]).to_numpy(),
        max_sites=3,
    )

    assert selected.tolist() == farthest_point_sample_latlon(
        latitude_deg=pd.Series([-10.0, 0.0, 10.0, 20.0]).to_numpy(),
        longitude_deg=pd.Series([0.0, 90.0, 180.0, -90.0]).to_numpy(),
        max_sites=3,
    ).tolist()


def test_write_candidate_and_proxy_backhaul_csvs() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        candidates_path = write_candidate_grid_csv(
            generate_candidate_grid(latitude_step_deg=30.0, longitude_step_deg=30.0, max_sites=5),
            root / "candidates.csv",
        )
        backhaul_path = write_backhaul_points_csv(proxy_backhaul_hubs()[:2], root / "backhaul.csv")
        candidates = pd.read_csv(candidates_path)
        backhaul = pd.read_csv(backhaul_path)

    assert candidates.shape[0] == 5
    assert backhaul["type"].tolist() == ["proxy_internet_hub", "proxy_internet_hub"]
