"""Backhaul feasibility filters."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.constraints.backhaul import (
    compute_backhaul_mask,
    compute_backhaul_mask_from_points,
    haversine_distance_km,
    load_backhaul_points_csv,
    write_backhaul_mask_csv,
)


def test_haversine_distance_is_reasonable_for_one_degree_longitude_at_equator() -> None:
    distance = haversine_distance_km([0.0], [0.0], [0.0], [1.0])

    assert distance.shape == (1, 1)
    assert 111.0 <= float(distance[0, 0]) <= 112.0


def test_compute_backhaul_mask_flags_nearby_sites() -> None:
    mask = compute_backhaul_mask(
        [0.0, 0.0],
        [0.0, 5.0],
        [0.0],
        [0.0],
        max_distance_km=100.0,
    )

    assert mask.feasible.tolist() == [True, False]
    assert mask.nearest_point_index.tolist() == [0, 0]
    assert mask.nearest_distance_km[0] == 0.0


def test_load_points_and_write_backhaul_mask_csv() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        points_csv = root / "points.csv"
        output_csv = root / "mask.csv"
        points_csv.write_text("id,lat,lon,type\nixp-0,0,0,ixp\n", encoding="utf-8")

        points = load_backhaul_points_csv(points_csv)
        mask = compute_backhaul_mask_from_points([0.0], [0.0], points, max_distance_km=10.0)
        output = write_backhaul_mask_csv(output_csv, site_ids=["site-0"], mask=mask, backhaul_points=points)
        frame = pd.read_csv(output)

    assert frame["site_id"].tolist() == ["site-0"]
    assert frame["b_i"].tolist() == [1]
    assert frame["nearest_backhaul_id"].tolist() == ["ixp-0"]
