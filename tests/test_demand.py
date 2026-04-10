"""Demand model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.simulation.demand import (
    build_population_weighted_demand_frame,
    build_uniform_demand_frame,
    demand_vector_from_frame,
    ecef_to_geodetic_lat_lon,
    load_visibility_metadata,
    population_weighted_demand_vector,
    uniform_demand_vector,
    write_demand_outputs,
)


def test_uniform_demand_vector_matches_flattened_satellite_time_rows() -> None:
    demand = uniform_demand_vector(2, 3, weight=2.0)

    assert demand.tolist() == [2.0] * 6


def test_uniform_demand_can_be_normalized() -> None:
    demand = uniform_demand_vector(2, 3, normalize=True)

    assert np.isclose(float(demand.sum()), 1.0)


def test_demand_frame_round_trips_to_parquet_and_npy() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        frame = build_uniform_demand_frame(2, 2, weight=3.0)
        parquet_path, npy_path = write_demand_outputs(
            frame,
            parquet_path=root / "demand.parquet",
            npy_path=root / "demand.npy",
        )
        loaded_frame = pd.read_parquet(parquet_path)
        loaded_vector = np.load(npy_path)

    assert demand_vector_from_frame(loaded_frame).tolist() == [3.0, 3.0, 3.0, 3.0]
    assert loaded_vector.tolist() == [3.0, 3.0, 3.0, 3.0]


def test_load_visibility_metadata_dimensions() -> None:
    with TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "metadata.json"
        metadata_path.write_text(json.dumps({"num_satellites": 2, "num_times": 4}), encoding="utf-8")

        dimensions = load_visibility_metadata(metadata_path)

    assert dimensions == (2, 4)


def test_ecef_to_geodetic_lat_lon_equator() -> None:
    lat, lon = ecef_to_geodetic_lat_lon(np.array([[[6378.137, 0.0, 0.0]]]))

    assert np.isclose(float(lat[0, 0]), 0.0)
    assert np.isclose(float(lon[0, 0]), 0.0)


def test_population_weighted_demand_prioritizes_nearby_population() -> None:
    positions = np.array(
        [
            [
                [6378.137, 0.0, 0.0],
                [0.0, 6378.137, 0.0],
            ]
        ]
    )
    points = pd.DataFrame(
        {
            "latitude_deg": [0.0],
            "longitude_deg": [0.0],
            "population": [1_000_000.0],
        }
    )

    demand = population_weighted_demand_vector(
        positions,
        points,
        kernel_radius_km=500.0,
        floor_weight=0.01,
        normalize_total=2.0,
    )

    assert np.isclose(float(demand.sum()), 2.0)
    assert demand[0] > demand[1]


def test_build_population_weighted_demand_frame_shape() -> None:
    positions = np.array([[[6378.137, 0.0, 0.0]], [[0.0, 6378.137, 0.0]]])
    points = pd.DataFrame(
        {
            "latitude_deg": [0.0],
            "longitude_deg": [0.0],
            "population": [1_000_000.0],
        }
    )

    frame = build_population_weighted_demand_frame(positions, points, kernel_radius_km=500.0)

    assert frame.shape[0] == 2
    assert np.isclose(float(frame["demand"].sum()), 2.0)
