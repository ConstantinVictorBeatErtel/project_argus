"""Precompute script helpers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import load_npz

from scripts.build_visibility_tensor import apply_backhaul_mask_csv, build_visibility_outputs, load_candidates_csv
from src.simulation.visibility import geodetic_to_ecef_km


def test_load_candidates_csv_accepts_common_column_names() -> None:
    with TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "candidates.csv"
        csv_path.write_text(
            "id,lat,lon,alt_m,b_i,allowed\n"
            "oakland,37.8,-122.3,10,1,true\n"
            "blocked,38.0,-122.0,20,0,true\n",
            encoding="utf-8",
        )

        candidates = load_candidates_csv(csv_path)

    assert [candidate.site_id for candidate in candidates] == ["oakland", "blocked"]
    assert candidates[0].backhaul_feasible is True
    assert candidates[1].backhaul_feasible is False


def test_build_visibility_outputs_writes_canonical_npz_and_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        csv_path = root / "candidates.csv"
        visibility_path = root / "visibility.npz"
        range_path = root / "slant_range.npz"
        metadata_path = root / "visibility_metadata.json"

        csv_path.write_text("site_id,latitude_deg,longitude_deg\nsite-0,0.0,0.0\n", encoding="utf-8")
        candidates = load_candidates_csv(csv_path)

        site = geodetic_to_ecef_km([0.0], [0.0], [0.0])[0]
        positions = np.array([[[*(site + np.array([500.0, 0.0, 0.0]))]]], dtype=np.float32)

        build_visibility_outputs(
            positions_ecef_km=positions,
            candidates=candidates,
            min_elevation_deg=10.0,
            site_chunk_size=1,
            time_chunk_size=1,
            visibility_output=visibility_path,
            metadata_output=metadata_path,
            range_output=range_path,
        )

        visibility = load_npz(visibility_path)
        ranges = load_npz(range_path)

    assert visibility.shape == (1, 1)
    assert visibility.nnz == 1
    assert ranges.data.round(1).tolist() == [500.0]


def test_apply_backhaul_mask_csv_updates_candidate_feasibility() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        candidates_csv = root / "candidates.csv"
        mask_csv = root / "mask.csv"
        candidates_csv.write_text(
            "site_id,latitude_deg,longitude_deg\nsite-0,0,0\nsite-1,0,1\n",
            encoding="utf-8",
        )
        mask_csv.write_text("site_id,b_i\nsite-0,1\nsite-1,0\n", encoding="utf-8")

        candidates = apply_backhaul_mask_csv(load_candidates_csv(candidates_csv), mask_csv)

    assert [candidate.backhaul_feasible for candidate in candidates] == [True, False]
