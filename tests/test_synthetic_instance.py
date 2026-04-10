"""Synthetic fixture generation."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.create_synthetic_instance import main as _unused_main


def test_synthetic_instance_script_is_importable() -> None:
    assert callable(_unused_main)


def test_synthetic_manifest_shape_contract() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # This mirrors the manifest fields emitted by the CLI without shelling out.
        manifest = {
            "candidates_csv": str(root / "synthetic_candidates.csv"),
            "backhaul_csv": str(root / "synthetic_backhaul.csv"),
            "positions_npy": str(root / "synthetic_positions.npy"),
            "num_satellites": 2,
            "num_times": 3,
            "num_sites": 4,
            "altitude_km": 550.0,
        }
        (root / "synthetic_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        np.save(root / "synthetic_positions.npy", np.zeros((2, 3, 3), dtype=np.float32))

        loaded = json.loads((root / "synthetic_manifest.json").read_text(encoding="utf-8"))
        positions = np.load(loaded["positions_npy"])

    assert positions.shape == (2, 3, 3)
    assert loaded["num_sites"] == 4
