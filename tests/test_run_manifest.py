"""Run manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.run_manifest import ManifestCommand, build_run_manifest, write_run_manifest


def test_build_and_write_run_manifest() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = build_run_manifest(
            run_name="test-run",
            project_root=root,
            commands=[ManifestCommand("step", "python step.py")],
            artifacts={"output": "artifact.txt"},
            parameters={"coverage": 0.5},
            notes=["proxy"],
        )
        output = write_run_manifest(manifest, root / "manifest.json")
        loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["run_name"] == "test-run"
    assert loaded["commands"][0]["name"] == "step"
    assert loaded["parameters"]["coverage"] == 0.5
