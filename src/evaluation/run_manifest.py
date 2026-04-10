"""Run manifest creation for processed optimization artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ManifestCommand:
    """A command used to produce a run artifact."""

    name: str
    command: str


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def build_run_manifest(
    *,
    run_name: str,
    project_root: str | Path,
    commands: Sequence[ManifestCommand],
    artifacts: dict[str, str],
    parameters: dict[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a pipeline run."""

    root = Path(project_root)
    return {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project_root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(root),
        "commands": [asdict(command) for command in commands],
        "artifacts": artifacts,
        "parameters": parameters or {},
        "notes": list(notes),
    }


def write_run_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a run manifest as JSON."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out


__all__ = ["ManifestCommand", "build_run_manifest", "write_run_manifest"]
