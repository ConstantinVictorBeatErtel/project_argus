#!/usr/bin/env python3
"""Download CelesTrak TLE data into the raw data directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.propagator import TLERecord


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "parameters.yaml"
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
DEFAULT_USER_AGENT = "satellite-gs-optimizer/0.1"


@dataclass(frozen=True)
class TLEDownloadResult:
    """Summary of a TLE download."""

    output_path: str
    metadata_path: str
    source_url: str
    record_count: int
    sha256: str
    downloaded_at_utc: str


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def celestrak_group_url(group: str) -> str:
    """Build a CelesTrak GP endpoint URL for a satellite group."""

    if not group.strip():
        raise ValueError("group must be non-empty")
    return f"{CELESTRAK_GP_URL}?{urlencode({'GROUP': group.strip(), 'FORMAT': 'tle'})}"


def _ssl_context(*, verify_ssl: bool) -> ssl.SSLContext:
    if not verify_ssl:
        return ssl._create_unverified_context()  # noqa: SLF001 - explicit CLI escape hatch for broken local stores.

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_text(
    url: str,
    *,
    timeout_seconds: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    verify_ssl: bool = True,
) -> str:
    """Fetch a UTF-8-ish text payload over HTTP."""

    request = Request(url, headers={"User-Agent": user_agent})
    context = _ssl_context(verify_ssl=verify_ssl) if url.lower().startswith("https://") else None
    with urlopen(request, timeout=timeout_seconds, context=context) as response:  # noqa: S310 - user-controlled URL is intentional CLI input.
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def parse_tle_text(text: str, *, limit: int | None = None) -> list[TLERecord]:
    """Parse and validate two-line or three-line TLE text."""

    if limit is not None and limit <= 0:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[TLERecord] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("1 ") and idx + 1 < len(lines) and lines[idx + 1].startswith("2 "):
            line1 = lines[idx]
            line2 = lines[idx + 1]
            name = f"satellite_{len(records):05d}"
            idx += 2
        elif idx + 2 < len(lines) and lines[idx + 1].startswith("1 ") and lines[idx + 2].startswith("2 "):
            name = lines[idx]
            line1 = lines[idx + 1]
            line2 = lines[idx + 2]
            idx += 3
        else:
            raise ValueError(f"Invalid TLE payload near non-empty line {idx + 1}: {lines[idx]!r}")

        records.append(TLERecord(name=name, line1=line1, line2=line2))
        if limit is not None and len(records) >= limit:
            break

    if not records:
        raise ValueError("No TLE records found in payload")
    return records


def normalize_tle_text(records: list[TLERecord]) -> str:
    """Serialize TLE records as name/line1/line2 triplets."""

    return "\n".join(f"{record.name}\n{record.line1}\n{record.line2}" for record in records) + "\n"


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text via a same-directory temporary file and replace."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    return output


def write_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON metadata next to a downloaded TLE file."""

    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def download_tle(
    *,
    url: str,
    output_path: str | Path,
    metadata_path: str | Path,
    timeout_seconds: float = 30.0,
    limit: int | None = None,
    verify_ssl: bool = True,
) -> TLEDownloadResult:
    """Fetch, validate, normalize, and persist TLE data."""

    fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    raw_text = fetch_text(url, timeout_seconds=timeout_seconds, verify_ssl=verify_ssl)
    records = parse_tle_text(raw_text, limit=limit)
    normalized_text = normalize_tle_text(records)
    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    out = atomic_write_text(output_path, normalized_text)
    metadata = {
        "downloaded_at_utc": fetched_at,
        "record_count": len(records),
        "sha256": digest,
        "source_url": url,
    }
    meta = write_metadata(metadata_path, metadata)

    return TLEDownloadResult(
        output_path=str(out),
        metadata_path=str(meta),
        source_url=url,
        record_count=len(records),
        sha256=digest,
        downloaded_at_utc=fetched_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and validate CelesTrak TLE data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML parameter file.")
    parser.add_argument("--group", default="starlink", help="CelesTrak GP group, e.g. starlink or active.")
    parser.add_argument("--url", default=None, help="Explicit TLE URL. Overrides --group.")
    parser.add_argument("--output", type=Path, default=None, help="Output TLE text path.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Output metadata JSON path.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit after validation.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved URL and output paths without downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_parameters(args.config)
    paths = config["paths"]

    url = args.url or celestrak_group_url(args.group)
    output_path = args.output or PROJECT_ROOT / paths["tle_file"]
    metadata_path = args.metadata_output or PROJECT_ROOT / paths["tle_metadata"]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "source_url": url,
                    "output_path": str(output_path),
                    "metadata_path": str(metadata_path),
                    "dry_run": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = download_tle(
        url=url,
        output_path=output_path,
        metadata_path=metadata_path,
        timeout_seconds=args.timeout,
        limit=args.limit,
        verify_ssl=not args.insecure,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
