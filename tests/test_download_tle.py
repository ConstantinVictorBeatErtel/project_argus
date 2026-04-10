"""TLE download helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.download_tle import (
    atomic_write_text,
    celestrak_group_url,
    normalize_tle_text,
    parse_tle_text,
    write_metadata,
)


SAMPLE_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.00000000  .00016717  00000+0  10270-3 0  9000
2 25544  51.6416 236.3765 0006703 130.5360 325.0288 15.50000000  1000
"""


def test_celestrak_group_url_builds_gp_tle_endpoint() -> None:
    url = celestrak_group_url("starlink")

    assert url.startswith("https://celestrak.org/NORAD/elements/gp.php?")
    assert "GROUP=starlink" in url
    assert "FORMAT=tle" in url


def test_parse_and_normalize_tle_text() -> None:
    records = parse_tle_text(SAMPLE_TLE)
    normalized = normalize_tle_text(records)

    assert len(records) == 1
    assert records[0].name == "ISS (ZARYA)"
    assert normalized.endswith("\n")
    assert "1 25544U" in normalized
    assert "2 25544" in normalized


def test_atomic_write_and_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        output = atomic_write_text(Path(tmpdir) / "tles.txt", SAMPLE_TLE)
        metadata = write_metadata(Path(tmpdir) / "tles.metadata.json", {"record_count": 1})

        output_text = output.read_text(encoding="utf-8")
        payload = json.loads(metadata.read_text(encoding="utf-8"))

    assert output.name == "tles.txt"
    assert output_text == SAMPLE_TLE
    assert payload == {"record_count": 1}
