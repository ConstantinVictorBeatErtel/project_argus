"""SGP4 orbit propagation utilities backed by Skyfield.

The optimizer consumes satellite positions in the Earth-fixed ITRS/ECEF frame.
This module keeps Skyfield as a lazy runtime dependency so configuration and
downstream sparse-geometry code can still import in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - exercised only when Skyfield is installed.
    from skyfield.api import EarthSatellite, load
    from skyfield.framelib import itrs
except ImportError:  # pragma: no cover - local CI may not install Skyfield.
    EarthSatellite = None  # type: ignore[assignment]
    load = None  # type: ignore[assignment]
    itrs = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TLERecord:
    """A single two-line element record."""

    name: str
    line1: str
    line2: str


@dataclass(frozen=True)
class PropagationResult:
    """Earth-fixed position output for a batch of satellites.

    Attributes:
        satellite_ids: Satellite names in the first axis order of ``ecef_km``.
        epochs_utc: UTC epochs in the second axis order of ``ecef_km``.
        ecef_km: Position tensor with shape ``(num_satellites, num_times, 3)``.
    """

    satellite_ids: tuple[str, ...]
    epochs_utc: tuple[datetime, ...]
    ecef_km: NDArray[np.float32]


def _require_skyfield() -> None:
    if EarthSatellite is None or load is None or itrs is None:
        raise ImportError(
            "Skyfield is required for SGP4 propagation. Install it with "
            "`pip install skyfield` or provide precomputed ECEF positions."
        )


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def build_time_grid(
    start_utc: datetime,
    duration: timedelta,
    step_seconds: int,
    *,
    include_endpoint: bool = False,
) -> tuple[datetime, ...]:
    """Build an evenly spaced UTC time grid for propagation.

    Args:
        start_utc: First propagation epoch. Naive datetimes are interpreted as
            UTC to avoid accidental local-time drift.
        duration: Propagation window length.
        step_seconds: Positive spacing between epochs.
        include_endpoint: Whether to include ``start_utc + duration`` when it
            falls exactly on the grid.

    Returns:
        Tuple of UTC datetimes.
    """

    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if duration.total_seconds() <= 0:
        raise ValueError("duration must be positive")

    start = ensure_utc(start_utc)
    whole_steps = int(duration.total_seconds() // step_seconds)
    n_steps = whole_steps + (1 if include_endpoint else 0)
    if not include_endpoint:
        n_steps = max(1, whole_steps)

    return tuple(start + timedelta(seconds=step_seconds * idx) for idx in range(n_steps))


def load_tles(path: str | Path, *, limit: int | None = None) -> list[TLERecord]:
    """Parse two-line or three-line TLE records from a text file.

    The parser accepts either ``name, line1, line2`` triplets or bare ``line1,
    line2`` pairs. Blank lines are ignored.
    """

    if limit is not None and limit <= 0:
        return []

    tle_path = Path(path)
    lines = [line.strip() for line in tle_path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]

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
            raise ValueError(f"Invalid TLE record starting at non-empty line {idx + 1}: {lines[idx]!r}")

        records.append(TLERecord(name=name, line1=line1, line2=line2))
        if limit is not None and len(records) >= limit:
            break

    return records


def propagate_tles(
    tles: Sequence[TLERecord],
    epochs_utc: Sequence[datetime],
    *,
    dtype: np.dtype[np.float32] = np.dtype("float32"),
) -> PropagationResult:
    """Propagate TLEs to ECEF/ITRS positions in kilometers.

    Args:
        tles: TLE records to propagate.
        epochs_utc: UTC epochs at which to sample each orbit.
        dtype: Floating dtype used for the returned tensor. ``float32`` is the
            default because visibility geometry is memory-bound at scale.

    Returns:
        ``PropagationResult`` with shape ``(len(tles), len(epochs_utc), 3)``.
    """

    _require_skyfield()
    if len(tles) == 0:
        raise ValueError("At least one TLE record is required")
    if len(epochs_utc) == 0:
        raise ValueError("At least one epoch is required")

    epochs = tuple(ensure_utc(epoch) for epoch in epochs_utc)
    timescale = load.timescale()  # type: ignore[union-attr]
    times = timescale.from_datetimes(epochs)

    positions = np.empty((len(tles), len(epochs), 3), dtype=dtype)
    for sat_idx, record in enumerate(tles):
        satellite = EarthSatellite(record.line1, record.line2, record.name, timescale)  # type: ignore[operator]
        xyz = np.asarray(satellite.at(times).frame_xyz(itrs).km, dtype=dtype)  # type: ignore[arg-type]
        if xyz.ndim == 1:
            xyz = xyz.reshape(3, 1)
        positions[sat_idx, :, :] = xyz.T

    return PropagationResult(
        satellite_ids=tuple(record.name for record in tles),
        epochs_utc=epochs,
        ecef_km=positions,
    )


def propagate_tle_file(
    tle_path: str | Path,
    start_utc: datetime,
    duration: timedelta,
    step_seconds: int,
    *,
    satellite_limit: int | None = None,
    include_endpoint: bool = False,
    dtype: np.dtype[np.float32] = np.dtype("float32"),
) -> PropagationResult:
    """Load TLEs from disk and propagate them over a regular time grid."""

    tles = load_tles(tle_path, limit=satellite_limit)
    epochs = build_time_grid(
        start_utc=start_utc,
        duration=duration,
        step_seconds=step_seconds,
        include_endpoint=include_endpoint,
    )
    return propagate_tles(tles=tles, epochs_utc=epochs, dtype=dtype)


def write_ecef_hdf5(
    result: PropagationResult,
    output_path: str | Path,
    *,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> Path:
    """Persist propagated ECEF positions to HDF5.

    Datasets:
        ``ecef_km``: ``(num_satellites, num_times, 3)`` float tensor.
        ``satellite_ids``: UTF-8 satellite names.
        ``epochs_utc``: ISO-8601 UTC timestamps.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional h5py.
        raise ImportError("h5py is required for HDF5 export. Install it with `pip install h5py`.") from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as handle:
        handle.attrs["frame"] = "ITRS/ECEF"
        handle.attrs["units"] = "km"
        handle.create_dataset(
            "ecef_km",
            data=result.ecef_km,
            compression=compression,
            compression_opts=compression_opts if compression else None,
            chunks=(1, min(max(result.ecef_km.shape[1], 1), 256), 3),
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("satellite_ids", data=np.asarray(result.satellite_ids, dtype=object), dtype=string_dtype)
        handle.create_dataset(
            "epochs_utc",
            data=np.asarray([epoch.isoformat().replace("+00:00", "Z") for epoch in result.epochs_utc], dtype=object),
            dtype=string_dtype,
        )

    return out


def iter_time_blocks(epochs_utc: Sequence[datetime], block_size: int) -> Iterable[tuple[int, tuple[datetime, ...]]]:
    """Yield ``(start_index, epochs)`` blocks for streaming propagation workflows."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    for start in range(0, len(epochs_utc), block_size):
        yield start, tuple(epochs_utc[start : start + block_size])
