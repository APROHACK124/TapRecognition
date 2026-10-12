"""Stable, filename-based recording-type labels used in evaluation reports."""

from __future__ import annotations

from pathlib import Path
import re


_RECORDING_NAME = re.compile(
    r"^(?P<collection>imu(?:Strict)?)_(?P<person>[^_]+)_"
    r"(?P<action>.+)_(?P<timestamp>\d{8}_\d{6})"
    r"(?P<crop>__crop_f(?P<crop_start>\d+)_f(?P<crop_end>\d+))?$",
    re.IGNORECASE,
)


def recording_type_from_action(action: str) -> str:
    """Normalize a stored CSV action label to its report type."""
    normalized = str(action).strip().lower()
    if not normalized:
        raise ValueError("Recording action must not be empty")
    if normalized == "negative_mining":
        return "hard_negative"
    if normalized.startswith("knock_twice"):
        return "knock_twice"
    return normalized


def recording_type_from_filename(filename: str | Path, fallback_action: str | None = None) -> str:
    """Return a report type, grouping sides and mined negatives deliberately.

    Standard left/right double-tap collections share ``knock_twice``. Both a
    full ``negative_mining`` recording and an exported
    ``negative_mining_<timestamp>__crop_f<start>_f<end>.csv`` use
    ``hard_negative``. Crop suffixes on other collection actions are rejected
    rather than silently grouped under an unintended type. ``fallback_action``
    keeps ad-hoc or fixture CSVs usable in replay tables when they do not use
    the collection naming convention.
    """
    path = Path(filename)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected an IMU CSV filename: {filename}")
    match = _RECORDING_NAME.fullmatch(path.stem)
    if match is None:
        if fallback_action is not None:
            return recording_type_from_action(fallback_action)
        raise ValueError(f"Unexpected IMU recording filename: {filename}")
    fields = match.groupdict()
    action = fields["action"].lower()
    if fields["crop"] is not None:
        if action != "negative_mining":
            raise ValueError(f"Only negative_mining recordings may use a crop suffix: {filename}")
        if int(fields["crop_start"]) >= int(fields["crop_end"]):
            raise ValueError(f"Invalid hard-negative crop bounds: {filename}")
    return recording_type_from_action(action)
