"""Load IMU recordings and persist human annotations."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .labels import gaussian_second_tap_labels

IMU_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
SegmentKind = Literal["double_tap", "single_tap", "background", "invalid", "unlabeled"]


@dataclass
class LabelParams:
    sigma: float = 0.04
    peak_offset: float = 0.02


@dataclass
class SegmentAnnotation:
    segment_index: int
    kind: SegmentKind = "unlabeled"
    t1_sec: float | None = None
    t2_sec: float | None = None
    t1_sample: int | None = None
    t2_sample: int | None = None
    delta_t_sec: float | None = None
    reviewed: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentAnnotation:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RecordingLabels:
    version: int = 1
    source_file: str = ""
    user_name: str = ""
    action_label: str = ""
    sample_rate_hz: float = 100.0
    imu_columns: list[str] = field(default_factory=lambda: list(IMU_COLUMNS))
    label_params: LabelParams = field(default_factory=LabelParams)
    created_at: str = ""
    updated_at: str = ""
    segments: list[SegmentAnnotation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_file": self.source_file,
            "user_name": self.user_name,
            "action_label": self.action_label,
            "sample_rate_hz": self.sample_rate_hz,
            "imu_columns": self.imu_columns,
            "label_params": asdict(self.label_params),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecordingLabels:
        params = LabelParams(**data.get("label_params", {}))
        segments = [SegmentAnnotation.from_dict(s) for s in data.get("segments", [])]
        return cls(
            version=data.get("version", 1),
            source_file=data.get("source_file", ""),
            user_name=data.get("user_name", ""),
            action_label=data.get("action_label", ""),
            sample_rate_hz=float(data.get("sample_rate_hz", 100.0)),
            imu_columns=list(data.get("imu_columns", IMU_COLUMNS)),
            label_params=params,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            segments=segments,
        )


@dataclass
class IMUSegment:
    segment_index: int
    timestamp_ms: np.ndarray
    imu: np.ndarray  # [T, 6]
    user_name: str
    action_label: str
    sample_rate_hz: float

    @property
    def duration_sec(self) -> float:
        return len(self.imu) / self.sample_rate_hz

    @property
    def time_sec(self) -> np.ndarray:
        t0 = self.timestamp_ms[0]
        return (self.timestamp_ms - t0) / 1000.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_sample_rate_hz(timestamp_ms: np.ndarray) -> float:
    if len(timestamp_ms) < 2:
        return 100.0
    diffs = np.diff(timestamp_ms)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 100.0
    median_dt_ms = float(np.median(diffs))
    return 1000.0 / median_dt_ms


def load_imu_csv(csv_path: str | Path) -> tuple[list[IMUSegment], dict[str, Any]]:
    """Load a collected IMU CSV and split into segments."""
    csv_path = Path(csv_path)
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    by_segment: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        seg = int(row["segment_index"])
        by_segment.setdefault(seg, []).append(row)

    meta = {
        "source_file": csv_path.name,
        "user_name": rows[0].get("user_name", ""),
        "action_label": rows[0].get("action_label", ""),
    }

    segments: list[IMUSegment] = []
    for seg_idx in sorted(by_segment):
        chunk = by_segment[seg_idx]
        ts = np.array([float(r["timestamp_ms"]) for r in chunk], dtype=np.float64)
        imu = np.array(
            [[float(r[c]) for c in IMU_COLUMNS] for r in chunk],
            dtype=np.float32,
        )
        fs = estimate_sample_rate_hz(ts)
        segments.append(
            IMUSegment(
                segment_index=seg_idx,
                timestamp_ms=ts,
                imu=imu,
                user_name=chunk[0].get("user_name", ""),
                action_label=chunk[0].get("action_label", ""),
                sample_rate_hz=fs,
            )
        )

    return segments, meta


def labels_path_for_csv(csv_path: str | Path, labels_dir: str | Path = "data/labels") -> Path:
    csv_path = Path(csv_path)
    return Path(labels_dir) / f"{csv_path.stem}.json"


def load_labels(json_path: str | Path) -> RecordingLabels:
    json_path = Path(json_path)
    with json_path.open(encoding="utf-8") as f:
        return RecordingLabels.from_dict(json.load(f))


def save_labels(labels: RecordingLabels, json_path: str | Path) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    labels.updated_at = _now_iso()
    if not labels.created_at:
        labels.created_at = labels.updated_at
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(labels.to_dict(), f, indent=2, ensure_ascii=False)
    return json_path


def init_labels_from_recording(
    csv_path: str | Path,
    segments: list[IMUSegment],
    meta: dict[str, Any],
) -> RecordingLabels:
    csv_path = Path(csv_path)
    return RecordingLabels(
        source_file=meta.get("source_file", csv_path.name),
        user_name=meta.get("user_name", ""),
        action_label=meta.get("action_label", ""),
        sample_rate_hz=float(segments[0].sample_rate_hz if segments else 100.0),
        segments=[
            SegmentAnnotation(segment_index=s.segment_index)
            for s in segments
        ],
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )


def load_or_create_labels(
    csv_path: str | Path,
    labels_dir: str | Path = "data/labels",
) -> tuple[RecordingLabels, list[IMUSegment]]:
    csv_path = Path(csv_path)
    segments, meta = load_imu_csv(csv_path)
    label_path = labels_path_for_csv(csv_path, labels_dir)

    if label_path.exists():
        labels = load_labels(label_path)
        existing = {s.segment_index: s for s in labels.segments}
        merged = []
        for seg in segments:
            if seg.segment_index in existing:
                merged.append(existing[seg.segment_index])
            else:
                merged.append(SegmentAnnotation(segment_index=seg.segment_index))
        labels.segments = merged
        return labels, segments

    return init_labels_from_recording(csv_path, segments, meta), segments


def sample_from_time(time_sec: float, sample_rate_hz: float) -> int:
    return int(round(time_sec * sample_rate_hz))


def time_from_sample(sample_idx: int, sample_rate_hz: float) -> float:
    return sample_idx / sample_rate_hz


def update_annotation_from_taps(
    ann: SegmentAnnotation,
    *,
    kind: SegmentKind,
    t1_sec: float | None,
    t2_sec: float | None,
    sample_rate_hz: float,
    params: LabelParams,
    n_samples: int,
) -> SegmentAnnotation:
    ann.kind = kind
    ann.t1_sec = t1_sec
    ann.t2_sec = t2_sec
    ann.t1_sample = sample_from_time(t1_sec, sample_rate_hz) if t1_sec is not None else None
    ann.t2_sample = sample_from_time(t2_sec, sample_rate_hz) if t2_sec is not None else None

    if t1_sec is not None and t2_sec is not None:
        ann.delta_t_sec = t2_sec - t1_sec
    elif kind == "double_tap" and t2_sec is not None:
        ann.delta_t_sec = None
    else:
        ann.delta_t_sec = None

    if kind == "single_tap":
        ann.t2_sec = None
        ann.t2_sample = None
        if t1_sec is None and t2_sec is not None:
            ann.t1_sec = t2_sec
            ann.t1_sample = sample_from_time(t2_sec, sample_rate_hz)
    elif kind in ("background", "invalid"):
        ann.t1_sec = None
        ann.t2_sec = None
        ann.t1_sample = None
        ann.t2_sample = None

    ann.reviewed = kind != "unlabeled"
    return ann


def soft_labels_for_annotation(
    ann: SegmentAnnotation,
    n_samples: int,
    sample_rate_hz: float,
    params: LabelParams,
) -> np.ndarray:
    if ann.kind == "background" or ann.kind == "invalid":
        return np.zeros(n_samples, dtype=np.float32)
    if ann.kind == "single_tap":
        return np.zeros(n_samples, dtype=np.float32)
    if ann.kind == "double_tap" and ann.t2_sec is not None:
        return gaussian_second_tap_labels(
            n_samples,
            sample_rate_hz,
            ann.t2_sec,
            sigma=params.sigma,
            peak_offset=params.peak_offset,
        )
    return np.zeros(n_samples, dtype=np.float32)


def export_labeled_dataset(
    csv_path: str | Path,
    labels: RecordingLabels,
    segments: list[IMUSegment],
    out_dir: str | Path = "data/labeled",
) -> Path:
    """Export reviewed segments to NPZ + manifest for training."""
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_by_idx = {s.segment_index: s for s in segments}
    manifest: list[dict[str, Any]] = []

    for ann in labels.segments:
        if not ann.reviewed or ann.kind in ("unlabeled", "invalid"):
            continue
        seg = seg_by_idx[ann.segment_index]
        soft = soft_labels_for_annotation(
            ann, len(seg.imu), seg.sample_rate_hz, labels.label_params
        )
        npz_name = f"seg_{ann.segment_index:03d}.npz"
        np.savez(
            out_dir / npz_name,
            imu=seg.imu,
            soft_labels=soft,
            timestamp_ms=seg.timestamp_ms,
            segment_index=ann.segment_index,
            kind=ann.kind,
            t1_sec=-1 if ann.t1_sec is None else ann.t1_sec,
            t2_sec=-1 if ann.t2_sec is None else ann.t2_sec,
        )
        manifest.append(
            {
                "file": npz_name,
                "segment_index": ann.segment_index,
                "kind": ann.kind,
                "t1_sec": ann.t1_sec,
                "t2_sec": ann.t2_sec,
                "delta_t_sec": ann.delta_t_sec,
                "n_samples": len(seg.imu),
            }
        )

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": labels.source_file,
                "sample_rate_hz": labels.sample_rate_hz,
                "label_params": asdict(labels.label_params),
                "segments": manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return out_dir
