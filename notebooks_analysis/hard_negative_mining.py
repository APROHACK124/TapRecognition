"""Inspect causal model traces and export reviewed, single-session negatives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from notebooks_analysis.model_comparison_examples import emit_alarms_pair
from tap_recognition.dataset import load_trigger_events
from tap_recognition.model_factory import build_model
from tap_recognition.physics import IMUSimulator
from tap_recognition.recording import IMU_COLUMNS, estimate_sample_rate_hz


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_recordings(source_dir: Path, input_dir: Path) -> list[Path]:
    """Copy raw recordings and sparse labels once; refuse conflicting copies."""
    source_dir, input_dir = Path(source_dir).resolve(), Path(input_dir).resolve()
    paths = sorted(p for p in source_dir.glob('*.csv') if not p.name.endswith('.labels.csv'))
    if not paths:
        raise ValueError(f'No recordings in {source_dir}')
    input_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in paths:
        for source in (csv_path, csv_path.with_suffix('.txt')):
            if not source.is_file():
                continue
            dest = input_dir / source.name
            if dest.exists():
                if file_sha256(dest) != file_sha256(source):
                    raise FileExistsError(f'Different recording already staged: {dest}')
            else:
                shutil.copy2(source, dest)
    return [input_dir / p.name for p in paths]


@dataclass
class MiningRecording:
    path: Path
    data: pd.DataFrame
    fs: float
    segments: list[tuple[int, int, int]]
    events: list[tuple[int, int]]
    sha256: str

    @property
    def raw(self) -> np.ndarray:
        return self.data[IMU_COLUMNS].to_numpy(dtype=np.float32)

    @property
    def time(self) -> np.ndarray:
        timestamps = self.data.timestamp_ms.to_numpy(dtype=np.float64)
        return (timestamps - timestamps[0]) / 1000


def load_recording(path: Path) -> MiningRecording:
    path = Path(path).resolve()
    data = pd.read_csv(path)
    required = {'timestamp_ms', 'user_name', 'action_label', *IMU_COLUMNS}
    if not required <= set(data.columns) or len(data) < 2:
        raise ValueError(f'{path}: requires at least two rows and columns {sorted(required)}')
    timestamps = data.timestamp_ms.to_numpy(dtype=np.float64)
    fs = estimate_sample_rate_hz(timestamps)
    if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
        raise ValueError(f'{path}: timestamps must be finite and increasing')
    if not np.allclose(np.diff(timestamps), 1000 / fs, atol=.01, rtol=0):
        raise ValueError(f'{path}: expected uniformly sampled frames')
    if not np.isfinite(data[IMU_COLUMNS].to_numpy(dtype=np.float32)).all():
        raise ValueError(f'{path}: non-finite IMU data')
    if 'segment_index' not in data:
        data['segment_index'] = 1
    ids = data.segment_index.to_numpy(dtype=int)
    if not np.array_equal(ids, data.segment_index.to_numpy()):
        raise ValueError(f'{path}: non-integer segment_index')
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(data)]
    if len(np.unique(ids[starts])) != len(starts):
        raise ValueError(f'{path}: segment IDs must form contiguous runs')
    segments = [(int(ids[a]), int(a), int(b)) for a, b in zip(starts, ends)]
    events = load_trigger_events(path.with_suffix('.txt'))
    if any(not 0 <= frame < len(data) or cls not in (1, 2) for frame, cls in events):
        raise ValueError(f'{path}: invalid annotated event')
    return MiningRecording(path, data, fs, segments, events, file_sha256(path))


def recording_inventory(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(input_dir).glob('*.csv')):
        if path.name.endswith('.labels.csv'):
            continue
        rec = load_recording(path)
        rows.append(dict(file=path.name, user=rec.data.user_name.iloc[0],
                         action=rec.data.action_label.iloc[0], frames=len(rec.data),
                         seconds=len(rec.data) / rec.fs, sample_rate_hz=rec.fs,
                         sessions=len(rec.segments), annotated_events=len(rec.events)))
    return pd.DataFrame(rows)


@dataclass
class MiningModel:
    path: Path
    model: torch.nn.Module
    highpass: bool
    window_samples: int
    sha256: str


def load_mining_model(checkpoint_path: Path) -> MiningModel:
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['model_config'], checkpoint.get('model_type')).cpu().eval()
    model.load_state_dict(checkpoint['model_state'])
    data = checkpoint['train_config']['data']
    return MiningModel(path, model, bool(data['highpass']), int(data['window_samples']), file_sha256(path))


def predict_recording(rec: MiningRecording, model: MiningModel,
                      reset_at_segments: bool = False) -> np.ndarray:
    """Causal full-length inference; preprocessing follows each checkpoint."""
    imu = rec.raw
    if model.highpass:
        imu = IMUSimulator(sample_rate=rec.fs).highpass(imu)
    bounds = [(a, b) for _, a, b in rec.segments] if reset_at_segments else [(0, len(imu))]
    probs = np.empty((len(imu), 3), dtype=np.float32)
    with torch.inference_mode():
        for start, end in bounds:
            logits, _ = model.model(torch.from_numpy(imu[start:end]).unsqueeze(0))
            probs[start:end] = logits[0].softmax(-1).numpy()
    return probs


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return int(value)


def crop_bounds(rec: MiningRecording, anchor_frame: int, before_frames: int,
                after_frames: int, context_before_frames: int | None = None,
                context_after_frames: int | None = None) -> dict:
    """Inclusive marked interval, half-open CSV bounds; never join sessions.

    Optional context enlarges the export. Near a boundary the context window
    slides inside that session when possible, keeping the marked interval.
    """
    anchor = _nonnegative_int(anchor_frame, 'anchor_frame')
    before = _nonnegative_int(before_frames, 'before_frames')
    after = _nonnegative_int(after_frames, 'after_frames')
    left = max(before, before if context_before_frames is None else
               _nonnegative_int(context_before_frames, 'context_before_frames'))
    right = max(after, after if context_after_frames is None else
                _nonnegative_int(context_after_frames, 'context_after_frames'))
    containing = [(sid, a, b) for sid, a, b in rec.segments if a <= anchor < b]
    if not containing:
        raise ValueError(f'Anchor {anchor} is outside {rec.path.name}')
    sid, seg_start, seg_end = containing[0]
    size = min(left + right + 1, seg_end - seg_start)
    start = max(seg_start, min(anchor - left, seg_end - size))
    end = start + size
    return dict(segment_index=sid, anchor_frame=anchor, crop_start_frame=start,
                crop_end_frame=end, crop_frames=size, crop_seconds=size / rec.fs,
                anchor_in_crop=anchor - start,
                core_start_frame=max(seg_start, anchor - before),
                core_end_frame=min(seg_end, anchor + after + 1),
                context_clipped=size < left + right + 1)


def find_candidates(rec: MiningRecording, probs: np.ndarray, model_name: str,
                    left_threshold: float, right_threshold: float,
                    lookahead_frames: int = 12, refractory_frames: int = 50,
                    anchor: str = 'peak', before_frames: int = 40, after_frames: int = 10,
                    context_before_frames: int | None = None,
                    context_after_frames: int | None = None,
                    training_window_frames: int = 300,
                    reset_at_segments: bool = False) -> pd.DataFrame:
    """Find review candidates using one model's causal, shared-cooldown alarms."""
    if anchor not in ('crossing', 'peak', 'alarm'):
        raise ValueError('anchor must be crossing, peak, or alarm')
    if not all(np.isfinite(t) and 0 < t <= 1 for t in (left_threshold, right_threshold)):
        raise ValueError('Thresholds must be in (0, 1]')
    lookahead_frames = _nonnegative_int(lookahead_frames, 'lookahead_frames')
    refractory_frames = _nonnegative_int(refractory_frames, 'refractory_frames')
    training_window_frames = _nonnegative_int(training_window_frames, 'training_window_frames')
    if training_window_frames == 0:
        raise ValueError('training_window_frames must be positive')
    probs = np.asarray(probs)
    if probs.shape != (len(rec.data), 3) or not np.isfinite(probs).all():
        raise ValueError('Expected finite [recording frames, 3] probabilities')
    bounds = [(a, b) for _, a, b in rec.segments] if reset_at_segments else [(0, len(rec.data))]
    alarms = emit_alarms_pair(probs, left_threshold, right_threshold, lookahead_frames,
                             refractory_frames, bounds)
    columns = ['candidate_id', 'file', 'source_sha256', 'model', 'class_id', 'side',
               'probability', 'left_threshold', 'right_threshold', 'lookahead_frames',
               'refractory_frames', 'reset_at_segments', 'anchor', 'crossing_frame',
               'alarm_frame', 'peak_frame', 'anchor_frame', 'anchor_seconds',
               'segment_index', 'crop_start_frame', 'crop_end_frame', 'crop_frames',
               'crop_seconds', 'anchor_in_crop', 'core_start_frame', 'core_end_frame',
               'context_clipped', 'training_retains_core', 'annotated_events_in_crop']
    rows = []
    for index, alarm in enumerate(alarms):
        crossing = alarm['alarm_frame'] - lookahead_frames
        frame = crossing if anchor == 'crossing' else alarm[f'{anchor}_frame']
        crop = crop_bounds(rec, frame, before_frames, after_frames,
                           context_before_frames, context_after_frames)
        start, end = crop['crop_start_frame'], crop['crop_end_frame']
        visible_end = min(end, start + training_window_frames)
        rows.append(dict(candidate_id=index, file=rec.path.name, source_sha256=rec.sha256,
                         model=model_name, **alarm, side='left' if alarm['class_id'] == 1 else 'right',
                         left_threshold=left_threshold, right_threshold=right_threshold,
                         lookahead_frames=lookahead_frames, refractory_frames=refractory_frames,
                         reset_at_segments=reset_at_segments, anchor=anchor, crossing_frame=crossing,
                         anchor_seconds=rec.time[frame], **crop,
                         training_retains_core=crop['core_end_frame'] <= visible_end,
                         annotated_events_in_crop=sum(start <= f < end for f, _ in rec.events)))
    return pd.DataFrame(rows, columns=columns)


def plot_recording(rec: MiningRecording, traces: dict[str, np.ndarray],
                   thresholds: dict[str, tuple[float, float]], candidates: pd.DataFrame,
                   start_frame: int = 0, end_frame: int | None = None):
    """Raw IMU axes, all model traces, candidate anchors and export/core spans."""
    end = len(rec.data) if end_frame is None else min(len(rec.data), int(end_frame))
    start = max(0, int(start_frame))
    if not start < end:
        raise ValueError('Empty plot interval')
    t = rec.time[start:end]
    fig, axes = plt.subplots(2 + len(traces), 1, sharex=True,
                             figsize=(16, 2.5 * (2 + len(traces))), layout='constrained')
    for axis, columns, unit in ((axes[0], IMU_COLUMNS[:3], 'Acceleration'),
                                (axes[1], IMU_COLUMNS[3:], 'Gyroscope')):
        for col in columns:
            axis.plot(t, rec.data[col].iloc[start:end], lw=.8, label=col)
        axis.set_ylabel(unit)
        axis.legend(loc='upper right', ncol=3, fontsize=8)
    for axis, (name, probs) in zip(axes[2:], traces.items()):
        left, right = thresholds[name]
        for cls, side, color, threshold in ((1, 'left', 'tab:blue', left),
                                             (2, 'right', 'tab:orange', right)):
            axis.plot(t, probs[start:end, cls], color=color, lw=1, label=f'P({side})')
            axis.axhline(threshold, color=color, ls=':', label=f'{side} threshold {threshold:g}')
        own = candidates[candidates.model == name]
        visible = own[own.anchor_frame.between(start, end - 1)]
        for _, row in visible.iterrows():
            color = 'tab:blue' if row.class_id == 1 else 'tab:orange'
            axis.scatter(rec.time[int(row.anchor_frame)], row.probability, color=color, marker='v')
            axis.annotate(str(row.candidate_id), (rec.time[int(row.anchor_frame)], row.probability),
                          xytext=(3, 6), textcoords='offset points', fontsize=8)
        axis.set(ylabel=name, ylim=(-.03, 1.1))
        axis.legend(loc='upper right', ncol=4, fontsize=8)
    visible_crops = candidates[(candidates.crop_end_frame > start) & (candidates.crop_start_frame < end)]
    for axis in axes:
        for _, row in visible_crops.iterrows():
            axis.axvspan(rec.time[int(row.crop_start_frame)],
                         rec.time[int(row.crop_end_frame) - 1] + 1 / rec.fs, color='gray', alpha=.08)
            axis.axvspan(rec.time[int(row.core_start_frame)],
                         rec.time[int(row.core_end_frame) - 1] + 1 / rec.fs, color='red', alpha=.1)
            axis.axvline(rec.time[int(row.anchor_frame)], color='red', ls='--', lw=.7)
        for _, a, _ in rec.segments[1:]:
            if start <= a < end:
                axis.axvline(rec.time[a], color='gray', ls=':', lw=.8)
        for frame, _ in rec.events:
            if start <= frame < end:
                axis.axvline(rec.time[frame], color='green', lw=.8)
        axis.grid(alpha=.2)
    axes[0].set_title(rec.path.name + ' | gray: export context; red: hard interval; numbers: candidate IDs')
    axes[-1].set_xlabel('Seconds from recording start')
    return fig, axes


def export_candidates(rec: MiningRecording, selected: pd.DataFrame,
                      model: MiningModel, output_dir: Path,
                      training_window_frames: int = 300) -> pd.DataFrame:
    """Write raw CSV + explicit empty labels + provenance; identical reruns skip."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_window_frames = _nonnegative_int(training_window_frames, 'training_window_frames')
    if training_window_frames == 0:
        raise ValueError('training_window_frames must be positive')
    rows, plans, planned_paths = [], [], set()
    for _, row in selected.iterrows():
        if row.file != rec.path.name or row.source_sha256 != rec.sha256:
            raise ValueError('Candidates belong to a different source recording; regenerate them')
        start, end = int(row.crop_start_frame), int(row.crop_end_frame)
        if not 0 <= start < end <= len(rec.data) or end - start < 2:
            raise ValueError('A crop needs at least two valid frames')
        if not any(a <= start < end <= b for _, a, b in rec.segments):
            raise ValueError('Crop crosses source sessions')
        if any(start <= frame < end for frame, _ in rec.events):
            raise ValueError('A negative crop cannot contain an annotated positive tap')
        if not start <= int(row.core_start_frame) <= int(row.anchor_frame) < int(row.core_end_frame) <= end:
            raise ValueError('Marked hard interval must lie inside the crop')
        if int(row.core_end_frame) > start + training_window_frames:
            raise ValueError('Training truncates negative sessions to their first window_samples frames; '
                             'reduce context_before_frames so the hard interval is retained')
        # Keep original raw values, not high-passed inference inputs.
        crop = rec.data.iloc[start:end].copy()
        crop['timestamp_ms'] = crop.timestamp_ms - crop.timestamp_ms.iloc[0]
        crop['segment_index'] = 1
        crop['phase'] = 'recording'
        crop['action_label'] = 'negative_mining'
        stem = f'{rec.path.stem}__crop_f{start:06d}_f{end:06d}'
        csv_path = output_dir / f'{stem}.csv'
        label_path = csv_path.with_suffix('.txt')
        metadata_path = output_dir / f'{stem}.mining.json'
        if csv_path in planned_paths:
            raise ValueError('Selected candidates produce the same crop; select only one of them')
        planned_paths.add(csv_path)
        metadata = dict(source_path=str(rec.path), source_sha256=rec.sha256,
                        checkpoint_path=str(model.path), checkpoint_sha256=model.sha256,
                        highpass=model.highpass, label_class=0, training_window_frames=training_window_frames,
                        crop_bounds='[start, end), original CSV row indices; anchor/core interval is inclusive',
                        selection=row.to_dict())
        contents = (crop.to_csv(index=False, float_format='%.9f').encode(), b'',
                    (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
        paths = (csv_path, label_path, metadata_path)
        exists = any(p.exists() for p in paths)
        if exists and not all(p.is_file() and p.read_bytes() == content for p, content in zip(paths, contents)):
            raise FileExistsError(f'Different or incomplete crop already exists: {csv_path}')
        plans.append((paths, contents, exists))
        rows.append(dict(candidate_id=int(row.candidate_id), csv=str(csv_path), label=str(label_path),
                         provenance=str(metadata_path), frames=end - start,
                         seconds=(end - start) / rec.fs, anchor_in_crop=int(row.anchor_frame) - start,
                         status='already exported' if exists else 'exported'))
    for paths, contents, exists in plans:
        if not exists:
            for path, content in zip(paths, contents):
                path.write_bytes(content)
    return pd.DataFrame(rows)
