"""Measure alarms on complete normal-use originals, independently of mined crops."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks_analysis.hard_negative_mining import (
    MiningRecording, file_sha256, load_recording, load_mining_model, predict_recording,
)
from notebooks_analysis.model_comparison_examples import emit_alarms_pair
from notebooks_analysis.recording_types import recording_type_from_filename


def load_normal_usage(source_dir: Path, filenames: list[str] | None = None):
    """Load full originals only, deduplicating identical source content.

    None discovers all normal_usage originals; [] disables this population.
    The source folder is an explicit evaluation population, NOT implicitly a
    held-out split. Missing labels are allowed for the explicitly negative
    normal_usage action, but any annotated positive event is an error.
    """
    directory = Path(source_dir).resolve()
    if filenames is not None:
        if len(filenames) != len(set(filenames)) or any(Path(n).name != n for n in filenames):
            raise ValueError('NORMAL_USAGE_FILES must contain unique basenames')
        paths = [directory / name for name in filenames]
    else:
        paths = sorted(p for p in directory.glob('*.csv') if 'normal_usage' in p.name.lower())
    recordings, rows, seen = [], [], set()
    for path in paths:
        if '__crop_' in path.name.lower() or path.name.lower().endswith('.labels.csv'):
            if filenames is not None:
                raise ValueError(f'{path.name}: select original normal_usage CSVs, not crops')
            continue
        rec = load_recording(path)
        kind = recording_type_from_filename(path.name, rec.data.action_label.iloc[0])
        if kind != 'normal_usage' or rec.events:
            raise ValueError(f'{path.name}: full normal_usage evaluation requires negative-only normal_usage originals')
        if rec.sha256 in seen:
            continue
        seen.add(rec.sha256)
        recordings.append(rec)
        rows.append(dict(file=path.name, source_sha256=rec.sha256, frames=len(rec.data),
                         sample_rate_hz=rec.fs, minutes=len(rec.data) / rec.fs / 60))
    manifest = pd.DataFrame(rows, columns=['file', 'source_sha256', 'frames', 'sample_rate_hz', 'minutes'])
    payload = manifest.sort_values('file').round(10).to_dict('records')
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return recordings, manifest, fingerprint


def evaluate_normal_usage(recordings: list[MiningRecording], traces: dict[str, np.ndarray],
                           left_threshold: float, right_threshold: float,
                           lookahead_frames: int = 12, refractory_sec: float = .5, *, emitter=None):
    """Actual emitted alarms / ALL full-source minutes, including zero-alarm files."""
    rows, alarms = [], []
    for rec in recordings:
        if rec.events:
            raise ValueError(f'{rec.path.name}: normal_usage originals must contain no annotated double taps')
        p = traces[rec.path.name]
        if p.shape != (len(rec.data), 3) or not np.isfinite(p).all():
            raise ValueError(f'{rec.path.name}: expected full-source [frames, 3] probabilities')
        detector = emit_alarms_pair if emitter is None else emitter
        detected = detector(p, left_threshold, right_threshold, lookahead_frames,
                            round(refractory_sec * rec.fs), [(0, len(p))])
        minutes = len(p) / rec.fs / 60
        left = sum(a['class_id'] == 1 for a in detected)
        right = len(detected) - left
        rows.append(dict(file=rec.path.name, recording_type='normal_usage', source_sha256=rec.sha256,
                         minutes=minutes, frames=len(p), event_fp=len(detected), event_left_fp=left,
                         event_right_fp=right, event_fp_per_min=len(detected) / minutes))
        alarms.extend(dict(file=rec.path.name, **a) for a in detected)
    files = pd.DataFrame(rows, columns=['file', 'recording_type', 'source_sha256', 'minutes', 'frames',
        'event_fp', 'event_left_fp', 'event_right_fp', 'event_fp_per_min'])
    minutes = float(files.minutes.sum())
    fp = int(files.event_fp.sum())
    overall = dict(normal_usage_recordings=len(files), normal_usage_minutes=minutes,
                   normal_usage_fp=fp, normal_usage_fp_per_min=fp / minutes if minutes else np.nan)
    sides = {cls: dict(normal_usage_minutes=minutes, normal_usage_fp=int(files[column].sum()),
                       normal_usage_fp_per_min=float(files[column].sum()) / minutes if minutes else np.nan)
             for cls, column in ((1, 'event_left_fp'), (2, 'event_right_fp'))}
    alarm_table = pd.DataFrame(alarms, columns=['file', 'alarm_frame', 'peak_frame', 'class_id', 'probability'])
    return overall, sides, files, alarm_table


def replay_normal_usage(root: Path, operating_points: pd.DataFrame, source_dir: Path,
                         filenames: list[str] | None = None, chunk_frames: int = 4096, *, runtime=None):
    """Replay frozen thresholds for every model on the same full-source population."""
    if runtime is None:
        recordings, manifest, fingerprint = load_normal_usage(source_dir, filenames)
    else:
        recordings, manifest, fingerprint = runtime.load_normal_usage(source_dir, filenames)
    summaries, files, alarms = [], [], []
    looks = {'instant': 0, 'lookahead_12f': 12}
    if recordings:
        for row in operating_points.to_dict('records'):
            checkpoint = row.get('checkpoint_path')
            if not isinstance(checkpoint, str):
                raise ValueError(f"{row['model']}: full-source replay needs evaluation_context.json with checkpoint_path")
            path = Path(root) / checkpoint
            if runtime is None:
                if file_sha256(path) != row['checkpoint_sha256']:
                    raise ValueError(f"{row['model']}: checkpoint differs from its selected-threshold export")
                model = load_mining_model(path)
                # An explicit evaluation-data preprocessing override is authoritative.
                data = row.get('evaluation_data')
                if isinstance(data, dict):
                    model.highpass = bool(data['highpass'])
                traces = {r.path.name: predict_recording(r, model, False, chunk_frames) for r in recordings}
            else:
                checkpoint_data = runtime.checkpoint(path, row['checkpoint_sha256'])
                traces = {r.path.name: runtime.normal_prediction(
                    path, checkpoint_data, row['checkpoint_sha256'], row.get('evaluation_data'), r, chunk_frames)
                    for r in recordings}
            overall, _, by_file, by_alarm = evaluate_normal_usage(
                recordings, traces, row['left_threshold'], row['right_threshold'],
                looks[row['policy']], row.get('refractory_sec', .5),
                emitter=None if runtime is None else runtime.emit_alarms_pair)
            identifiers = dict(model=row['model'], selection_mode=row['selection_mode'], policy=row['policy'],
                               left_threshold=row['left_threshold'], right_threshold=row['right_threshold'],
                               normal_usage_sha256=fingerprint)
            summaries.append(identifiers | overall)
            files.append(by_file.assign(**identifiers))
            alarms.append(by_alarm.assign(**identifiers))
    return (pd.DataFrame(summaries), pd.concat(files, ignore_index=True) if files else pd.DataFrame(),
            pd.concat(alarms, ignore_index=True) if alarms else pd.DataFrame(), manifest, fingerprint)
