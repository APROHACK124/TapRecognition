"""Auditable, deduplicated source exposure for mined-negative FP-rate denominators.

Alarms are still counted on evaluated inputs. These denominators do not turn
crop-only inference into a full-source replay or an unbiased normal-use rate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tap_recognition.recording import estimate_sample_rate_hz


DURATION_BASIS = 'unique_hard_negative_sources_v2'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _original_duration(path: Path) -> tuple[int, float, float]:
    stamps = pd.read_csv(path, usecols=['timestamp_ms']).timestamp_ms.to_numpy(dtype=np.float64)
    if len(stamps) < 2 or not np.isfinite(stamps).all():
        raise ValueError(f'{path}: source needs at least two finite timestamps')
    fs = estimate_sample_rate_hz(stamps)
    if not np.all(np.diff(stamps) > 0) or not np.allclose(np.diff(stamps), 1000 / fs, atol=.01, rtol=0):
        raise ValueError(f'{path}: source timestamps must be increasing and uniformly sampled')
    return len(stamps), fs, len(stamps) / fs / 60


def recording_durations(root: Path, inputs: list[dict], source_dir: Path | None = None,
                        *, cache=None) -> pd.DataFrame:
    """One row per evaluated file; sum `minutes` for the deduplicated denominator.

    Inputs require path, evaluated_minutes, and n_true_events. Ordinary files
    retain included duration. Hard-negative and normal_usage crops resolve .mining.json and a
    hash-verified source CSV, counting that CSV once across all supplied inputs.
    Durations are divided equally among members of each source to keep ordinary
    group-by/sum reports additive. Invoke separately for each split/model, not
    for duplicated threshold/policy rows. A full mining source plus its crops is
    also counted only once. No original recording is inferred from crop bounds.
    """
    root = Path(root).resolve()
    source_dir = Path(source_dir) if source_dir is not None else root / 'hard_negative_mining/source_recordings'
    if not source_dir.is_absolute():
        source_dir = root / source_dir
    rows, duration_cache, names = [], {}, set()
    digest_of = _sha256 if cache is None else cache.file_sha256
    original_duration = _original_duration if cache is None else cache.original_duration
    for item in inputs:
        path = Path(item['path'])
        if not path.is_absolute():
            path = root / path
        name = path.name
        if name in names:
            raise ValueError(f'Duplicate evaluated file in duration inputs: {name}')
        names.add(name)
        evaluated = float(item['evaluated_minutes'])
        if not np.isfinite(evaluated) or evaluated <= 0:
            raise ValueError(f'{name}: evaluated_minutes must be finite and positive')
        lower = name.lower()
        mining = ('negative_mining' in lower or 'hard_negative' in lower or 'hard-negatives' in lower
                  or ('normal_usage' in lower and '__crop_' in lower))
        original_path, original_name, digest = path, name, ''
        source_frames, source_fs, source_minutes = np.nan, np.nan, evaluated
        if mining:
            if int(item['n_true_events']) != 0:
                raise ValueError(f'{name}: negative_mining exposure requires no annotated double taps')
            if '__crop_' in name:
                meta_path = path.with_suffix('.mining.json')
                if not meta_path.is_file():
                    raise FileNotFoundError(f'{meta_path}: required to recover the original recording duration')
                meta = json.loads(meta_path.read_text()) if cache is None else cache.read_json(meta_path)
                selection = meta.get('selection', {})
                original_name, digest = selection.get('file'), meta.get('source_sha256')
                if not original_name or Path(original_name).name != original_name or not digest:
                    raise ValueError(f'{meta_path}: missing/invalid source filename or SHA-256')
                if selection.get('source_sha256', digest) != digest or meta.get('label_class') != 0:
                    raise ValueError(f'{meta_path}: inconsistent negative-source provenance')
                recorded_path = Path(meta.get('source_path', original_name))
                # Prefer a portable local source directory over old absolute pod paths.
                candidates = [source_dir / original_name, path.parent / original_name,
                              recorded_path if recorded_path.is_absolute() else root / recorded_path]
                original_path = next((p for p in candidates if p.is_file() and digest_of(p) == digest), None)
                if original_path is None:
                    raise FileNotFoundError(f'{original_name}: no hash-matching original CSV found; '
                                            f'copy it into {source_dir} (expected SHA-256 {digest})')
            else:
                if not path.is_file():
                    raise FileNotFoundError(path)
                digest = digest_of(path)
            if digest not in duration_cache:
                duration_cache[digest] = original_duration(original_path)
            source_frames, source_fs, source_minutes = duration_cache[digest]
            if '__crop_' in name:
                start, end = selection.get('crop_start_frame'), selection.get('crop_end_frame')
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= source_frames:
                    raise ValueError(f'{meta_path}: crop bounds exceed the verified original recording')
        rows.append(dict(file=name, evaluated_minutes=evaluated,
                         n_true_events=int(item['n_true_events']), is_mining=mining,
                         source_file=original_name, source_sha256=digest,
                         source_path=str(original_path.resolve()), source_frames=source_frames,
                         source_sample_rate_hz=source_fs, source_minutes=source_minutes,
                         exposure_key=f'source:{digest}' if mining else f'file:{name}'))
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError('No recordings supplied for duration accounting')
    table['source_members'] = table.groupby('exposure_key').file.transform('size')
    table['minutes'] = table.source_minutes / table.source_members
    table['duration_basis'] = DURATION_BASIS
    return table


def durations_for_recordings(root: Path, recordings, source_dir: Path | None = None,
                             *, cache=None) -> pd.DataFrame:
    """Adapter for both continuous-evaluation and comparison replay recordings."""
    inputs = []
    for rec in recordings:
        frames = sum(end - start for _, start, end in rec.segments) if hasattr(rec, 'segments') else len(rec.imu)
        inputs.append(dict(path=rec.path, evaluated_minutes=frames / rec.fs / 60,
                           n_true_events=len(rec.events)))
    return recording_durations(root, inputs, source_dir, cache=cache)


def duration_fingerprint(table: pd.DataFrame) -> str:
    """Portable fingerprint of actual duration inputs, independent of host paths."""
    fields = ['file', 'evaluated_minutes', 'n_true_events', 'source_file', 'source_sha256',
              'source_minutes', 'source_members', 'minutes', 'duration_basis']
    # CSV round trips can perturb the final float bit; do not fingerprint that noise.
    payload = table.sort_values('file')[fields].round(10).to_dict('records')
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def durations_for_export(root: Path, by_file: pd.DataFrame, data: dict | None,
                         source_dir: Path | None = None, *, cache=None) -> pd.DataFrame:
    """Resolve current source durations for exported counts without rerunning inference."""
    directories = []
    for value in (data or {}).get('val_dir', '').split(','):
        if value.strip():
            path = Path(value.strip())
            directories.extend([path if path.is_absolute() else root / path,
                                root / path.parent.name / path.name])
    inputs = []
    for row in by_file.itertuples(index=False):
        path = next((d / row.file for d in directories if (d / row.file).is_file()), root / row.file)
        inputs.append(dict(path=path, evaluated_minutes=getattr(row, 'evaluated_minutes', row.minutes),
                           n_true_events=row.n_true_events))
    return recording_durations(root, inputs, source_dir, cache=cache)


def apply_export_durations(summary: pd.DataFrame, by_type: pd.DataFrame, by_file: pd.DataFrame,
                            durations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute only duration-dependent display rates; counts/thresholds stay frozen."""
    summary, by_type, by_file = summary.copy(), by_type.copy(), by_file.copy()
    indexed = durations.set_index('file')
    for field in ('minutes', 'evaluated_minutes', 'source_file', 'source_sha256', 'source_minutes', 'source_members'):
        by_file[field] = by_file.file.map(indexed[field])
    by_file['event_fp_per_min'] = by_file.event_fp / by_file.minutes
    for idx, row in by_type.iterrows():
        files = by_file[by_file.recording_type == row.recording_type]
        by_type.loc[idx, 'minutes'] = files.minutes.sum()
        by_type.loc[idx, 'evaluated_minutes'] = files.evaluated_minutes.sum()
        by_type.loc[idx, 'event_fp_per_min'] = row.event_fp / files.minutes.sum()
    total_minutes = float(durations.minutes.sum())
    negative_minutes = float(durations.loc[durations.n_true_events == 0, 'minutes'].sum())
    negative = by_file[by_file.n_true_events == 0]
    for idx, row in summary[summary.level == 'event'].iterrows():
        fp_column = 'event_fp' if row.side == 'all' else f'event_{row.side}_fp'
        summary.loc[idx, 'minutes'] = total_minutes
        summary.loc[idx, 'evaluated_minutes'] = durations.evaluated_minutes.sum()
        summary.loc[idx, 'negative_minutes'] = negative_minutes
        summary.loc[idx, 'fp_per_min'] = row.fp / total_minutes
        summary.loc[idx, 'negative_fp_per_min'] = negative[fp_column].sum() / negative_minutes if negative_minutes else np.nan
    for table in (summary, by_type, by_file):
        table['duration_basis'] = DURATION_BASIS
        table['duration_sha256'] = duration_fingerprint(durations)
    return summary, by_type, by_file
