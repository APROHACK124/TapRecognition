"""Mine low-confidence labeled positive segments, not negative alarm crops.

The correct-side peak must reach its model's threshold in the inclusive scoring
window around the label. One failing model is enough to keep the whole segment.
Unlabeled/ambiguous segments are audited but are never hard-positive candidates.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks_analysis.hard_negative_mining import (
    MiningModel, MiningRecording, file_sha256, load_recording, predict_recording,
)
from tap_recognition.labels import class_id_from_filename
from tap_recognition.recording import IMU_COLUMNS


SEGMENT_COLUMNS = [
    'candidate_id', 'file', 'source_sha256', 'source_labels_sha256', 'segment_index',
    'crop_start_frame', 'crop_end_frame', 'crop_frames', 'crop_seconds',
    'class_id', 'side', 'true_frame', 'label_in_crop', 'score_start_frame',
    'score_end_frame', 'score_before_ms', 'score_after_ms', 'window_clipped',
    'status', 'is_hard', 'model', 'trigger_models', 'score_details',
    'reset_at_segments', 'training_window_frames', 'training_retains_event',
]
SCORE_COLUMNS = ['candidate_id', 'file', 'segment_index', 'model', 'class_id',
                 'threshold', 'peak_probability', 'peak_frame', 'is_hard']
EXPORT_COLUMNS = ['candidate_id', 'file', 'csv', 'label', 'provenance',
                  'frames', 'seconds', 'label_in_crop', 'class_id', 'status']


def _positive_class(rec: MiningRecording) -> int:
    cls = class_id_from_filename(rec.path.name)
    if cls not in (1, 2) or '__crop_' in rec.path.name:
        raise ValueError(f'{rec.path.name}: choose an original left/right double-tap recording')
    if not rec.path.with_suffix('.txt').is_file():
        raise FileNotFoundError(f'{rec.path.name}: auto-label the source before mining')
    if any(label_class != cls for _, label_class in rec.events):
        raise ValueError(f'{rec.path.name}: event side differs from source filename')
    return cls


def predict_positive_recording(rec: MiningRecording, model: MiningModel,
                               reset_at_segments: bool = True,
                               chunk_frames: int = 4096) -> np.ndarray:
    """By default reset preprocessing AND model state for each training segment.

    Continuous inference is an explicit alternative. Scoring never crosses a
    segment boundary in either mode. Inputs remain raw for subsequent exports.
    """
    _positive_class(rec)
    if not reset_at_segments:
        return predict_recording(rec, model, False, chunk_frames)
    traces = np.empty((len(rec.data), 3), dtype=np.float32)
    for sid, start, end in rec.segments:
        segment = replace(rec, data=rec.data.iloc[start:end].copy(),
                          segments=[(sid, 0, end - start)],
                          events=[(frame - start, cls) for frame, cls in rec.events
                                  if start <= frame < end])
        traces[start:end] = predict_recording(segment, model, False, chunk_frames)
    return traces


def score_segments(rec: MiningRecording, traces: dict[str, np.ndarray],
                   thresholds: dict[str, tuple[float, float]],
                   score_before_ms: float = 500., score_after_ms: float = 500.,
                   reset_at_segments: bool = True,
                   training_window_frames: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every segment plus one per-model score for each unambiguous label.

    End bounds are exclusive; the requested +/-500 ms limits are inclusive.
    A peak exactly equal to its threshold is easy for that model. A high
    wrong-side probability does not rescue a low correct-side probability.
    """
    cls = _positive_class(rec)
    if not traces or set(traces) != set(thresholds):
        raise ValueError('Provide traces and thresholds for the same nonempty model set')
    if (not np.isfinite([score_before_ms, score_after_ms]).all()
            or min(score_before_ms, score_after_ms) < 0):
        raise ValueError('Scoring window milliseconds must be finite and non-negative')
    if (isinstance(training_window_frames, bool) or training_window_frames <= 0
            or int(training_window_frames) != training_window_frames):
        raise ValueError('training_window_frames must be a positive integer')
    for name, trace in traces.items():
        if trace.shape != (len(rec.data), 3) or not np.isfinite(trace).all():
            raise ValueError(f'{name}: expected finite [recording frames, 3] probabilities')
        if not np.isfinite(thresholds[name]).all() or len(thresholds[name]) != 2 or not all(
                0 < threshold <= 1 for threshold in thresholds[name]):
            raise ValueError(f'{name}: left/right thresholds must be in (0, 1]')
    before = int(np.floor(score_before_ms * rec.fs / 1000 + 1e-9))
    after = int(np.floor(score_after_ms * rec.fs / 1000 + 1e-9))
    label_hash = file_sha256(rec.path.with_suffix('.txt'))
    rows, scores = [], []
    for index, (sid, start, end) in enumerate(rec.segments):
        events = [(frame, side) for frame, side in rec.events if start <= frame < end]
        row = dict(candidate_id=index, file=rec.path.name, source_sha256=rec.sha256,
                   source_labels_sha256=label_hash, segment_index=sid,
                   crop_start_frame=start, crop_end_frame=end, crop_frames=end - start,
                   crop_seconds=(end - start) / rec.fs, class_id=cls,
                   side='left' if cls == 1 else 'right', true_frame=None, label_in_crop=None,
                   score_start_frame=None, score_end_frame=None,
                   score_before_ms=float(score_before_ms), score_after_ms=float(score_after_ms),
                   window_clipped=False, status='unlabeled' if not events else 'ambiguous',
                   is_hard=False, model=None, trigger_models=[], score_details=[],
                   reset_at_segments=bool(reset_at_segments),
                   training_window_frames=int(training_window_frames), training_retains_event=False)
        if len(events) == 1:
            frame, _ = events[0]
            lo, hi = max(start, frame - before), min(end, frame + after + 1)
            details = []
            for name in traces:
                trace = traces[name]
                peak = lo + int(np.argmax(trace[lo:hi, cls]))
                probability = float(trace[peak, cls])
                threshold = float(thresholds[name][cls - 1])
                details.append(dict(model=name, threshold=threshold, peak_probability=probability,
                                    peak_frame=peak, is_hard=probability < threshold))
            hard_models = sorted(d['model'] for d in details if d['is_hard'])
            representative = min(details, key=lambda d: (d['peak_probability'] - d['threshold'], d['model']))
            # Mirror RecordedIMUDataset._fit_to_window for a positive segment.
            local_label = frame - start
            fit_start = max(0, min(local_label - 200, end - start - training_window_frames))
            retains = fit_start <= local_label < fit_start + training_window_frames
            row.update(true_frame=frame, label_in_crop=local_label, score_start_frame=lo,
                       score_end_frame=hi, window_clipped=lo != frame - before or hi != frame + after + 1,
                       status='labeled', is_hard=bool(hard_models), model=representative['model'],
                       trigger_models=hard_models, score_details=details, training_retains_event=retains)
            scores.extend(dict(candidate_id=index, file=rec.path.name, segment_index=sid,
                               class_id=cls, **detail) for detail in details)
        rows.append(row)
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS), pd.DataFrame(scores, columns=SCORE_COLUMNS)


def mine_recordings(paths: list[Path], models: dict[str, MiningModel],
                    thresholds: dict[str, tuple[float, float]],
                    score_before_ms: float = 500., score_after_ms: float = 500.,
                    reset_at_segments: bool = True, training_window_frames: int = 300,
                    chunk_frames: int = 4096, progress=None):
    """Return hard union candidates, all segments, per-model scores, file stats.

    Source copies are deduplicated by raw hash. Conflicting labels on identical
    raw inputs are errors, never silently deduplicated. All-easy files are shown.
    """
    if not models or set(models) != set(thresholds):
        raise ValueError('Choose the same nonempty model set for models and thresholds')
    tables, score_tables, stats, seen = [], [], [], {}
    offset = 0
    for index, path in enumerate(paths):
        rec = load_recording(path)
        _positive_class(rec)
        label_hash = file_sha256(rec.path.with_suffix('.txt'))
        if rec.sha256 in seen:
            if seen[rec.sha256] != label_hash:
                raise ValueError(f'{rec.path.name}: identical source has conflicting labels')
            continue
        seen[rec.sha256] = label_hash
        if progress is not None:
            progress(f'[{index + 1}/{len(paths)}] {rec.path.name}')
        traces = {name: predict_positive_recording(rec, model, reset_at_segments, chunk_frames)
                  for name, model in models.items()}
        segments, scores = score_segments(rec, traces, thresholds, score_before_ms, score_after_ms,
                                          reset_at_segments, training_window_frames)
        segments['candidate_id'] += offset
        scores['candidate_id'] += offset
        offset += len(segments)
        provenance = [dict(model=name, checkpoint_path=str(model.path), checkpoint_sha256=model.sha256,
                           highpass=model.highpass, left_threshold=float(thresholds[name][0]),
                           right_threshold=float(thresholds[name][1])) for name, model in models.items()]
        segments['mining_models'] = [provenance for _ in range(len(segments))]
        tables.append(segments)
        score_tables.append(scores)
        hard = segments[segments.is_hard]
        stats.append(dict(file=rec.path.name, source_sha256=rec.sha256, segments=len(segments),
                          labeled_segments=int(segments.status.eq('labeled').sum()),
                          hard_segments=len(hard),
                          easy_segments=int((segments.status.eq('labeled') & ~segments.is_hard).sum()),
                          unlabeled_segments=int(segments.status.eq('unlabeled').sum()),
                          ambiguous_segments=int(segments.status.eq('ambiguous').sum()),
                          recording_seconds=len(rec.data) / rec.fs,
                          hard_segment_seconds=float(hard.crop_seconds.sum())))
    all_segments = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=SEGMENT_COLUMNS + ['mining_models'])
    model_scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame(columns=SCORE_COLUMNS)
    return all_segments[all_segments.is_hard.astype(bool)].copy(), all_segments, model_scores, pd.DataFrame(stats)


def export_batch(selected: pd.DataFrame, input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Export hard whole-segment CSV/positive label/provenance triplets.

    Checks all plans before writing, including raw/label hashes and idempotency.
    No positive action is renamed to negative_mining. Dense labels are generated
    by the training loader from the remapped sparse label.
    """
    output_dir = Path(output_dir).resolve()
    plans, rows, seen_paths = [], [], set()
    for file, group in selected.groupby('file', sort=True) if not selected.empty else []:
        if Path(file).name != file:
            raise ValueError('Choose source basenames, not paths')
        rec = load_recording(Path(input_dir) / file)
        cls = _positive_class(rec)
        label_hash = file_sha256(rec.path.with_suffix('.txt'))
        for selection in group.to_dict('records'):
            if selection['source_sha256'] != rec.sha256 or selection['source_labels_sha256'] != label_hash:
                raise ValueError('Source recording or labels changed; rerun mining')
            if not selection['is_hard'] or selection['status'] != 'labeled' or not selection['trigger_models']:
                raise ValueError('Export only unambiguously labeled hard segments')
            if not selection['training_retains_event']:
                raise ValueError('Training window would lose the positive label')
            start, end = int(selection['crop_start_frame']), int(selection['crop_end_frame'])
            if (int(selection['segment_index']), start, end) not in rec.segments or end - start < 2:
                raise ValueError('Positive export must be one complete original segment')
            events = [(frame, side) for frame, side in rec.events if start <= frame < end]
            if events != [(int(selection['true_frame']), cls)] or int(selection['class_id']) != cls:
                raise ValueError('Positive segment label differs from selection')
            crop = rec.data.iloc[start:end].copy()
            crop['timestamp_ms'] -= crop.timestamp_ms.iloc[0]
            crop['segment_index'] = 1
            crop['phase'] = 'recording'
            stem = f'{rec.path.stem}__crop_f{start:06d}_f{end:06d}'
            csv_path = output_dir / f'{stem}.csv'
            paths = (csv_path, csv_path.with_suffix('.txt'), csv_path.with_suffix('.mining.json'))
            if csv_path in seen_paths:
                raise ValueError('Duplicate segment selected for export')
            seen_paths.add(csv_path)
            labeling_path = rec.path.with_suffix('.labeling.json')
            labeling = json.loads(labeling_path.read_text()) if labeling_path.is_file() else None
            if labeling is not None and (labeling.get('source_sha256') != rec.sha256
                                         or labeling.get('source_labels_sha256') != label_hash):
                raise ValueError('Strict labeling audit differs from current source/labels; update the audit before exporting')
            metadata = dict(mining_kind='hard_positive',
                            selection_rule='any_model_correct_side_peak_below_threshold', source_path=str(rec.path),
                            source_sha256=rec.sha256, source_labels_sha256=label_hash,
                            source_frames=len(rec.data), source_sample_rate_hz=rec.fs,
                            source_duration_seconds=len(rec.data) / rec.fs, source_recording_type='knock_twice',
                            label_class=cls, labeling=labeling,
                            crop_bounds='[start, end), complete original segment', selection=selection)
            contents = (crop.to_csv(index=False, float_format='%.9f').encode(),
                        f'{events[0][0] - start}, {cls}\n'.encode(),
                        (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
            exists = any(path.exists() for path in paths)
            if exists and not all(path.is_file() and path.read_bytes() == content
                                  for path, content in zip(paths, contents)):
                raise FileExistsError(f'Different or incomplete positive crop exists: {csv_path}')
            plans.append((paths, contents, exists))
            rows.append(dict(candidate_id=int(selection['candidate_id']), file=file, csv=str(paths[0]),
                             label=str(paths[1]), provenance=str(paths[2]), frames=end - start,
                             seconds=(end - start) / rec.fs, label_in_crop=events[0][0] - start,
                             class_id=cls, status='already exported' if exists else 'exported'))
    if plans:
        output_dir.mkdir(parents=True, exist_ok=True)
        for paths, contents, exists in plans:
            if not exists:
                for path, content in zip(paths, contents):
                    path.write_bytes(content)
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS).sort_values('candidate_id')


def plot_segment(rec: MiningRecording, traces: dict[str, np.ndarray],
                  thresholds: dict[str, tuple[float, float]], row):
    """Detailed raw-sensor and all-model view of one hard (or easy) segment."""
    if row['file'] != rec.path.name or row['source_sha256'] != rec.sha256:
        raise ValueError('Choose a segment belonging to this source recording')
    start, end = int(row['crop_start_frame']), int(row['crop_end_frame'])
    t = np.arange(end - start) / rec.fs
    fig, axes = plt.subplots(2 + len(traces), 1, sharex=True,
                             figsize=(14, 2.5 * (2 + len(traces))), layout='constrained')
    for axis, columns, title in ((axes[0], IMU_COLUMNS[:3], 'Acceleration'),
                                 (axes[1], IMU_COLUMNS[3:], 'Gyroscope')):
        for column in columns:
            axis.plot(t, rec.data[column].iloc[start:end], lw=.8, label=column)
        axis.set_ylabel(title)
        axis.legend(loc='upper right', ncol=3, fontsize=8)
    cls = int(row['class_id'])
    for axis, (name, trace) in zip(axes[2:], traces.items()):
        for side, label in ((1, 'left'), (2, 'right')):
            axis.plot(t, trace[start:end, side], label=f'P({label})', lw=1)
        axis.axhline(thresholds[name][cls - 1], ls=':', color='red', label='Correct-side threshold')
        detail = next((d for d in row['score_details'] if d['model'] == name), None)
        if detail:
            axis.scatter((detail['peak_frame'] - start) / rec.fs, detail['peak_probability'], color='red')
            axis.set_title(f'{name}: correct-side peak={detail["peak_probability"]:.3f}; '
                           f'threshold={detail["threshold"]:.3f}; '
                           f'{"HARD" if detail["is_hard"] else "easy"}', fontsize=10)
        axis.set(ylabel=name, ylim=(-.03, 1.05))
        axis.legend(loc='upper left', ncol=3, fontsize=8)
    for axis in axes:
        if pd.notna(row['true_frame']):
            axis.axvline((row['true_frame'] - start) / rec.fs, color='green', ls='--')
            axis.axvspan((row['score_start_frame'] - start) / rec.fs,
                          (row['score_end_frame'] - start) / rec.fs, color='gray', alpha=.12)
        axis.grid(alpha=.2)
    axes[0].set_title(f'{rec.path.name} | segment {row["segment_index"]} | '
                      f'candidate {row["candidate_id"]} | {row["side"]} | '
                      f'{"HARD" if row["is_hard"] else row["status"]}; '
                      'green: label, gray: scoring window')
    axes[-1].set_xlabel('Seconds from segment start')
    return fig, axes
