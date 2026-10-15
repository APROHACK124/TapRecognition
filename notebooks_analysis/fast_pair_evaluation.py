"""Exact candidate-based replay and reusable metrics for evaluate_fast.ipynb.

The reference detector lives in model_comparison_examples.py. This implementation
skips quiet frames but preserves qualifying peaks, earliest-frame/left ties,
lookahead, shared peak-based cooldown, and the absence of an end-of-file flush.
No model, matching rule, threshold selection, or duration definition is changed.
"""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import __version__ as scipy_version


ALGORITHM_VERSION = 'candidate_pair_v1'


class PreparedTrace:
    """Immutable side scores, threshold crossings, and forward-window maxima."""

    def __init__(self, probabilities):
        p = np.asarray(probabilities)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError('Expected finite [frames, 3] probabilities')
        # The reference compares float32 scores to a float64 threshold ARRAY.
        # Explicit promotion avoids NumPy scalar-promotion differences at equality.
        self.scores = np.array(p[:, 1:], dtype=np.float64, copy=True)
        self.scores.flags.writeable = False
        self.crossings = {}
        self.windows = {}
        self.count_cache = {}

    def _crossings(self, side, threshold):
        key = (side, float(threshold))
        if key not in self.crossings:
            self.crossings[key] = np.flatnonzero(self.scores[:, side] >= threshold)
        return self.crossings[key]

    def _windows(self, look):
        if look not in self.windows:
            n = max(0, len(self.scores) - look)
            maxima = self.scores[:n].copy()
            peaks = np.broadcast_to(np.arange(n)[:, None], (n, 2)).copy()
            for offset in range(1, look + 1):
                values = self.scores[offset:offset + n]
                greater = values > maxima  # Strict > retains the earliest peak on ties.
                maxima[greater] = values[greater]
                positions = np.broadcast_to((np.arange(n) + offset)[:, None], (n, 2))
                peaks[greater] = positions[greater]
            self.windows[look] = maxima, peaks
        return self.windows[look]

    def events(self, left_threshold, right_threshold, look, refractory):
        """Yield (emission, peak, class, probability) without per-frame Python work."""
        if int(look) != look or look < 0 or int(refractory) != refractory:
            raise ValueError('look must be nonnegative and look/refractory must be integers')
        look, refractory = int(look), max(1, int(refractory))
        left_threshold, right_threshold = float(left_threshold), float(right_threshold)
        if not np.isfinite([left_threshold, right_threshold]).all():
            raise ValueError('Thresholds must be finite')
        left = self._crossings(0, left_threshold)
        right = self._crossings(1, right_threshold)
        maxima, peaks = self._windows(look)
        cursor, n = 0, len(self.scores)
        while cursor < len(maxima):
            li = int(np.searchsorted(left, cursor))
            ri = int(np.searchsorted(right, cursor))
            crossing = min(int(left[li]) if li < len(left) else n,
                           int(right[ri]) if ri < len(right) else n)
            if crossing >= len(maxima):
                break
            lp, rp = maxima[crossing]
            lf, rf = peaks[crossing]
            if lp < left_threshold:
                side = 1
            elif rp < right_threshold:
                side = 0
            elif lp == rp:
                # Reference flattens [frame, side]: earliest frame, then left.
                side = 0 if lf <= rf else 1
            else:
                side = 0 if lp > rp else 1
            peak = int(peaks[crossing, side])
            emission = crossing + look
            yield emission, peak, side + 1, float(maxima[crossing, side])
            cursor = max(emission + 1, peak + refractory)

    def emit(self, left_threshold, right_threshold, look, refractory):
        return [dict(alarm_frame=emission, peak_frame=peak, class_id=cls, probability=p)
                for emission, peak, cls, p in self.events(
                    left_threshold, right_threshold, look, refractory)]

    def counts(self, left_threshold, right_threshold, look, refractory):
        key = (float(left_threshold), float(right_threshold), int(look), int(refractory))
        if key not in self.count_cache:
            counts = [0, 0]
            for _, _, cls, _ in self.events(*key):
                counts[cls - 1] += 1
            self.count_cache[key] = tuple(counts)
        return self.count_cache[key]


class TracePool:
    """Notebook-lifetime preparation; retain input identity to prevent id reuse.

    Recreate the pool after changing probabilities (the copy notebook does this
    when its detector cell is rerun). Input arrays must not be mutated in place.
    """

    def __init__(self):
        self.traces = {}

    def get(self, probabilities):
        key = id(probabilities)
        if key not in self.traces:
            self.traces[key] = (probabilities, PreparedTrace(probabilities))
        return self.traces[key][1]

    def emit(self, probabilities, left_threshold, right_threshold, look, refractory):
        return self.get(probabilities).emit(left_threshold, right_threshold, look, refractory)


def normal_usage_counts(pool, recordings, traces, left_threshold, right_threshold,
                        lookahead_frames=12, refractory_sec=.5):
    """Reference summary dictionaries, without alarm dictionaries or DataFrames.

    Repeated population subsets reuse each trace's side counts. Population order,
    zero-alarm exposure, and empty-population NaNs match evaluate_normal_usage.
    Full originals remain continuous through internal source sessions.
    """
    minutes, counts = [], []
    for rec in recordings:
        if rec.events:
            raise ValueError(f'{rec.path.name}: normal_usage originals must contain no annotated double taps')
        p = traces[rec.path.name]
        if p.shape != (len(rec.data), 3):
            raise ValueError(f'{rec.path.name}: expected full-source [frames, 3] probabilities')
        prepared = pool.get(p)
        counts.append(prepared.counts(left_threshold, right_threshold, lookahead_frames,
                                       round(refractory_sec * rec.fs)))
        minutes.append(len(p) / rec.fs / 60)
    # Match pandas/NumPy's reduction order, rather than Python's sequential sum.
    exposure = float(np.asarray(minutes, dtype=float).sum())
    side_fp = np.asarray(counts, dtype=np.int64).reshape(-1, 2).sum(axis=0)
    fp = int(side_fp.sum())
    overall = dict(normal_usage_recordings=len(recordings), normal_usage_minutes=exposure,
                   normal_usage_fp=fp, normal_usage_fp_per_min=fp / exposure if exposure else np.nan)
    sides = {cls: dict(normal_usage_minutes=exposure, normal_usage_fp=int(side_fp[cls - 1]),
                       normal_usage_fp_per_min=float(side_fp[cls - 1]) / exposure if exposure else np.nan)
             for cls in (1, 2)}
    return overall, sides, None, None


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default).encode()).hexdigest()


def sweep_context(recordings, probabilities, duration_audit, populations, settings):
    """Content-based cache identity; never rely on just checkpoint/source names.

    Include actual probability bytes, truth, sample rates, ordered populations,
    exposure, metric code and timing settings. Selection modes/budgets and the
    grid are excluded: those do not change a single pair's metrics.
    """
    def trace_rows(recs, traces):
        rows = []
        for rec in recs:
            p = np.ascontiguousarray(traces[rec.path.name])
            rows.append(dict(file=rec.path.name, fs=rec.fs, events=rec.events,
                             shape=p.shape, dtype=p.dtype.str,
                             trace=hashlib.sha256(p.tobytes()).hexdigest()))
        return rows

    return _digest(dict(version=ALGORITHM_VERSION,
                        implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        numpy_version=np.__version__, scipy_version=scipy_version,
                        settings=settings,
                        validation=trace_rows(recordings, probabilities),
                        duration=duration_audit.to_dict('split'),
                        populations=[trace_rows(recs, traces) for recs, traces in populations]))


class PairMetricsCache:
    """Memoized metric results with optional atomic, JSON-only disk persistence."""

    def __init__(self, directory, context):
        self.directory = Path(directory) / context if directory is not None else None
        self.memory = {}
        self.hits = self.disk_hits = self.misses = 0

    def memoize(self, function):
        @wraps(function)
        def cached(left_threshold, right_threshold, policy):
            key = (float(left_threshold), float(right_threshold), policy)
            if key in self.memory:
                self.hits += 1
                return deepcopy(self.memory[key])
            path = self.directory / f'{_digest(key)}.json' if self.directory is not None else None
            if path is not None and path.is_file():
                payload = json.loads(path.read_text())
                result = payload['overall'], {int(cls): row for cls, row in payload['sides'].items()}
                self.disk_hits += 1
            else:
                result = function(*key)
                self.misses += 1
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = dict(overall=result[0], sides=result[1])
                    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp',
                                                     delete=False, encoding='utf-8') as handle:
                        temporary = Path(handle.name)
                        json.dump(payload, handle, default=_json_default)
                    temporary.replace(path)
            self.memory[key] = deepcopy(result)
            return deepcopy(result)
        return cached
