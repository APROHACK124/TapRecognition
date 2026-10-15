"""Verified CPU prediction caches and exact alarm replay for fast_comparison.

The reference comparison functions retain all scoring/aggregation logic. Their
optional runtime hooks call this module only for input reuse, predictions, and
alarm emission. Window inference stays independent of continuous inference; no
batching, precision changes, GPU changes, or reset-boundary changes are made.
"""

from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import torch

from notebooks_analysis import model_comparison_examples as examples
from notebooks_analysis import normal_usage_evaluation as normal
from notebooks_analysis.fast_pair_evaluation import PreparedTrace
from notebooks_analysis.hard_negative_mining import MiningModel, predict_recording
from notebooks_analysis.recording_durations import _original_duration
from tap_recognition.model_factory import build_model


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default).encode()).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value)
    return dict(shape=array.shape, dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


class ComparisonRuntime:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else None
        self._files, self._json, self._durations = {}, {}, {}
        self._checkpoints, self._models = {}, {}
        self._inputs, self._normal_inputs, self._predictions = {}, {}, {}
        self._detectors = OrderedDict()
        self.stats = dict(prediction_computes=0, prediction_disk_hits=0, prediction_memory_hits=0,
                          model_forwards=0, prepared_input_hits=0, file_hash_computes=0)
        root = Path(__file__).resolve().parents[1]
        dependencies = [*sorted((root / 'notebooks_analysis').glob('*.py')),
                        *sorted((root / 'tap_recognition').glob('*.py')), root / 'train.py']
        self.code_hash = _digest([(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                                  for p in dependencies])

    def _backend(self):
        return dict(code=self.code_hash, torch=torch.__version__, numpy=np.__version__,
                    scipy=scipy.__version__, machine=platform.machine(),
                    cpu=torch.backends.cpu.get_cpu_capability(), threads=torch.get_num_threads(),
                    interop_threads=torch.get_num_interop_threads(), mkldnn=torch.backends.mkldnn.enabled,
                    deterministic=torch.are_deterministic_algorithms_enabled())

    def file_sha256(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        previous = self._files.get(path)
        if previous is None or previous[0] != stamp:
            self._files[path] = (stamp, hashlib.sha256(path.read_bytes()).hexdigest())
            self.stats['file_hash_computes'] += 1
        return self._files[path][1]

    def read_json(self, path):
        key = (str(Path(path).resolve()), self.file_sha256(path))
        if key not in self._json:
            self._json[key] = json.loads(Path(path).read_text())
        return deepcopy(self._json[key])

    def original_duration(self, path):
        key = self.file_sha256(path)
        if key not in self._durations:
            self._durations[key] = _original_duration(path)
        return self._durations[key]

    def checkpoint(self, path, expected_sha256):
        if self.file_sha256(path) != expected_sha256:
            raise ValueError(f'{path}: checkpoint differs from selected-threshold export')
        if expected_sha256 not in self._checkpoints:
            self._checkpoints[expected_sha256] = torch.load(path, map_location='cpu', weights_only=True)
        return self._checkpoints[expected_sha256]

    def _model(self, checkpoint, digest):
        key = (digest, _digest(self._backend()))
        if key not in self._models:
            config = dict(checkpoint['model_config'])
            config['dilations'] = tuple(config['dilations'])
            model = build_model(config, checkpoint.get('model_type')).cpu().eval()
            model.load_state_dict(checkpoint['model_state'])
            self._models[key] = model
        return self._models[key]

    def _file_manifest(self, paths):
        return [(str(Path(p).resolve()), self.file_sha256(p) if Path(p).is_file() else None) for p in paths]

    def load_examples(self, root, data, labels, seed, split):
        paths = []
        for folder in data['train_dir' if split == 'train' else 'val_dir'].split(','):
            directory = examples._local_data_path(root, folder.strip())
            for path in sorted(directory.glob('*.csv')):
                if not path.name.endswith('.labels.csv'):
                    paths.extend((path, path.with_suffix('.txt')))
        if data.get('session_exclusions_file'):
            paths.append(examples._local_data_path(root, data['session_exclusions_file']))
        key = _digest(dict(data=data, labels=labels, seed=seed, split=split,
                           files=self._file_manifest(paths)))
        if key not in self._inputs:
            self._inputs[key] = examples.load_examples(root, data, labels, seed, split)
        else:
            self.stats['prepared_input_hits'] += 1
        return self._inputs[key]

    def load_normal_usage(self, source_dir, filenames):
        directory = Path(source_dir).resolve()
        paths = ([directory / name for name in filenames] if filenames is not None else
                 sorted(p for p in directory.glob('*.csv') if 'normal_usage' in p.name.lower()))
        files = [p for path in paths for p in (path, path.with_suffix('.txt'))]
        key = _digest(dict(directory=directory, filenames=filenames, files=self._file_manifest(files)))
        if key not in self._normal_inputs:
            self._normal_inputs[key] = normal.load_normal_usage(directory, filenames)
        else:
            self.stats['prepared_input_hits'] += 1
        return self._normal_inputs[key]

    def _prediction(self, kind, identity, compute):
        key = _digest(dict(kind=kind, identity=identity, backend=self._backend()))
        if key in self._predictions:
            self.stats['prediction_memory_hits'] += 1
            return self._predictions[key]
        path = self.directory / kind / f'{key}.npz' if self.directory is not None else None
        if path is not None and path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                result = {name: archive[name] for name in archive.files}
            self.stats['prediction_disk_hits'] += 1
        else:
            result = compute()
            self.stats['prediction_computes'] += 1
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez(handle, **result)
                temporary.replace(path)
        for array in result.values():
            array.flags.writeable = False
        self._predictions[key] = result
        return result

    def example_predictions(self, checkpoint, digest, rec):
        identity = dict(checkpoint=digest, imu=_array_digest(rec.imu), bounds=rec.bounds,
                        windows=[(sid, _array_digest(x)) for sid, (x, _) in rec.windows.items()])

        @torch.inference_mode()
        def compute():
            model = self._model(checkpoint, digest)
            probs = np.zeros((len(rec.imu), 3), dtype=np.float32)
            probs[:, 0] = 1
            for start, end in rec.bounds:
                logits, _ = model(torch.from_numpy(rec.imu[start:end]).unsqueeze(0))
                probs[start:end] = logits[0].softmax(-1).numpy()
                self.stats['model_forwards'] += 1
            scores = []
            for x, _ in rec.windows.values():
                logits, _ = model(torch.from_numpy(x).unsqueeze(0))
                scores.append(logits[0].softmax(-1)[:, 1:].max(dim=0).values.tolist())
                self.stats['model_forwards'] += 1
            return dict(probabilities=probs, window_ids=np.array(list(rec.windows), dtype=np.int64),
                        window_scores=np.asarray(scores, dtype=float).reshape(-1, 2))

        arrays = self._prediction('examples', identity, compute)
        return arrays['probabilities'], dict(zip(arrays['window_ids'].tolist(), arrays['window_scores'].tolist()))

    def normal_prediction(self, path, checkpoint, digest, evaluation_data, rec, chunk_frames):
        data = checkpoint['train_config']['data']
        highpass = bool(evaluation_data['highpass']) if isinstance(evaluation_data, dict) else bool(data['highpass'])
        identity = dict(checkpoint=digest, source=rec.sha256, fs=rec.fs, highpass=highpass,
                        window_samples=int(data['window_samples']), frames=len(rec.data),
                        chunk_frames=chunk_frames, reset_at_segments=False)

        def compute():
            spec = MiningModel(Path(path), self._model(checkpoint, digest), highpass, int(data['window_samples']), digest)
            result = predict_recording(rec, spec, False, chunk_frames)
            self.stats['model_forwards'] += (len(rec.data) + chunk_frames - 1) // chunk_frames
            return dict(probabilities=result)

        return self._prediction('normal_usage', identity, compute)['probabilities']

    def emit_alarms_pair(self, probs, left_threshold, right_threshold, look, refractory, bounds):
        # Preserve separate resets/EOF handling for each INCLUDED contiguous bound.
        alarms = []
        for start, end in bounds:
            key = (id(probs), start, end)
            if key not in self._detectors:
                self._detectors[key] = (probs, PreparedTrace(probs[start:end]))
                if len(self._detectors) > 64:
                    self._detectors.popitem(last=False)
            self._detectors.move_to_end(key)
            prepared = self._detectors[key][1]
            for alarm in prepared.emit(left_threshold, right_threshold, look, refractory):
                alarms.append(dict(alarm_frame=start + alarm['alarm_frame'], peak_frame=start + alarm['peak_frame'],
                                   class_id=alarm['class_id'], probability=alarm['probability']))
        return alarms

    def timed(self, label, function, *args, **kwargs):
        before = dict(self.stats)
        start = perf_counter()
        result = function(*args, **kwargs)
        delta = {key: value - before[key] for key, value in self.stats.items()}
        print(f'{label}: {perf_counter() - start:.2f} s | '
              f"predictions computed={delta['prediction_computes']}, "
              f"disk hits={delta['prediction_disk_hits']}, memory hits={delta['prediction_memory_hits']}")
        return result
