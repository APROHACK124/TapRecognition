"""Exact comparison replay, cold/disk caches, reset boundaries and invalidation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from notebooks_analysis.fast_comparison import ComparisonRuntime
from notebooks_analysis.model_comparison_examples import emit_alarms_pair, replay_examples
from notebooks_analysis.normal_usage_evaluation import replay_normal_usage
from notebooks_analysis.recording_durations import recording_durations
from tap_recognition.config import LSTMModelConfig, ModelConfig, TrainConfig
from tap_recognition.model_factory import build_model
from tests import test_model_comparison_examples as fixtures
from tests import test_recording_durations as duration_fixtures
from tools.build_fast_comparison_notebook import build_notebook


ROOT = Path(__file__).resolve().parents[1]


def same_table(test, expected, actual):
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    test.assertEqual(expected.to_csv(index=False), actual.to_csv(index=False))


class FastComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / 'cache'
        self.data, self.labels, self.names = fixtures.ModelComparisonExamplesTests()._write_recordings(self.root)
        rows = []
        for kind, config in (
            ('gru', ModelConfig(cnn_channels=4, gru_hidden=8, kernel_size=3, dilations=(1,), dropout=0.)),
            ('lstm', LSTMModelConfig(cnn_channels=4, lstm_hidden=8, kernel_size=3, dilations=(1,), dropout=0.)),
        ):
            model = build_model(config.to_model_kwargs())
            with torch.no_grad():
                model.head[-1].weight.zero_()
                model.head[-1].bias.copy_(torch.tensor([0., 8., 0.]))
            path = self.root / kind / 'best.pt'
            path.parent.mkdir()
            torch.save(dict(model_type=kind, model_config=config.to_model_kwargs(), model_state=model.state_dict(),
                            train_config={**TrainConfig(model=config).to_dict(), 'data': self.data, 'labels': self.labels}), path)
            rows.append(dict(model=kind, checkpoint_path=f'{kind}/best.pt',
                             checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             validation_sha256='historical', evaluation_data=self.data, evaluation_labels=self.labels,
                             evaluation_seed=42, policy='instant', left_threshold=.9, right_threshold=.95,
                             refractory_sec=.05, early_allowance_frames=1, max_late_frames=30,
                             selection_mode='max_f1'))
        self.points = pd.DataFrame(rows)

    def assert_replay(self, expected, actual):
        self.assertEqual(expected[0].keys(), actual[0].keys())
        self.assertEqual(expected[1].keys(), actual[1].keys())
        for key in expected[1]:
            np.testing.assert_array_equal(expected[1][key], actual[1][key])
        for left, right in zip(expected[2:], actual[2:]):
            same_table(self, left, right)

    def test_both_splits_cold_and_fresh_disk_replay_are_exact(self):
        cold = ComparisonRuntime(self.cache)
        for split in ('valid', 'train'):
            expected = replay_examples(self.root, self.points, split=split)
            actual = replay_examples(self.root, self.points, split=split, runtime=cold)
            self.assert_replay(expected, actual)
            warm = ComparisonRuntime(self.cache)
            with patch.object(warm, '_model', side_effect=AssertionError('Warm cache must not run a model')):
                self.assert_replay(expected, replay_examples(self.root, self.points, split=split, runtime=warm))
            self.assertEqual(warm.stats['model_forwards'], 0)
            self.assertEqual(warm.stats['prediction_disk_hits'], 2)
        self.assertEqual(cold.stats['prediction_computes'], 4)

    def test_changing_thresholds_and_matching_reuses_predictions(self):
        runtime = ComparisonRuntime(self.cache)
        replay_examples(self.root, self.points, runtime=runtime)
        original_forwards = runtime.stats['model_forwards']
        changed = self.points.copy()
        changed['left_threshold'] = .99999
        changed['early_allowance_frames'] = 25
        changed['max_late_frames'] = 40
        expected = replay_examples(self.root, changed)
        actual = replay_examples(self.root, changed, runtime=runtime)
        self.assert_replay(expected, actual)
        self.assertEqual(runtime.stats['model_forwards'], original_forwards)
        self.assertEqual(runtime.stats['prepared_input_hits'], 1)

    def test_reset_gaps_ties_and_short_eof_bounds(self):
        runtime = ComparisonRuntime()
        rng = np.random.default_rng(131)
        for frames in (12, 55, 500):
            p = rng.choice([0., .5, .7, .9], (frames, 3)).astype(np.float32)
            bounds = [(0, 4), (6, frames)]
            for look in (0, 1, 12):
                self.assertEqual(emit_alarms_pair(p, .5, .8, look, 50, bounds),
                                 runtime.emit_alarms_pair(p, .5, .8, look, 50, bounds))

    def test_input_label_and_registry_changes_invalidate_prepared_inputs(self):
        runtime = ComparisonRuntime()
        recs, digest = runtime.load_examples(self.root, self.data, self.labels, 42, 'valid')
        path = self.root / 'valid' / self.names['valid']
        label = path.with_suffix('.txt')
        label.write_text('5, 2\n30, 2\n')
        changed, new_digest = runtime.load_examples(self.root, self.data, self.labels, 42, 'valid')
        self.assertNotEqual(digest, new_digest)
        self.assertIsNot(changed, recs)
        registry = self.root / 'exclusions.csv'
        registry.write_text(registry.read_text() + f'{path.name},1,new exclusion\n')
        excluded, excluded_digest = runtime.load_examples(self.root, self.data, self.labels, 42, 'valid')
        self.assertNotEqual(new_digest, excluded_digest)
        self.assertEqual([sid for sid, _, _ in excluded[path.name].segments], [0, 2])

    def test_inference_inputs_and_boundaries_invalidate_predictions(self):
        runtime = ComparisonRuntime()
        recs, _ = runtime.load_examples(self.root, self.data, self.labels, 42, 'valid')
        rec = next(iter(recs.values()))
        row = self.points.iloc[0]
        ckpt = runtime.checkpoint(self.root / row.checkpoint_path, row.checkpoint_sha256)
        runtime.example_predictions(ckpt, row.checkpoint_sha256, rec)
        rec.imu[0, 0] += 1
        runtime.example_predictions(ckpt, row.checkpoint_sha256, rec)
        rec.segments = [segment for segment in rec.segments if segment[0] != 1]
        runtime.example_predictions(ckpt, row.checkpoint_sha256, rec)
        self.assertEqual(runtime.stats['prediction_computes'], 3)

    def test_checkpoint_and_source_verification_are_not_bypassed_by_cache(self):
        runtime = ComparisonRuntime()
        row = self.points.iloc[0]
        path = self.root / row.checkpoint_path
        runtime.checkpoint(path, row.checkpoint_sha256)
        with path.open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            runtime.checkpoint(path, row.checkpoint_sha256)
        fixture = duration_fixtures.RecordingDurationTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        source = fixture.source()
        inputs = [fixture.crop(source), fixture.crop(source, 1000, 1300)]
        expected = recording_durations(fixture.root, inputs)
        same_table(self, expected, recording_durations(fixture.root, inputs, cache=runtime))
        hashes = runtime.stats['file_hash_computes']
        same_table(self, expected, recording_durations(fixture.root, inputs, cache=runtime))
        self.assertEqual(hashes, runtime.stats['file_hash_computes'])
        source.write_text('timestamp_ms\n0\n10\n')
        with self.assertRaisesRegex(FileNotFoundError, 'hash-matching original'):
            recording_durations(fixture.root, inputs, cache=runtime)

    def test_full_normal_replay_and_empty_population(self):
        source_dir = self.root / 'sources'
        source_dir.mkdir()
        frames = np.arange(120)
        path = source_dir / 'imu_fixture_normal_usage_20261015_120000.csv'
        pd.DataFrame(dict(timestamp_ms=frames * 10, segment_index=np.where(frames < 60, 1, 2),
                          user_name='fixture', action_label='normal_usage',
                          **{key: np.sin(frames / 3) for key in ('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')})).to_csv(path, index=False)
        path.with_suffix('.txt').write_text('')
        copied = source_dir / 'imu_fixture_normal_usage_20261015_120001.csv'
        copied.write_bytes(path.read_bytes())
        expected = replay_normal_usage(self.root, self.points, source_dir, chunk_frames=32)
        for runtime in (ComparisonRuntime(self.cache), ComparisonRuntime(self.cache)):
            actual = replay_normal_usage(self.root, self.points, source_dir, chunk_frames=32, runtime=runtime)
            for left, right in zip(expected[:4], actual[:4]):
                same_table(self, left, right)
            self.assertEqual(expected[4], actual[4])
            self.assertEqual(actual[0].normal_usage_recordings.tolist(), [1, 1])
        self.assertEqual(runtime.stats['model_forwards'], 0)
        empty = replay_normal_usage(self.root, self.points, source_dir, [], runtime=runtime)
        self.assertTrue(all(table.empty for table in empty[:4]))

    def test_new_notebook_wiring_compiles_and_keeps_model_selection(self):
        original = json.loads((ROOT / 'main_notebooks/compare.ipynb').read_text())
        copy = build_notebook(original)
        actual = json.loads((ROOT / 'main_notebooks/fast_comparison.ipynb').read_text())
        self.assertEqual(actual, copy)
        source = ''.join(original['cells'][1]['source'])
        setup = ''.join(copy['cells'][1]['source'])
        self.assertEqual(source.split('RUNS = ', 1)[1], setup.split('RUNS = ', 1)[1])
        for index, cell in enumerate(copy['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'fast-comparison-cell{index}', 'exec')
                self.assertEqual(cell['outputs'], [])


if __name__ == '__main__':
    unittest.main()
