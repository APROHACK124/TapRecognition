"""Normal-use metrics must use complete originals, not failure-enriched crops."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from notebooks_analysis.normal_usage_evaluation import (
    evaluate_normal_usage, load_normal_usage, replay_normal_usage,
)
from notebooks_analysis.recording_durations import recording_durations
from tap_recognition.recording import IMU_COLUMNS


class NormalUsageEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source_dir = self.root/'sources'
        self.source_dir.mkdir()

    def write_source(self, stamp='20261014_120000', frames=6000, offset=0):
        path = self.source_dir/f'imu_fixture_normal_usage_{stamp}.csv'
        t = np.arange(frames)
        pd.DataFrame(dict(timestamp_ms=offset+t*10, user_name='fixture', action_label='normal_usage',
                          segment_index=np.where(t < frames//2, 1, 2),
                          **{c: np.zeros(frames) for c in IMU_COLUMNS})).to_csv(path, index=False)
        path.with_suffix('.txt').write_text('')
        return path

    def test_full_originals_only_hash_dedup_and_zero_alarm_minutes(self):
        first = self.write_source()
        second = self.write_source('20261014_120001', offset=10)
        copied = self.source_dir/'imu_fixture_normal_usage_20261014_120002.csv'
        copied.write_bytes(first.read_bytes())
        crop = self.source_dir/f'{first.stem}__crop_f000000_f000300.csv'
        crop.write_bytes(first.read_bytes())  # Discovery must exclude even a plausible crop CSV.
        recordings, manifest, digest = load_normal_usage(self.source_dir)
        self.assertEqual(len(recordings), 2)
        self.assertAlmostEqual(manifest.minutes.sum(), 2.)
        traces = {rec.path.name: np.tile([1., 0., 0.], (6000, 1)) for rec in recordings}
        traces[first.name][5000] = [0., 1., 0.]  # Well outside the example crop.
        overall, sides, files, alarms = evaluate_normal_usage(recordings, traces, .9, .9, 0)
        self.assertEqual(overall['normal_usage_fp'], 1)
        self.assertEqual(overall['normal_usage_minutes'], 2.)
        self.assertEqual(overall['normal_usage_fp_per_min'], .5)
        self.assertEqual(sides[1]['normal_usage_fp_per_min'], .5)
        self.assertEqual(files.event_fp.tolist(), [1, 0])
        self.assertEqual(alarms.alarm_frame.tolist(), [5000])
        self.assertEqual(digest, load_normal_usage(self.source_dir)[2])

    def test_state_and_cooldown_are_not_reset_at_source_sessions(self):
        path = self.write_source(frames=600)
        recordings, _, _ = load_normal_usage(self.source_dir)
        p = np.tile([1., 0., 0.], (600, 1)); p[299] = [0., 1., 0.]; p[301] = [0., 1., 0.]
        overall, _, _, alarms = evaluate_normal_usage(recordings, {path.name: p}, .9, .9, 12)
        self.assertEqual(overall['normal_usage_fp'], 1)
        self.assertEqual(alarms.alarm_frame.tolist(), [311])

    def test_explicit_crops_positive_labels_and_truncated_traces_rejected(self):
        path = self.write_source()
        with self.assertRaisesRegex(ValueError, 'not crops'):
            load_normal_usage(self.source_dir, [f'{path.stem}__crop_f000000_f000300.csv'])
        recordings, _, _ = load_normal_usage(self.source_dir)
        with self.assertRaisesRegex(ValueError, 'full-source'):
            evaluate_normal_usage(recordings, {path.name: np.zeros((300, 3))}, .9, .9)
        path.with_suffix('.txt').write_text('10, 1\n')
        with self.assertRaisesRegex(ValueError, 'negative-only'):
            load_normal_usage(self.source_dir)

    def test_no_population_has_undefined_rate_not_zero(self):
        recs, manifest, _ = load_normal_usage(self.source_dir)
        result, _, files, _ = evaluate_normal_usage(recs, {}, .9, .9)
        self.assertEqual(result['normal_usage_minutes'], 0.)
        self.assertTrue(np.isnan(result['normal_usage_fp_per_min']))
        self.assertTrue(files.empty and manifest.empty)

    def test_normal_crop_duration_provenance_counts_source_once(self):
        source = self.write_source()
        inputs = []
        for start in (0, 600):
            crop = self.root/f'{source.stem}__crop_f{start:06d}_f{start+300:06d}.csv'
            crop.write_text('')
            crop.with_suffix('.mining.json').write_text(json.dumps(dict(
                source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                label_class=0, selection=dict(file=source.name, crop_start_frame=start, crop_end_frame=start+300))))
            inputs.append(dict(path=crop, evaluated_minutes=.05, n_true_events=0))
        audit = recording_durations(self.root, inputs, self.source_dir)
        self.assertEqual(audit.minutes.sum(), 1.)
        self.assertEqual(audit.source_members.tolist(), [2, 2])

    def test_comparison_replays_each_model_over_full_original(self):
        from tap_recognition.config import LSTMModelConfig
        from tap_recognition.model_factory import build_model
        torch.set_num_threads(1)
        self.write_source(frames=1000)
        config = LSTMModelConfig(cnn_channels=4, lstm_hidden=8, kernel_size=3, dilations=(1,))
        model = build_model(config.to_model_kwargs())
        with torch.no_grad():
            model.head[-1].weight.zero_(); model.head[-1].bias.copy_(torch.tensor([0., 8., 0.]))
        checkpoint = self.root/'best.pt'
        torch.save(dict(model_config=config.to_model_kwargs(), model_state=model.state_dict(),
                        train_config=dict(data=dict(highpass=False, window_samples=300))), checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        choices = pd.DataFrame([dict(model=name, checkpoint_path='best.pt', checkpoint_sha256=digest,
                                     selection_mode='max_f1', policy='instant', left_threshold=.9,
                                     right_threshold=.9, refractory_sec=.5) for name in ('a', 'b')])
        scores, files, alarms, manifest, _ = replay_normal_usage(self.root, choices, self.source_dir, chunk_frames=117)
        self.assertEqual(scores.normal_usage_fp.tolist(), [20, 20])
        self.assertEqual(scores.normal_usage_recordings.tolist(), [1, 1])
        np.testing.assert_allclose(scores.normal_usage_minutes, 1/6)
        np.testing.assert_allclose(scores.normal_usage_fp_per_min, 120.)
        self.assertEqual(len(alarms), 40)
        self.assertEqual(files.frames.tolist(), [1000, 1000])

    def test_updated_notebook_sources_compile(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('mine_hard_negatives', 'evaluate', 'compare'):
            notebook = json.loads((root/f'main_notebooks/{name}.ipynb').read_text())
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'{name}:cell{i}', 'exec')

    def test_evaluator_pairs_keep_full_normal_population_separate_from_crop_counts(self):
        path = self.write_source()
        normal, _, _ = load_normal_usage(self.source_dir)
        normal_probs = np.tile([1., 0., 0.], (6000, 1)); normal_probs[5000] = [0., 1., 0.]
        positive = SimpleNamespace(path=Path('positive.csv'), fs=100., imu=np.zeros((6000, 6)),
                                   events=[(100, 1), (200, 2)])
        notebook = json.loads((Path(__file__).resolve().parents[1]/'main_notebooks/evaluate.ipynb').read_text())
        code = next(''.join(c['source']) for c in notebook['cells']
                    if c['cell_type'] == 'code' and 'def evaluate_pair_events(' in ''.join(c['source']))
        ns = dict(np=np, POLICIES={'instant': 0}, DEADLINES=(5, 10), TIMELY_BUDGET_FRAMES=10,
                  recordings=[positive], probabilities={'positive.csv': [dict(alarm_frame=100, class_id=1),
                                                                        dict(alarm_frame=200, class_id=2)]},
                  duration_lookup=pd.DataFrame([dict(file='positive.csv', minutes=1.)]).set_index('file'),
                  duration_audit=pd.DataFrame([dict(evaluated_minutes=1.)]),
                  emit_alarms_pair=lambda p, *_: p,
                  match_alarms=lambda *_args, **_kwargs: [(0, 0), (1, 1)],
                  divide=lambda a, b: a/b if b else np.nan,
                  evaluate_normal_usage=evaluate_normal_usage,
                  normal_usage_recordings=normal, normal_usage_probabilities={path.name: normal_probs})
        exec(code.split('sweep_rows = []')[0], ns)
        overall, sides = ns['evaluate_pair_events'](.9, .9, 'instant')
        self.assertEqual(overall['tp'], 2)
        self.assertEqual(overall['fp'], 0)
        self.assertEqual(overall['minutes'], 1.)
        self.assertEqual(overall['normal_usage_minutes'], 1.)
        self.assertEqual(overall['normal_usage_fp'], 1)
        self.assertEqual(overall['normal_usage_fp_per_min'], 1.)
        self.assertEqual(sides[1]['normal_usage_fp_per_min'], 1.)
        self.assertEqual(sides[2]['normal_usage_fp_per_min'], 0.)


if __name__ == '__main__':
    unittest.main()
