"""Source duration accounting: shared crops, portable provenance, and notebook wiring."""

import hashlib
import ast
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from notebooks_analysis.recording_durations import (
    DURATION_BASIS, apply_export_durations, duration_fingerprint,
    durations_for_export, durations_for_recordings, recording_durations,
)
from notebooks_analysis.normal_usage_evaluation import evaluate_normal_usage


class RecordingDurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / 'hard_negative_mining/source_recordings'
        self.sources.mkdir(parents=True)
        self.dataset = self.root / 'data_example/valid_data'
        self.dataset.mkdir(parents=True)
        self.inputs = []

    def source(self, tag='one', frames=6000, step=10):
        path = self.sources / f'imu_Miguel_negative_mining_{tag}.csv'
        pd.DataFrame({'timestamp_ms': np.arange(frames) * step}).to_csv(path, index=False)
        return path

    def crop(self, source, start=0, end=300, split='valid_data'):
        folder = self.root / 'data_example' / split
        folder.mkdir(exist_ok=True)
        path = folder / f'{source.stem}__crop_f{start:06d}_f{end:06d}.csv'
        path.write_text('timestamp_ms\n0\n10\n')
        meta = dict(source_path=f'/old-machine/project/source_recordings/{source.name}',
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), label_class=0,
                    selection=dict(file=source.name, crop_start_frame=start, crop_end_frame=end))
        path.with_suffix('.mining.json').write_text(json.dumps(meta))
        return dict(path=path, evaluated_minutes=(end-start)/100/60, n_true_events=0)

    def test_shared_sources_count_once_with_actual_lengths(self):
        one = self.source(frames=6017)
        two = self.source(tag='two', frames=20217)
        inputs = [self.crop(one), self.crop(one, 1000, 1300), self.crop(two)]
        inputs.append(dict(path=self.dataset/'ordinary.csv', evaluated_minutes=2., n_true_events=1))
        table = recording_durations(self.root, inputs)
        self.assertAlmostEqual(table.minutes.sum(), 2 + 6017/6000 + 20217/6000)
        self.assertAlmostEqual(table.loc[0, 'minutes'], 6017/6000/2)
        self.assertEqual(table.source_members.tolist(), [2, 2, 1, 1])
        self.assertEqual(table.loc[0, 'source_frames'], 6017)
        self.assertEqual(table.loc[0, 'source_sample_rate_hz'], 100)
        self.assertAlmostEqual(table.evaluated_minutes.sum(), 2.15)
        self.assertTrue(table.duration_basis.eq(DURATION_BASIS).all())

    def test_dedup_full_source_and_crops_and_handle_non_100hz(self):
        source = self.source(frames=6000, step=20)
        inputs = [self.crop(source), dict(path=source, evaluated_minutes=2., n_true_events=0)]
        table = recording_durations(self.root, inputs)
        self.assertAlmostEqual(table.minutes.sum(), 2.)
        self.assertEqual(table.source_members.tolist(), [2, 2])

    def test_missing_or_changed_original_is_not_silently_approximated(self):
        source = self.source()
        item = self.crop(source)
        source.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'hash-matching original'):
            recording_durations(self.root, [item])
        source.write_text('timestamp_ms\n0\n10\n')
        with self.assertRaisesRegex(FileNotFoundError, 'hash-matching original'):
            recording_durations(self.root, [item])
        item['path'].with_suffix('.mining.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'required to recover'):
            recording_durations(self.root, [item])

    def test_bad_bounds_positive_mining_and_duplicate_inputs_rejected(self):
        source = self.source(frames=400)
        with self.assertRaisesRegex(ValueError, 'crop bounds'):
            recording_durations(self.root, [self.crop(source, 200, 500)])
        item = self.crop(source)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            recording_durations(self.root, [item, item])
        with self.assertRaisesRegex(ValueError, 'no annotated'):
            recording_durations(self.root, [item | {'n_true_events': 1}])

    def test_split_and_model_accounting_is_independent(self):
        source = self.source()
        train = [self.crop(source, split='train_data'), self.crop(source, 500, 800, split='train_data')]
        valid = [self.crop(source, 1000, 1300)]
        for inputs in (train, valid):
            self.assertAlmostEqual(recording_durations(self.root, inputs).minutes.sum(), 1.)
        # Two models evaluating the same inputs each get a one-minute denominator.
        self.assertEqual(duration_fingerprint(recording_durations(self.root, valid)),
                         duration_fingerprint(recording_durations(self.root, valid)))

    def test_export_corrections_are_idempotent_and_leave_counts_unchanged(self):
        source = self.source()
        inputs = [self.crop(source), self.crop(source, 500, 800)]
        by_file = pd.DataFrame([dict(file=x['path'].name, minutes=.05, n_true_events=0,
                                     recording_type='hard_negative', event_fp=1,
                                     event_left_fp=1, event_right_fp=0) for x in inputs])
        by_type = pd.DataFrame([dict(recording_type='hard_negative', minutes=.1, event_fp=2)])
        summary = pd.DataFrame([dict(level='event', side=side, fp=fp, minutes=.1,
                                    fp_per_min=fp/.1, negative_fp_per_min=fp/.1)
                                for side, fp in [('all', 2), ('left', 2), ('right', 0)]])
        # Simulate a remote export's absolute dataset paths.
        data = dict(val_dir='/old-machine/project/data_example/valid_data')
        audit = durations_for_export(self.root, by_file, data)
        corrected = apply_export_durations(summary, by_type, by_file, audit)
        self.assertAlmostEqual(corrected[0].iloc[0].fp_per_min, 2.)
        self.assertAlmostEqual(corrected[0].iloc[0].negative_fp_per_min, 2.)
        self.assertAlmostEqual(corrected[1].minutes.sum(), 1.)
        self.assertAlmostEqual(corrected[2].minutes.sum(), 1.)
        self.assertAlmostEqual(corrected[2].evaluated_minutes.sum(), .1)
        self.assertEqual(corrected[2].event_fp.tolist(), by_file.event_fp.tolist())
        again_audit = durations_for_export(self.root, corrected[2], data)
        again = apply_export_durations(*corrected, again_audit)
        for first, second in zip(corrected, again):
            pd.testing.assert_frame_equal(first, second)
        reloaded = pd.read_csv(io.StringIO(audit.to_csv(index=False)), keep_default_na=False)
        self.assertEqual(duration_fingerprint(audit), duration_fingerprint(reloaded))

    def test_replay_adapter_uses_only_included_segments_for_ordinary_inputs(self):
        rec = SimpleNamespace(path=self.dataset/'ordinary.csv', fs=100., events=[],
                              imu=np.zeros((6000, 6)), segments=[(1, 0, 300), (3, 600, 900)])
        audit = durations_for_recordings(self.root, [rec])
        self.assertAlmostEqual(audit.minutes.sum(), .1)

    def test_real_replay_counts_source_once_per_model_and_split(self):
        import torch
        from notebooks_analysis.model_comparison_examples import ExampleRecording, replay_examples
        from tap_recognition.config import LSTMModelConfig
        from tap_recognition.model_factory import build_model

        torch.set_num_threads(1)
        source = self.source()
        inputs = [self.crop(source), self.crop(source, 500, 800)]
        recordings = {}
        for item in inputs:
            rec = ExampleRecording(path=item['path'], action='negative_mining', user='test', fs=100.,
                                   time=np.arange(300)/100, raw=np.zeros((300, 6), dtype=np.float32),
                                   imu=np.zeros((300, 6), dtype=np.float32), events=[],
                                   segments=[(0, 0, 150), (1, 150, 300)], windows={})
            recordings[rec.path.name] = rec
        config = LSTMModelConfig(cnn_channels=4, lstm_hidden=8, kernel_size=3,
                                dilations=(1,), dropout=0.)
        model = build_model(config.to_model_kwargs())
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.copy_(torch.tensor([0., 8., 0.]))
        checkpoint = self.root/'best.pt'
        torch.save(dict(model_config=config.to_model_kwargs(), model_type='lstm',
                        model_state=model.state_dict(),
                        train_config=dict(data={}, labels={}, seed=1)), checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        points = pd.DataFrame([dict(model=name, checkpoint_path='best.pt',
                                   checkpoint_sha256=digest, validation_sha256='test',
                                   policy='instant', left_threshold=.9, right_threshold=.9,
                                   early_allowance_frames=25, max_late_frames=40,
                                   refractory_sec=.5) for name in ('a', 'b')])
        for split in ('train', 'valid'):
            with patch('notebooks_analysis.model_comparison_examples.load_examples',
                       return_value=(recordings, 'test')):
                _, _, segments, files, _, _ = replay_examples(self.root, points, split=split)
            for name in ('a', 'b'):
                selected = files[files.model == name]
                self.assertAlmostEqual(selected.minutes.sum(), 1.)
                self.assertAlmostEqual(selected.evaluated_minutes.sum(), .1)
                self.assertEqual(selected.source_members.tolist(), [2, 2])
                self.assertEqual(selected.event_fp.tolist(), [6, 6])
                self.assertEqual(selected.event_fp_per_min.tolist(), [12., 12.])
                self.assertAlmostEqual(segments.loc[segments.model == name, 'minutes'].sum(), 1.)

    def test_actual_notebook_sweeps_agree_on_unique_source_denominator(self):
        source = self.source()
        inputs = [self.crop(source), self.crop(source, 500, 800)]
        recordings = [SimpleNamespace(path=x['path'], fs=100., events=[], user='test',
                                      action='negative_mining', imu=np.zeros((300, 6))) for x in inputs]
        audit = durations_for_recordings(self.root, recordings)
        probs = np.zeros((300, 3)); probs[:, 0] = 1.; probs[100] = [0., 1., 0.]
        notebook = json.loads((Path(__file__).resolve().parents[1]/'main_notebooks/evaluate.ipynb').read_text())
        cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        ns = dict(np=np, pd=pd, linear_sum_assignment=linear_sum_assignment,
                  recordings=recordings, probabilities={r.path.name: probs for r in recordings},
                  duration_audit=audit, duration_lookup=audit.set_index('file'),
                  display=lambda _: None, divide=lambda a, b: a/b if b else np.nan)
        ns.update(evaluate_normal_usage=evaluate_normal_usage, normal_usage_recordings=[], normal_usage_probabilities={},
                  validation_crop_normal_usage_recordings=[], validation_crop_normal_usage_probabilities={})
        single = next(s for s in cells if 'def emit_alarms(' in s)
        single = 'THRESHOLDS = [.5]\n' + single.split('\n', 1)[1]
        exec(single, ns)
        sessions = ns['sessions']
        self.assertAlmostEqual(sessions.minutes.sum(), 1.)
        self.assertEqual(sessions.fp.sum(), 2)
        paired_detector = next(s for s in cells if 'def emit_alarms_pair(' in s)
        exec(paired_detector.split('# Detector parity')[0].split('for rec in recordings:')[0], ns)
        ns.update(DEADLINES=(5, 10, 20, 30), TIMELY_BUDGET_FRAMES=10)
        paired = next(s for s in cells if 'def evaluate_pair_events(' in s)
        exec(paired.split('sweep_rows = []')[0], ns)
        overall, sides = ns['evaluate_pair_events'](.5, .5, 'lookahead_12f')
        self.assertAlmostEqual(overall['minutes'], 1.)
        self.assertAlmostEqual(overall['negative_minutes'], 1.)
        self.assertAlmostEqual(overall['evaluated_minutes'], .1)
        self.assertEqual(overall['fp'], 2)
        self.assertEqual(overall['negative_fp_per_min'], 2.)
        self.assertEqual(sides[1]['negative_fp_per_min'], 2.)

    def test_both_notebooks_compile_and_export_duration_identity(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('evaluate', 'compare'):
            notebook = json.loads((root/f'main_notebooks/{name}.ipynb').read_text())
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'{name}:cell{i}', 'exec')
        code = (root/'main_notebooks/evaluate.ipynb').read_text()
        self.assertIn('recording_duration_audit.csv', code)
        self.assertIn("'duration_basis', 'duration_sha256'", code)

    def test_compare_notebook_uses_an_exported_selection_mode(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'main_notebooks/compare.ipynb').read_text())
        setup = ''.join(notebook['cells'][1]['source'])
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse(setup).body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {'SELECTION_MODE', 'POLICY'}}
        self.assertIn(constants['SELECTION_MODE'], {'max_f1', 'fp_budget_max_recall', 'recall_target_min_fp',
                                                   'recall_target_max_threshold', 'timely_recall_fp_budget'})
        self.assertIn(constants['POLICY'], {'instant', 'lookahead_12f'})
        self.assertIn("SELECTION_MODE in {'fp_per_min', 'negative_fp_per_min', 'normal_usage_fp_per_min'}", setup)


if __name__ == '__main__':
    unittest.main()
