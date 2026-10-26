"""Segment-local confidence mining, positive labels, and safe raw crop exports."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from notebooks_analysis import hard_positive_mining as mining
from notebooks_analysis.hard_negative_mining import MiningModel, load_recording
from notebooks_analysis.recording_durations import durations_for_recordings
from notebooks_analysis.recording_types import recording_type_from_filename
from tap_recognition.dataset import RecordedIMUDataset, load_trigger_events
from tap_recognition.recording import IMU_COLUMNS


class HardPositiveMiningTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'sources'
        self.source.mkdir()
        self.path = self.source / 'imuStrict_fixture_knock_twice_left_20261022_175714.csv'
        frame = np.arange(900)
        self.data = pd.DataFrame(dict(
            timestamp_ms=123000 + frame * 10, user_name='fixture', phase='recording',
            action_label='knock_twice_left', segment_index=np.repeat([1, 2, 3], 300),
            **{col: np.sin(frame / (index + 3)) for index, col in enumerate(IMU_COLUMNS)},
        ))
        self.data.to_csv(self.path, index=False)
        self.path.with_suffix('.txt').write_text('250, 1\n550, 1\n850, 1\n')
        self.rec = load_recording(self.path)
        self.models = {name: MiningModel(self.root / (name + '.pt'), torch.nn.Identity(), False, 300, name)
                       for name in ('a', 'b')}
        self.thresholds = {'a': (.8, .9), 'b': (.7, .9)}
        self.output = self.root / 'crops'

    def traces(self):
        a = np.zeros((900, 3)); a[:, 0] = 1
        b = a.copy()
        for frame, pa, pb in ((250, .9, .8), (550, .6, .8), (850, .9, .6)):
            a[frame] = [1 - pa, pa, 0]
            b[frame] = [1 - pb, pb, 0]
        return {'a': a, 'b': b}

    def score(self, traces=None):
        return mining.score_segments(self.rec, traces or self.traces(), self.thresholds)

    def test_any_failing_model_selects_segment_once(self):
        segments, scores = self.score()
        self.assertEqual(segments.is_hard.tolist(), [False, True, True])
        self.assertEqual(segments.trigger_models.tolist(), [[], ['a'], ['b']])
        self.assertEqual(segments[segments.is_hard].candidate_id.tolist(), [1, 2])
        self.assertEqual(len(scores), 6)
        self.assertTrue(segments.training_retains_event.all())

    def test_score_window_is_inclusive_clipped_and_segment_local(self):
        traces = self.traces()
        traces['a'][550] = [1, 0, 0]
        traces['a'][500] = [.2, .8, 0]  # Exactly -500 ms; equality is easy.
        traces['a'][499] = [0, 1, 0]  # Outside the scoring window.
        traces['a'][600] = [0, 1, 0]  # Neighbor segment cannot rescue segment 2.
        segments, scores = self.score(traces)
        row = segments.iloc[1]
        self.assertEqual((row.score_start_frame, row.score_end_frame), (500, 600))
        self.assertTrue(row.window_clipped)
        self.assertFalse(row.is_hard)
        self.assertEqual(scores[(scores.candidate_id == 1) & (scores.model == 'a')].peak_frame.iloc[0], 500)
        traces['a'][500] = [1, 0, 0]
        segments, _ = self.score(traces)
        self.assertTrue(segments.iloc[1].is_hard)

    def test_right_side_and_positive_window_endpoint(self):
        path = self.source / self.path.name.replace('_left_', '_right_')
        self.data.assign(action_label='knock_twice_right').to_csv(path, index=False)
        path.with_suffix('.txt').write_text('150, 2\n450, 2\n750, 2\n')
        rec = load_recording(path)
        trace = np.zeros((900, 3)); trace[:, 0] = 1
        trace[200] = [.1, 0, .9]  # Exactly +500 ms.
        trace[450] = [0, 1, 0]  # High wrong-side peak does not rescue right.
        segments, scores = mining.score_segments(rec, {'a': trace}, {'a': (.8, .9)})
        self.assertEqual(segments.is_hard.tolist(), [False, True, True])
        self.assertEqual(scores.peak_frame.iloc[0], 200)

    def test_unlabeled_and_ambiguous_segments_never_become_training_negatives(self):
        self.path.with_suffix('.txt').write_text('250, 1\n820, 1\n850, 1\n')
        self.rec = load_recording(self.path)
        segments, scores = self.score()
        self.assertEqual(segments.status.tolist(), ['labeled', 'unlabeled', 'ambiguous'])
        self.assertFalse(segments.iloc[1:].is_hard.any())
        self.assertEqual(len(scores), 2)
        with self.assertRaisesRegex(ValueError, 'hard segments'):
            mining.export_batch(segments.iloc[1:2], self.source, self.output)

    def test_export_positive_labels_raw_values_idempotency_and_loader(self):
        segments, _ = self.score()
        hard = segments[segments.is_hard]
        exports = mining.export_batch(hard, self.source, self.output)
        self.assertEqual(len(exports), 2)
        path = Path(exports.csv.iloc[0])
        crop = pd.read_csv(path)
        np.testing.assert_allclose(crop[IMU_COLUMNS], self.data[IMU_COLUMNS].iloc[300:600], atol=1e-8)
        self.assertEqual(crop.segment_index.unique().tolist(), [1])
        self.assertEqual(crop.timestamp_ms.iloc[0], 0)
        self.assertEqual(set(crop.action_label), {'knock_twice_left'})
        self.assertEqual(load_trigger_events(path.with_suffix('.txt')), [(250, 1)])
        metadata = json.loads(path.with_suffix('.mining.json').read_text())
        self.assertEqual(metadata['mining_kind'], 'hard_positive')
        self.assertEqual(metadata['label_class'], 1)
        self.assertEqual(metadata['selection']['trigger_models'], ['a'])
        self.assertEqual(recording_type_from_filename(path.name), 'knock_twice')
        dataset = RecordedIMUDataset(self.output, highpass=False)
        self.assertEqual(len(dataset), 2)
        self.assertEqual([int(item['window_label']) for item in dataset], [1, 1])
        self.assertEqual(dataset[0]['frame_labels'][250].argmax().item(), 1)
        durations = durations_for_recordings(self.root, [load_recording(path)], self.source)
        self.assertFalse(durations.is_mining.iloc[0])
        self.assertEqual(durations.minutes.iloc[0], 3 / 60)
        self.assertTrue(mining.export_batch(hard, self.source, self.output).status.eq('already exported').all())

    def test_export_rejects_easy_conflicting_stale_or_duplicate_selections(self):
        segments, _ = self.score()
        with self.assertRaisesRegex(ValueError, 'hard segments'):
            mining.export_batch(segments.iloc[:1], self.source, self.output)
        hard = segments[segments.is_hard]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            mining.export_batch(pd.concat([hard, hard]), self.source, self.output)
        self.assertFalse(self.output.exists())
        mining.export_batch(hard, self.source, self.output)
        csv = next(self.output.glob('*.csv'))
        csv.with_suffix('.txt').write_text('0, 1\n')
        with self.assertRaises(FileExistsError):
            mining.export_batch(hard, self.source, self.output)
        self.path.with_suffix('.txt').write_text('249, 1\n550, 1\n850, 1\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            mining.export_batch(hard, self.source, self.root / 'stale')

    def test_segment_reset_inference_calls_each_raw_segment_separately(self):
        with patch.object(mining, 'predict_recording', return_value=np.zeros((300, 3))) as predict:
            p = mining.predict_positive_recording(self.rec, self.models['a'])
        self.assertEqual(p.shape, (900, 3))
        self.assertEqual(predict.call_count, 3)
        segment = predict.call_args_list[1].args[0]
        self.assertEqual(segment.events, [(250, 1)])
        np.testing.assert_allclose(segment.raw, self.rec.raw[300:600])
        with patch.object(mining, 'predict_recording', return_value=np.zeros((900, 3))) as predict:
            mining.predict_positive_recording(self.rec, self.models['a'], False)
        predict.assert_called_once()

    def test_batch_ids_deduplication_all_easy_stats_and_empty_exports(self):
        duplicate = self.source / self.path.name.replace('175714', '175715')
        duplicate.write_bytes(self.path.read_bytes())
        duplicate.with_suffix('.txt').write_bytes(self.path.with_suffix('.txt').read_bytes())
        with patch.object(mining, 'predict_positive_recording', side_effect=lambda rec, model, *a: self.traces()[model.sha256]):
            hard, all_segments, scores, stats = mining.mine_recordings(
                [self.path, duplicate], self.models, self.thresholds)
        self.assertEqual(len(stats), 1)
        self.assertEqual(len(all_segments), 3)
        self.assertEqual(len(hard), 2)
        self.assertEqual(len(scores), 6)
        self.assertEqual(stats.easy_segments.iloc[0], 1)
        duplicate.with_suffix('.txt').write_text('249, 1\n550, 1\n850, 1\n')
        with patch.object(mining, 'predict_positive_recording', side_effect=lambda rec, model, *a: self.traces()[model.sha256]):
            with self.assertRaisesRegex(ValueError, 'conflicting labels'):
                mining.mine_recordings([self.path, duplicate], self.models, self.thresholds)
        self.assertTrue(mining.export_batch(hard.iloc[:0], self.source, self.output).empty)
        empty, _, _, _ = mining.mine_recordings([], self.models, self.thresholds)
        self.assertTrue(empty.empty)
        easy = np.zeros((900, 3)); easy[:, 1] = 1
        with patch.object(mining, 'predict_positive_recording', return_value=easy):
            hard, _, _, stats = mining.mine_recordings([self.path], self.models, self.thresholds)
        self.assertTrue(hard.empty)
        self.assertEqual(stats.easy_segments.iloc[0], 3)

    def test_detailed_plot_shows_each_model(self):
        segments, _ = self.score()
        fig, axes = mining.plot_segment(self.rec, self.traces(), self.thresholds, segments.iloc[1])
        self.assertEqual(len(axes), 4)
        plt.close(fig)

    def test_inventory_accepts_positive_crops_without_negative_weighting(self):
        segments, _ = self.score()
        mining.export_batch(segments[segments.is_hard], self.source, self.output)
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks_analysis/04_recording_inventory_split_balance.ipynb').read_text())
        parser = next(''.join(c['source']) for c in notebook['cells']
                      if 'FILENAME_PATTERN = ' in ''.join(c.get('source', [])))
        import re
        env = dict(re=re, Path=Path, pd=pd, np=np, json=json, DATA_DIR=self.output, display=lambda *a: None)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(parser, 'inventory-parser', 'exec'), env)
        self.assertEqual(len(env['records']), 2)
        self.assertEqual(env['unparsed'], [])
        self.assertTrue(env['records'].gesture.eq('knock_twice_left').all())
        share = next(''.join(c['source']) for c in notebook['cells']
                     if 'HARD_NEGATIVE_CROPS_PER_RECORDING = ' in ''.join(c.get('source', [])))
        exec(compile(share.split('overview = ')[0], 'inventory-share', 'exec'), env)
        self.assertTrue(env['records'].gesture_share_weight.eq(1).all())

    def test_new_notebook_compiles_and_keeps_exports_off(self):
        root = Path(__file__).resolve().parents[1]
        n = json.loads((root / 'main_notebooks/mine_hard_positives.ipynb').read_text())
        for index, cell in enumerate(n['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'positive-cell-{index}', 'exec')
        text = '\n'.join(''.join(c['source']) for c in n['cells'])
        self.assertIn('SCORE_BEFORE_MS = 500.0', text)
        self.assertIn('SCORE_AFTER_MS = 500.0', text)
        self.assertIn('SAVE_ALL_HARD_POSITIVES = False', text)
        self.assertIn('SAVE_SELECTED_HARD_POSITIVES = False', text)


if __name__ == '__main__':
    unittest.main()
