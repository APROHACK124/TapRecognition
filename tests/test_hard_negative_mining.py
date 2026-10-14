"""Hard-negative crop boundaries, raw preservation, and training-loader checks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from notebooks_analysis.hard_negative_mining import (
    MiningModel,
    crop_bounds,
    export_candidates,
    export_batch,
    find_candidates,
    find_union_candidates,
    load_recording,
    stage_recordings,
    mine_recordings,
    predict_recording,
)
from notebooks_analysis.recording_types import recording_type_from_filename
from tap_recognition.training_sampling import repeat_hard_negatives
from tap_recognition.dataset import RecordedIMUDataset
from tap_recognition.recording import IMU_COLUMNS


class HardNegativeMiningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.path = self.source / 'imu_fixture_negative_mining_20261009_164938.csv'
        frames = np.arange(1000)
        self.data = pd.DataFrame(dict(
            timestamp_ms=123000 + frames * 10, user_name='fixture', phase='recording',
            action_label='negative_mining', segment_index=1,
            **{col: np.sin(frames / (index + 3)) for index, col in enumerate(IMU_COLUMNS)},
        ))
        self.data.to_csv(self.path, index=False)
        self.path.with_suffix('.txt').write_text('')
        self.rec = load_recording(self.path)
        self.model = MiningModel(self.root / 'best.pt', torch.nn.Identity(), False, 300, 'fixture-model')
        self.output = self.root / 'crops'

    def candidates(self, frame=500, **settings):
        probs = np.zeros((1000, 3))
        probs[:, 0] = 1
        probs[frame] = [.05, .9, .05]
        return find_candidates(self.rec, probs, 'fixture', .8, .8,
                               lookahead_frames=0, **settings)

    def test_core_is_inclusive_and_context_is_three_seconds(self):
        core = crop_bounds(self.rec, 500, 40, 10)
        self.assertEqual((core['crop_start_frame'], core['crop_end_frame']), (460, 511))
        self.assertEqual(core['crop_frames'], 51)
        context = crop_bounds(self.rec, 500, 40, 10, 200, 99)
        self.assertEqual((context['crop_start_frame'], context['crop_end_frame']), (300, 600))
        self.assertEqual(context['anchor_in_crop'], 200)
        self.assertEqual(context['crop_seconds'], 3)

    def test_context_slides_near_boundary_without_crossing_sessions(self):
        self.data['segment_index'] = np.repeat([1, 2], 500)
        self.data.to_csv(self.path, index=False)
        rec = load_recording(self.path)
        for anchor, expected in ((498, (200, 500)), (501, (500, 800))):
            with self.subTest(anchor=anchor):
                crop = crop_bounds(rec, anchor, 40, 10, 200, 99)
                self.assertEqual((crop['crop_start_frame'], crop['crop_end_frame']), expected)
                self.assertTrue(crop['crop_start_frame'] <= crop['core_start_frame'] <
                                crop['core_end_frame'] <= crop['crop_end_frame'])

    def test_peak_crossing_and_emission_are_distinct_and_side_eligible(self):
        probs = np.zeros((1000, 3))
        probs[:, 0] = 1
        probs[100] = [.1, .6, .3]
        probs[103] = [.05, .75, .2]
        probs[105] = [.15, .05, .8]  # Right is below its .9 threshold.
        probs[200] = [.03, .02, .95]
        for anchor, expected in (('crossing', 100), ('peak', 103), ('alarm', 112)):
            table = find_candidates(self.rec, probs, 'fixture', .5, .9, anchor=anchor)
            self.assertEqual(table.class_id.tolist(), [1, 2])
            self.assertEqual(table.anchor_frame.iloc[0], expected)
            self.assertEqual(table.peak_frame.iloc[0], 103)

    def test_export_preserves_raw_frames_and_is_one_negative_training_window(self):
        candidates = self.candidates(context_before_frames=200, context_after_frames=99)
        table = export_candidates(self.rec, candidates, self.model, self.output)
        csv_path = Path(table.csv.iloc[0])
        crop = pd.read_csv(csv_path)
        np.testing.assert_allclose(crop[IMU_COLUMNS], self.rec.data[IMU_COLUMNS].iloc[300:600], atol=1e-8)
        self.assertEqual(crop.segment_index.unique().tolist(), [1])
        self.assertEqual(crop.timestamp_ms.iloc[0], 0)
        self.assertEqual(csv_path.with_suffix('.txt').read_bytes(), b'')
        metadata = json.loads(Path(table.provenance.iloc[0]).read_text())
        self.assertEqual(metadata['source_sha256'], self.rec.sha256)
        self.assertEqual(metadata['selection']['anchor_frame'], 500)
        dataset = RecordedIMUDataset(self.output, highpass=False)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(int(sample['window_label']), 0)
        self.assertTrue(torch.all(sample['frame_labels'][:, 0] == 1))
        np.testing.assert_allclose(sample['imu'].numpy(), self.rec.raw[300:600], atol=1e-7)
        second = export_candidates(self.rec, candidates, self.model, self.output)
        self.assertEqual(second.status.iloc[0], 'already exported')

    def test_short_crop_is_padded_by_existing_loader(self):
        export_candidates(self.rec, self.candidates(), self.model, self.output)
        dataset = RecordedIMUDataset(self.output, highpass=False)
        self.assertEqual(dataset[0]['imu'].shape, (300, 6))
        np.testing.assert_allclose(dataset[0]['imu'].numpy()[:51], self.rec.raw[460:511], atol=1e-7)
        np.testing.assert_allclose(dataset[0]['imu'].numpy()[51:],
                                   np.repeat(self.rec.raw[510:511], 249, axis=0), atol=1e-7)

    def test_rejects_context_that_training_would_truncate(self):
        candidates = self.candidates(context_before_frames=400, context_after_frames=99)
        self.assertFalse(candidates.training_retains_core.iloc[0])
        with self.assertRaisesRegex(ValueError, 'Training truncates'):
            export_candidates(self.rec, candidates, self.model, self.output)
        self.assertEqual(list(self.output.glob('*.csv')), [])

    def test_rejects_positive_events_and_conflicting_outputs(self):
        candidates = self.candidates()
        export_candidates(self.rec, candidates, self.model, self.output)
        csv_path = next(self.output.glob('*.csv'))
        csv_path.with_suffix('.txt').write_text('1, 1\n')
        with self.assertRaises(FileExistsError):
            export_candidates(self.rec, candidates, self.model, self.output)
        self.rec.events = [(500, 1)]
        with self.assertRaisesRegex(ValueError, 'annotated positive'):
            export_candidates(self.rec, candidates, self.model, self.root / 'positive-crop')

    def test_staging_keeps_sources_and_refuses_different_copies(self):
        dest = self.root / 'inputs'
        paths = stage_recordings(self.source, dest)
        self.assertTrue(self.path.exists())
        self.assertEqual(paths, stage_recordings(self.source, dest))
        self.assertEqual(paths[0].with_suffix('.txt').read_bytes(), b'')
        paths[0].write_text('different recording')
        with self.assertRaises(FileExistsError):
            stage_recordings(self.source, dest)

    def test_empty_detection_table_supports_review_and_export(self):
        probs = np.zeros((1000, 3))
        probs[:, 0] = 1
        candidates = find_candidates(self.rec, probs, 'fixture', .5, .5)
        self.assertTrue(candidates.empty)
        self.assertIn('candidate_id', candidates.columns)
        self.assertTrue(export_candidates(self.rec, candidates, self.model, self.output).empty)

    def trace(self, frames):
        p = np.zeros((1000, 3)); p[:, 0] = 1
        for frame in frames:
            p[frame] = [.01, .98, .01]
        return p

    def test_union_uses_any_model_thresholds_and_shared_crop_gap(self):
        candidates = find_union_candidates(
            self.rec, {'a': self.trace([100, 500]), 'b': self.trace([110, 850])},
            {'a': (.9, .9), 'b': (.95, .95)}, min_gap_frames=300,
            lookahead_frames=0, context_before_frames=200, context_after_frames=99)
        self.assertEqual(candidates.anchor_frame.tolist(), [100, 500, 850])
        self.assertEqual(candidates.trigger_models.iloc[0], ['a', 'b'])
        self.assertEqual(candidates.crop_frames.tolist(), [300, 300, 300])
        self.assertEqual(candidates.candidate_id.tolist(), [0, 1, 2])
        self.assertTrue(all(a <= b for a, b in zip(candidates.crop_end_frame[:-1], candidates.crop_start_frame[1:])))
        # Raising only b's threshold removes its unique crop, not a's failures.
        fewer = find_union_candidates(
            self.rec, {'a': self.trace([100, 500]), 'b': self.trace([110, 850])},
            {'a': (.9, .9), 'b': (1., 1.)}, min_gap_frames=300,
            lookahead_frames=0, context_before_frames=200, context_after_frames=99)
        self.assertEqual(fewer.anchor_frame.tolist(), [100, 500])

    def test_gap_is_inclusive_and_boundary_duplicates_are_suppressed(self):
        candidates = find_union_candidates(
            self.rec, {'a': self.trace([300, 600, 601])}, {'a': (.9, .9)},
            min_gap_frames=300, lookahead_frames=0, refractory_frames=0,
            context_before_frames=200, context_after_frames=99)
        self.assertEqual(candidates.anchor_frame.tolist(), [300, 601])
        same = find_union_candidates(
            self.rec, {'a': self.trace([0, 99])}, {'a': (.9, .9)}, min_gap_frames=0,
            lookahead_frames=0, allow_overlapping_crops=True,
            context_before_frames=200, context_after_frames=99)
        self.assertEqual(len(same), 1)

    def test_batch_stats_include_zero_crop_sources_and_export_normal_usage(self):
        normal = self.source / 'imu_fixture_normal_usage_20261014_120000.csv'
        self.data.assign(action_label='normal_usage', timestamp_ms=self.data.timestamp_ms+10).to_csv(normal, index=False)
        normal.with_suffix('.txt').write_text('')
        traces = [self.trace([]), self.trace([500])]
        with patch('notebooks_analysis.hard_negative_mining.predict_recording', side_effect=traces):
            candidates, by_file, summary = mine_recordings(
                [self.path, normal], {'fixture': self.model}, {'fixture': (.9, .9)},
                crop_seconds=3., context_before_seconds=2., lookahead_frames=0,
                training_window_frames=300)
        self.assertEqual(by_file.crops.tolist(), [0, 1])
        self.assertEqual(summary['recordings'], 2)
        self.assertEqual(summary['recording_seconds'], 20.)
        self.assertEqual(summary['crop_seconds'], 3.)
        self.assertEqual(summary['crops_per_minute'], 3.)
        exported = export_batch(candidates, {'fixture': self.model}, self.source, self.output)
        path = Path(exported.csv.iloc[0])
        self.assertIn('normal_usage', path.name)
        self.assertEqual(recording_type_from_filename(path.name), 'hard_negative')
        meta = json.loads(path.with_suffix('.mining.json').read_text())
        self.assertEqual(meta['source_frames'], 1000)
        self.assertEqual(meta['source_duration_seconds'], 10.)
        self.assertEqual(meta['source_recording_type'], 'normal_usage')
        self.assertEqual(meta['selection']['trigger_models'], ['fixture'])
        self.assertEqual(meta['selection']['mining_models'][0]['checkpoint_sha256'], self.model.sha256)
        dataset = RecordedIMUDataset(self.output, highpass=False)
        repeated, report = repeat_hard_negatives(dataset, 5)
        self.assertEqual(len(repeated), 5)
        self.assertTrue(report.hard_negative.all())
        # Full normal-use training files are not filename-repeated as mined crops.
        plain = RecordedIMUDataset(self.source, highpass=False, csv_paths=[normal])
        _, report = repeat_hard_negatives(plain, 5)
        self.assertFalse(report.hard_negative.any())

    def test_chunked_full_inference_preserves_raw_and_feature_state(self):
        from tap_recognition.config import LSTMModelConfig
        from tap_recognition.imu_features import IMUFeatureConfig
        from tap_recognition.model_factory import build_model
        from tap_recognition.physics import IMUSimulator
        torch.set_num_threads(1)
        config = LSTMModelConfig(cnn_channels=4, lstm_hidden=8, kernel_size=3, dilations=(1,))
        for features in ({}, {'feature_config': IMUFeatureConfig().to_dict()}):
            model = build_model(config.to_model_kwargs() | features).eval()
            wrapper = MiningModel(self.root/'fixture.pt', model, True, 300, 'test')
            raw = IMUSimulator(sample_rate=self.rec.fs).highpass(self.rec.raw)
            with torch.inference_mode():
                logits, _ = model(torch.from_numpy(raw).unsqueeze(0))
                expected = logits[0].softmax(-1).numpy()
            actual = predict_recording(self.rec, wrapper, chunk_frames=137)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_batch_rejects_positive_recordings(self):
        self.rec.events = [(500, 1)]
        with self.assertRaisesRegex(ValueError, 'negative-only'):
            find_union_candidates(self.rec, {'a': self.trace([500])}, {'a': (.9, .9)})


if __name__ == '__main__':
    unittest.main()
