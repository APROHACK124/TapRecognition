"""Standard switch_doubletap negatives in evaluator, replay, and inventory."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from notebooks_analysis.model_comparison_examples import load_examples, replay_examples
from notebooks_analysis.state_reset_analysis import load_recordings
from tap_recognition.config import LSTMModelConfig, TrainConfig
from tap_recognition.dataset import RecordedIMUDataset
from tap_recognition.model_factory import build_model
from tap_recognition.recording import IMU_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


class SwitchDoubletapTests(unittest.TestCase):
    def test_standard_negative_sections_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = TrainConfig()
            cfg.data.train_dir = 'train_data'
            cfg.data.val_dir = 'valid_data'
            cfg.data.session_exclusions_file = ''
            cfg.data.highpass = False
            for split, stamp in (('train_data', '170000'), ('valid_data', '170100')):
                directory = root / split
                directory.mkdir()
                path = directory / f'imu_fixture_switch_doubletap_20261012_{stamp}.csv'
                frame = pd.DataFrame(dict(
                    timestamp_ms=np.arange(6000) * 10,
                    user_name='fixture', phase='recording', action_label='switch_doubletap',
                    segment_index=np.repeat(np.arange(1, 21), 300),
                    **{column: np.zeros(6000) for column in IMU_COLUMNS},
                ))
                frame.to_csv(path, index=False)
                path.with_suffix('.txt').write_text('')
                dataset = RecordedIMUDataset(directory, highpass=False)
                self.assertEqual(len(dataset), 20)
                self.assertTrue(all(int(item['window_label']) == 0 for item in dataset))
                self.assertTrue(all(torch.all(item['frame_labels'][:, 0] == 1) for item in dataset))

            recordings, audit, _ = load_recordings(root, cfg.to_dict()['data'], 10)
            self.assertEqual(len(recordings), 2)
            self.assertTrue(audit.events.eq(0).all())
            self.assertTrue(audit.included_segments.eq(20).all())
            self.assertTrue(np.allclose(audit.minutes, 1))
            for recording in recordings:
                self.assertEqual(recording.action, 'switch_doubletap')
                self.assertEqual(recording.events, [])

            # Force a positive prediction: every window and emitted alarm must be FP.
            config = LSTMModelConfig(cnn_channels=4, lstm_hidden=8, dilations=(1,))
            model = build_model(config.to_model_kwargs(), 'lstm')
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.head[-1].bias[1] = 10
            checkpoint = root / 'best.pt'
            torch.save(dict(model_type='lstm', model_config=config.to_model_kwargs(),
                            model_state=model.state_dict()), checkpoint)
            _, digest = load_examples(root, cfg.to_dict()['data'], cfg.to_dict()['labels'], cfg.seed)
            points = pd.DataFrame([dict(
                model='fixture', checkpoint_path='best.pt',
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                evaluation_data=cfg.to_dict()['data'], evaluation_labels=cfg.to_dict()['labels'],
                evaluation_seed=cfg.seed, validation_sha256=digest, policy='lookahead_12f',
                left_threshold=.5, right_threshold=.5, refractory_sec=.5,
                early_allowance_frames=20, max_late_frames=30,
            )])
            for split in ('train', 'valid'):
                _, _, segments, files, alarms, _ = replay_examples(root, points, split=split)
                self.assertEqual(len(segments), 20)
                self.assertTrue(segments.recording_type.eq('switch_doubletap').all())
                self.assertTrue(segments.window_true.eq(0).all())
                self.assertEqual(int(files.window_fp.sum()), 20)
                self.assertEqual(int(files.event_tp.sum()), 0)
                self.assertEqual(int(files.event_fn.sum()), 0)
                self.assertEqual(int(files.event_fp.sum()), len(alarms))
                self.assertGreater(len(alarms), 0)
                self.assertFalse(alarms.matched.any())

            # Execute the inventory's actual filename parser, without reading samples.
            notebook = json.loads((ROOT / 'notebooks_analysis/04_recording_inventory_split_balance.ipynb').read_text())
            parser = next(''.join(c['source']) for c in notebook['cells']
                          if c['cell_type'] == 'code' and 'FILENAME_PATTERN = ' in ''.join(c['source']))
            env = dict(re=re, Path=Path, pd=pd, np=np, DATA_DIR=root, display=lambda *args: None)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(parser, 'inventory-parser', 'exec'), env)
            inventory = env['records']
            self.assertEqual(len(inventory), 2)
            self.assertTrue(inventory.gesture.eq('switch_doubletap').all())
            self.assertTrue(inventory.source_action.eq('switch_doubletap').all())
            self.assertTrue(inventory.crop_start_frame.isna().all())
            self.assertEqual(set(inventory.split), {'train', 'validation'})
            share_cell = next(''.join(c['source']) for c in notebook['cells']
                              if c['cell_type'] == 'code' and 'HARD_NEGATIVE_CROPS_PER_RECORDING = ' in ''.join(c['source']))
            exec(compile(share_cell.split('overview = ')[0], 'inventory-share', 'exec'), env)
            self.assertTrue(inventory.gesture_share_weight.eq(1).all())

    def test_updated_notebook_sources_compile(self):
        for name in ('main_notebooks/evaluate.ipynb', 'main_notebooks/compare.ipynb',
                     'notebooks_analysis/04_recording_inventory_split_balance.ipynb'):
            notebook = json.loads((ROOT / name).read_text())
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'{name}:cell{index}', 'exec')


if __name__ == '__main__':
    unittest.main()
