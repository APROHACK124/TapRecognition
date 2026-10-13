"""Filename-selected hard negatives repeat in training, never validation."""

from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Subset

from tap_recognition.config import LSTMModelConfig, TrainConfig, TrainingConfig
from tap_recognition.dataset import RecordedIMUDataset, RecordedWindowMeta
from tap_recognition.feature_training import train_feature_lstm
from tap_recognition.imu_features import IMUFeatureConfig
from tap_recognition.training_sampling import print_training_sampling, repeat_hard_negatives
from train import main, train, train_one_epoch


def recorded_fixture():
    dataset = RecordedIMUDataset.__new__(RecordedIMUDataset)
    dataset.samples, dataset.window_meta = [], []
    for index, (name, cls) in enumerate([
        ('imu_Miguel_knock_once_fixture.csv', 0),
        ('imu_Miguel_negative_mining_fixture__crop_f000000_f000032.csv', 0),
        ('imu_Miguel_negative_mining_positive_fixture.csv', 1),
        ('imu_Miguel_knock_twice_right_fixture.csv', 2),
    ]):
        labels = np.zeros((32, 3), dtype=np.float32)
        labels[:, 0] = 1
        if cls:
            labels[12] = 0
            labels[12, cls] = 1
        dataset.samples.append((np.full((32, 6), index, dtype=np.float32), labels, cls))
        dataset.window_meta.append(RecordedWindowMeta(name, 12 if cls else None))
    return dataset


class TrainingSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.dataset = recorded_fixture()

    def test_repeats_only_filename_matching_negative_windows(self):
        sampled, table = repeat_hard_negatives(self.dataset, 5)
        self.assertEqual(len(sampled), 8)
        self.assertEqual(table.appearances_per_epoch.tolist(), [1, 5, 1, 1])
        self.assertEqual(table.hard_negative.tolist(), [False, True, False, False])
        seen = Counter(int(sampled[i]['imu'][0, 0]) for i in range(len(sampled)))
        self.assertEqual(seen, {0: 1, 1: 5, 2: 1, 3: 1})
        self.assertEqual(len(self.dataset), 4)
        self.assertEqual(len(self.dataset.window_meta), 4)
        self.assertEqual(int(table.appearances_per_epoch.sum()), len(sampled))

    def test_one_disables_repetition_and_keeps_dataset_identity(self):
        sampled, table = repeat_hard_negatives(self.dataset, 1)
        self.assertIs(sampled, self.dataset)
        self.assertTrue(table.appearances_per_epoch.eq(1).all())
        self.assertEqual(TrainingConfig().hard_negative_multiplier, 1)

    def test_multiple_directories_and_split_subset_use_correct_provenance(self):
        source = ConcatDataset([self.dataset, self.dataset])
        subset = Subset(source, [4, 5, 3, 1])
        sampled, table = repeat_hard_negatives(subset, 3)
        self.assertEqual(table.appearances_per_epoch.tolist(), [1, 3, 1, 3])
        self.assertEqual(len(sampled), 8)
        self.assertEqual(table.window_class.tolist(), [0, 0, 2, 0])

    def test_non_recorded_or_unmatched_windows_are_not_repeated(self):
        generic = [self.dataset[i] for i in range(len(self.dataset))]
        sampled, table = repeat_hard_negatives(generic, 5)
        self.assertIs(sampled, generic)
        self.assertFalse(table.hard_negative.any())
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            print_training_sampling(table, 5)
        self.assertIn('no effect', capture.getvalue())

    def test_invalid_multiplier_is_rejected(self):
        for value in (0, -1, 2.5, True, '5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                repeat_hard_negatives(self.dataset, value)

    def _config(self, output):
        cfg = TrainConfig(
            device='cpu', out_dir=output,
            model=LSTMModelConfig(cnn_channels=4, lstm_hidden=8,
                                  kernel_size=3, dilations=(1,), dropout=0.),
            training=TrainingConfig(epochs=1, batch_size=2, hard_negative_multiplier=3),
        )
        cfg.data.window_samples = 32
        return cfg

    def test_feature_training_repeats_only_train_and_normalizes_unique_windows(self):
        from tap_recognition.feature_training import _train_epoch

        with tempfile.TemporaryDirectory() as directory:
            cfg = self._config(directory)
            val = recorded_fixture()
            with contextlib.redirect_stdout(io.StringIO()), patch(
                'tap_recognition.feature_training._train_epoch', wraps=_train_epoch,
            ) as epoch, patch(
                'tap_recognition.feature_training.evaluate',
                return_value=dict(loss=1., frame_acc=.5, window_acc=.5, precision=.5, recall=.5),
            ) as evaluate:
                run_dir, _ = train_feature_lstm(cfg, IMUFeatureConfig(), self.dataset, val,
                                               label_delay_frames=4)
            train_loader = epoch.call_args.args[1]
            val_loader = evaluate.call_args.args[1]
            self.assertEqual(len(train_loader.dataset), 6)
            self.assertEqual(len(val_loader.dataset), 4)
            self.assertIs(val_loader.dataset.dataset, val)
            checkpoint = torch.load(run_dir / 'best.pt', weights_only=True)
            self.assertEqual(checkpoint['normalization_frames'], 4 * 32)
            self.assertEqual(checkpoint['train_config']['training']['hard_negative_multiplier'], 3)
            table = pd.read_csv(run_dir / 'training_sampling.csv')
            self.assertEqual(table.appearances_per_epoch.tolist(), [1, 3, 1, 1])
            self.assertEqual(len(self.dataset), 4)

    def test_cli_trainer_uses_repeated_loader_but_original_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._config(directory)
            val = recorded_fixture()
            with contextlib.redirect_stdout(io.StringIO()), patch(
                'train.build_datasets', return_value=(self.dataset, val),
            ), patch(
                'train.train_one_epoch', wraps=train_one_epoch,
            ) as epoch, patch(
                'train.evaluate', return_value=dict(loss=1., window_acc=.5, precision=.5, recall=.5),
            ) as evaluate:
                train(cfg)
            self.assertEqual(len(epoch.call_args.args[1].dataset), 6)
            self.assertIs(evaluate.call_args.args[1].dataset, val)
            checkpoint = torch.load(Path(directory) / 'best.pt', weights_only=True)
            self.assertEqual(checkpoint['train_config']['training']['hard_negative_multiplier'], 3)
            self.assertEqual(pd.read_csv(Path(directory) / 'training_sampling.csv').appearances_per_epoch.sum(), 6)

    def test_cli_option_sets_training_configuration(self):
        with patch('sys.argv', ['train.py', '--hard-negative-multiplier', '5']), patch('train.train') as trainer:
            main()
        self.assertEqual(trainer.call_args.args[0].training.hard_negative_multiplier, 5)

    def test_training_notebooks_compile_and_expose_the_setting(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('training_lstm.ipynb', 'training_gru.ipynb', 'training_feature_lstm.ipynb'):
            with self.subTest(notebook=name):
                notebook = json.loads((root / 'main_notebooks' / name).read_text())
                code = []
                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code':
                        source = ''.join(cell['source'])
                        compile(source, f'{name}:cell{index}', 'exec')
                        code.append(source)
                all_code = '\n'.join(code)
                self.assertIn('HARD_NEGATIVE_MULTIPLIER = 5', all_code)
                self.assertIn('hard_negative_multiplier=HARD_NEGATIVE_MULTIPLIER', all_code)
                self.assertIn('repeat_hard_negatives', all_code)
                if name != 'training_feature_lstm.ipynb':
                    dataset_cell = next(s for s in code if 'sampled_train_ds, sampling_table' in s)
                    from torch.utils.data import DataLoader
                    cfg = self._config('unused')
                    val = recorded_fixture()
                    namespace = dict(cfg=cfg, DataLoader=DataLoader,
                                     build_datasets=lambda _: (self.dataset, val),
                                     repeat_hard_negatives=repeat_hard_negatives,
                                     print_training_sampling=print_training_sampling)
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(dataset_cell, namespace)
                    self.assertEqual(len(namespace['train_loader'].dataset), 6)
                    self.assertIs(namespace['val_loader'].dataset, val)


if __name__ == '__main__':
    unittest.main()
