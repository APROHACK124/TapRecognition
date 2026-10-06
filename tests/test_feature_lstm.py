"""Causal features, train-only statistics, streaming, and experiment round trips."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from notebooks_analysis.state_reset_analysis import overlap_stream_logits
from tap_recognition.config import LSTMModelConfig, ModelConfig, TrainConfig, TrainingConfig
from tap_recognition.feature_training import DelayedTargets, train_feature_lstm
from tap_recognition.imu_features import CausalIMUFeatures, FeatureProjection, IMUFeatureConfig
from tap_recognition.inference import OnlineDoubleTapDetector
from tap_recognition.model_factory import build_model
from tap_recognition.model_lstm import CausalCNNLSTM
from tap_recognition.model import CausalCNNGRU


class FeatureLSTMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_known_features_and_no_future_or_input_mutation(self):
        cfg = IMUFeatureConfig(rms_windows=(2,))
        features = CausalIMUFeatures(cfg)
        x = torch.tensor([[[3., 4., 0., 0., 0., 2.], [6., 8., 0., 0., 0., 4.],
                           [0., 0., 0., 0., 0., 0.]]])
        original = x.clone()
        out = features(x)
        torch.testing.assert_close(out[..., :6], x)
        torch.testing.assert_close(out[0, :, 6:8], torch.tensor([[5., 2.], [10., 4.], [0., 0.]]))
        torch.testing.assert_close(out[0, 0, 8:16], torch.zeros(8))
        torch.testing.assert_close(out[0, 1, 8:14], x[0, 1] - x[0, 0])
        torch.testing.assert_close(out[0, 1, 14:16], torch.tensor([5., 2.]))
        torch.testing.assert_close(out[0, 1, 16:], torch.tensor([62.5, 10.]).sqrt() - 1e-4,
                                   atol=1e-6, rtol=1e-6)
        changed = x.clone()
        changed[:, 2:] = 1000
        torch.testing.assert_close(features(changed)[:, :2], out[:, :2])
        torch.testing.assert_close(x, original)
        self.assertEqual(out.shape[-1], len(cfg.names))
        self.assertEqual(len(IMUFeatureConfig().names), 20)
        self.assertEqual(IMUFeatureConfig().history_samples, 14)
        with self.assertRaises(ValueError):
            IMUFeatureConfig(rms_windows=(0,))

    def test_training_statistics_stable_and_validation_does_not_refit(self):
        cfg = IMUFeatureConfig(magnitudes=False, differences=False,
                               difference_magnitudes=False, rms_windows=())
        projection = FeatureProjection(cfg, 4)
        x = torch.arange(60, dtype=torch.float32).reshape(2, 5, 6)
        frames = projection.fit_normalization([x[:1], x[1:]])
        flat = x.flatten(0, 1)
        self.assertEqual(frames, 10)
        torch.testing.assert_close(projection.mean, flat.mean(0))
        torch.testing.assert_close(projection.scale, flat.std(0, correction=0))
        stats = (projection.mean.clone(), projection.scale.clone())
        projection.eval()(torch.full((1, 5, 6), 1e5))
        torch.testing.assert_close(projection.mean, stats[0])
        torch.testing.assert_close(projection.scale, stats[1])
        projection.fit_normalization([torch.zeros_like(x)])
        self.assertTrue((projection.scale >= 1e-3).all())
        output = projection(torch.zeros_like(x))
        output.sum().backward()
        self.assertTrue(torch.isfinite(projection.linear.weight.grad).all())
        with self.assertRaises(ValueError):
            projection.fit_normalization([])

    def test_streaming_and_future_independence_with_feature_history(self):
        for features in (IMUFeatureConfig(), IMUFeatureConfig(
            magnitudes=False, differences=False, difference_magnitudes=False, rms_windows=())):
            with self.subTest(features=features):
                config = LSTMModelConfig(cnn_channels=4, lstm_hidden=12, lstm_layers=2,
                                          kernel_size=3, dilations=(1, 2), dropout=0.)
                model = build_model(config.to_model_kwargs() | {"feature_config": features.to_dict()}).eval()
                model.input_proj.fit_normalization([torch.randn(2, 30, 6)])
                model = model.double()
                x = torch.randn(2, 65, 6, dtype=torch.float64)
                with torch.no_grad():
                    expected, expected_state = model(x)
                    changed = x.clone()
                    changed[:, 35:] += 20
                    torch.testing.assert_close(model(changed)[0][:, :35], expected[:, :35])
                    for chunks in ([1, 3, 4, 7, 50], [20, 45]):
                        actual = overlap_stream_logits(model, x, chunks)
                        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
                    state, buffer, probs = None, None, []
                    for t in range(x.shape[1]):
                        p, state, buffer = model.step(x[:, t], state, buffer)
                        probs.append(p)
                    torch.testing.assert_close(torch.cat(probs, 1), expected.softmax(-1), atol=1e-10, rtol=1e-10)
                    for got, wanted in zip(state, expected_state):
                        torch.testing.assert_close(got, wanted, atol=1e-10, rtol=1e-10)
                    reset_probability, _, _ = model.step(x[:, 0], None, None)
                    torch.testing.assert_close(reset_probability, expected[:, :1].softmax(-1))

    def test_factory_legacy_dispatch_and_feature_checkpoint_round_trip(self):
        self.assertIsInstance(build_model(ModelConfig().to_model_kwargs()), CausalCNNGRU)
        self.assertIsInstance(build_model(LSTMModelConfig().to_model_kwargs()), CausalCNNLSTM)
        kwargs = LSTMModelConfig().to_model_kwargs() | {"feature_config": IMUFeatureConfig().to_dict()}
        model = build_model(kwargs, "feature_lstm").eval()
        model.input_proj.fit_normalization([torch.randn(2, 40, 6)])
        buffer = io.BytesIO()
        torch.save(dict(model_config=kwargs, model_type="feature_lstm", model_state=model.state_dict()), buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_model(checkpoint["model_config"], checkpoint["model_type"]).eval()
        restored.load_state_dict(checkpoint["model_state"])
        with torch.no_grad():
            x = torch.randn(1, 40, 6)
            torch.testing.assert_close(restored(x)[0], model(x)[0])
        detector = OnlineDoubleTapDetector(restored)
        detector.process_sample(x[0, 0].numpy())
        self.assertIsInstance(detector._cnn_buffer, dict)
        detector.reset()
        self.assertIsNone(detector._cnn_buffer)

    def test_one_epoch_checkpoint_history_and_delayed_labels_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = TrainConfig(device="cpu", out_dir=temporary,
                              model=LSTMModelConfig(cnn_channels=4, lstm_hidden=12,
                                                     kernel_size=3, dilations=(1,), dropout=0.),
                              training=TrainingConfig(epochs=1, batch_size=2))
            cfg.data.window_samples = 32
            samples = []
            for cls in (1, 0, 2, 0):
                labels = torch.zeros(32, 3)
                labels[:, 0] = 1
                if cls:
                    labels[12] = 0
                    labels[12, cls] = 1
                samples.append(dict(imu=torch.randn(32, 6), frame_labels=labels,
                                    window_label=torch.tensor(cls)))
            original_labels = samples[0]["frame_labels"].clone()
            original_config = cfg.to_dict()
            delayed = DelayedTargets(samples, 4)
            self.assertEqual(int(delayed[0]["frame_labels"][16].argmax()), 1)
            with contextlib.redirect_stdout(io.StringIO()):
                run, history = train_feature_lstm(cfg, IMUFeatureConfig(), samples, samples[:2],
                                                  label_delay_frames=4)
            self.assertEqual(cfg.to_dict(), original_config)
            torch.testing.assert_close(samples[0]["frame_labels"], original_labels)
            self.assertEqual(len(history), 1)
            self.assertTrue((run / "history.csv").is_file())
            initial = torch.load(run / "initial.pt", weights_only=True)
            checkpoint = torch.load(run / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["normalization_fit_split"], "train")
            self.assertEqual(checkpoint["positive_label_delay_frames"], 4)
            torch.testing.assert_close(checkpoint["model_state"]["input_proj.mean"],
                                       initial["model_state"]["input_proj.mean"])
            self.assertTrue(any(not torch.equal(checkpoint["model_state"][name], value)
                                for name, value in initial["model_state"].items() if name.endswith("weight")))
            restored = build_model(checkpoint["model_config"], checkpoint["model_type"]).eval()
            restored.load_state_dict(checkpoint["model_state"])
            with torch.no_grad():
                logits, _ = restored(samples[0]["imu"].unsqueeze(0))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(checkpoint["model_config"]["feature_config"]["version"], 1)
            self.assertEqual(len(json.loads((run / "experiment.json").read_text())["feature_names"]), 20)
            # A rerun creates another directory, retaining the original checkpoint bytes.
            before = (run / "best.pt").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                another, _ = train_feature_lstm(cfg, IMUFeatureConfig(), samples, samples[:2],
                                                label_delay_frames=4, amplitude_augmentation=False)
            self.assertNotEqual(run, another)
            self.assertEqual((run / "best.pt").read_bytes(), before)

    def test_notebook_is_unexecuted_valid_json_and_compiles(self):
        path = Path(__file__).resolve().parents[1] / "main_notebooks/training_feature_lstm.ipynb"
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec")


if __name__ == "__main__":
    unittest.main()
