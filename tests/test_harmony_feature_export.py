"""Feature export must preserve cold starts, trailing RMS, and frozen stats."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from tap_recognition.imu_features import CausalIMUFeatures, IMUFeatureConfig
from tap_recognition.model_factory import build_model
from tools.export_step_for_harmony import export_step_onnx


class FeatureExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_fixed_features_match_full_sequence_including_first_delta_and_reset(self):
        for cfg in (IMUFeatureConfig(), IMUFeatureConfig(rms_windows=(2, 7), magnitudes=False),
                    IMUFeatureConfig(differences=False, difference_magnitudes=False, rms_windows=(5,))):
            features = CausalIMUFeatures(cfg)
            for x in (torch.zeros(1, 50, 6), torch.randn(1, 50, 6) * 5,
                      torch.randn(1, 50, 6) * 1e-6):
                history, ready = torch.zeros(1, cfg.history_samples, 6), torch.zeros(1)
                expected = features(x)
                for frame in range(x.shape[1]):
                    got, history, ready = features.step(x[:, frame:frame+1], history, ready)
                    torch.testing.assert_close(got, expected[:, frame:frame+1], atol=2e-6, rtol=1e-6)
                # A fresh stream beginning with a large sample still has zero delta.
                got, _, _ = features.step(x[:, :1], torch.zeros_like(history), torch.zeros_like(ready))
                torch.testing.assert_close(got, expected[:, :1], atol=2e-6, rtol=1e-6)

    def test_feature_onnx_matches_full_forward_with_independent_state_feedback(self):
        cfg = dict(input_dim=6, num_classes=3, cnn_channels=4, lstm_hidden=12,
                   lstm_layers=2, kernel_size=3, dilations=(1, 2), dropout=0.1,
                   feature_config=IMUFeatureConfig().to_dict())
        model = build_model(cfg, "feature_lstm").eval()
        model.input_proj.fit_normalization([torch.randn(2, 40, 6)])
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, output = Path(tmp) / "best.pt", Path(tmp) / "step.onnx"
            torch.save(dict(model_type="feature_lstm", model_config=cfg, model_state=model.state_dict()), checkpoint)
            export_step_onnx(checkpoint, output)
            graph = onnx.load(output)
            onnx.checker.check_model(graph)
            self.assertFalse(any(n.op_type in ("LSTM", "GRU") for n in graph.graph.node))
            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            names = [x.name for x in session.get_inputs()]
            for x in (torch.randn(1, 70, 6) * 4, torch.zeros(1, 50, 6)):
                states = (np.zeros((1, 4, 7), np.float32), np.zeros((2, 1, 12), np.float32),
                          np.zeros((2, 1, 12), np.float32), np.zeros((1, 14, 6), np.float32), np.zeros(1, np.float32))
                with torch.inference_mode():
                    expected, final = model(x)
                    torch_state, buffer = None, None
                    for frame in range(x.shape[1]):
                        imu = x[:, frame:frame+1]
                        prob, torch_state, buffer = model.step(imu, torch_state, buffer)
                        p, h, c, cnn, history, ready = session.run(None, dict(zip(names, (imu.numpy(), *states))))
                        np.testing.assert_allclose(p, expected.softmax(-1)[:, frame:frame+1], atol=1e-5, rtol=1e-5)
                        np.testing.assert_allclose(h, torch_state[0], atol=1e-5, rtol=1e-5)
                        np.testing.assert_allclose(c, torch_state[1], atol=1e-5, rtol=1e-5)
                        np.testing.assert_allclose(cnn, buffer["cnn"], atol=1e-5, rtol=1e-5)
                        states = (cnn, h, c, history, ready)
                    np.testing.assert_allclose(h, final[0], atol=1e-5, rtol=1e-5)
                    np.testing.assert_allclose(c, final[1], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
