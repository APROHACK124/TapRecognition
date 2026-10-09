"""Verify a CNN/LSTM streaming export against full PyTorch forward().

Use the same arguments as tools/verify_harmony_step.py, with the LSTM .ms.
Run with LD_LIBRARY_PATH including the MindSpore Lite converter/lib and runtime/lib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tap_recognition.dataset import load_full_recording
from tap_recognition.model_factory import build_model
from tap_recognition.physics import IMUSimulator
from tap_recognition.model_feature_lstm import FeatureCNNLSTM
from verify_harmony_step import MindSporeStep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "onnx", "ms", "runtime-lib", "recording"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--reset-every", type=int, default=0, help="Reset every N frames to verify cold starts")
    parser.add_argument("--report", type=Path, help="Save verification provenance and maximum errors as JSON")
    args = parser.parse_args()
    assert args.reset_every >= 0
    assert args.frames > 0

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["model_config"]
    assert ckpt["model_type"] in ("lstm", "feature_lstm")
    assert (cfg["input_dim"], cfg["num_classes"], cfg["cnn_channels"],
            cfg["lstm_hidden"]) == (6, 3, 32, 64)
    layers = cfg["lstm_layers"]
    assert layers in (1, 2)
    state_shape = (layers, 1, 64)
    model = build_model(cfg, ckpt["model_type"]).eval()
    model.load_state_dict(ckpt["model_state"])
    assert args.frames > model.receptive_field
    feature_model = isinstance(model, FeatureCNNLSTM)
    cnn_frames = model._receptive_field if feature_model else model.receptive_field
    raw, fs = load_full_recording(args.recording)
    assert abs(fs - ckpt["train_config"]["data"]["sample_rate"]) < 0.5
    samples = IMUSimulator(sample_rate=fs).highpass(raw)[:args.frames]
    assert len(samples) == args.frames

    graph = onnx.load(args.onnx)
    onnx.checker.check_model(graph)
    assert not any(node.op_type in ("GRU", "LSTM") for node in graph.graph.node)
    input_names = ("imu", "cnn_buffer", "h_in", "c_in")
    output_names = ("prob", "h_out", "c_out", "cnn_buffer_out")
    output_shapes = ((1, 1, 3), state_shape, state_shape, (1, 32, cnn_frames))
    if feature_model:
        assert model.input_proj.normalization_fitted.item()
        input_names += ("feature_history", "feature_ready")
        output_names += ("feature_history_out", "feature_ready_out")
        output_shapes += ((1, model.feature_config.history_samples, 6), (1,))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    assert [x.name for x in session.get_inputs()] == list(input_names)
    assert [x.name for x in session.get_outputs()] == list(output_names)

    ms = MindSporeStep(args.runtime_lib, args.ms, input_names, output_shapes, output_names)
    torch.set_num_threads(1)
    errors = {label: 0.0 for label in ("torch_prob", "torch_h", "torch_c",
              "onnx_prob", "onnx_h", "onnx_c", "onnx_buffer",
              "ms_prob", "ms_h", "ms_c", "ms_buffer")}
    if feature_model:
        errors.update({label: 0.0 for label in ("onnx_feature_history", "ms_feature_history",
                                               "onnx_feature_ready", "ms_feature_ready")})

    with torch.inference_mode():
        boundaries = list(range(0, len(samples), args.reset_every or len(samples))) + [len(samples)]
        expected = np.zeros((1, len(samples), 3), dtype=np.float32)
        final_states = {}
        for start, end in zip(boundaries, boundaries[1:]):
            logits, final_states[end - 1] = model(torch.from_numpy(samples[start:end]).unsqueeze(0))
            expected[:, start:end] = logits.softmax(-1).numpy()
        for frame in range(len(samples)):
            if frame in boundaries[:-1]:
                state, buffer = None, None
                h_onnx = np.zeros(state_shape, dtype=np.float32)
                c_onnx, b_onnx = h_onnx.copy(), np.zeros((1, 32, cnn_frames), dtype=np.float32)
                h_ms, c_ms, b_ms = h_onnx.copy(), c_onnx.copy(), b_onnx.copy()
                extra_onnx = tuple(np.zeros(shape, dtype=np.float32) for shape in output_shapes[4:])
                extra_ms = tuple(x.copy() for x in extra_onnx)
            imu = samples[frame:frame+1][None, ...]
            prob, state, buffer = model.step(torch.from_numpy(imu), state, buffer)
            onnx_values = session.run(None, dict(zip(input_names, (imu, b_onnx, h_onnx, c_onnx, *extra_onnx))))
            ms_values = ms.step(imu, b_ms, h_ms, c_ms, *extra_ms)
            onnx_prob, h_onnx, c_onnx, b_onnx = onnx_values[:4]
            ms_prob, h_ms, c_ms, b_ms = ms_values[:4]
            extra_onnx, extra_ms = tuple(onnx_values[4:]), tuple(ms_values[4:])
            cnn_reference = buffer["cnn"] if feature_model else buffer
            reference = expected[:, frame:frame+1]
            for name, actual, target in (
                ("torch_prob", prob.numpy(), reference),
                ("onnx_prob", onnx_prob, reference),
                ("ms_prob", ms_prob, reference),
                ("onnx_h", h_onnx, state[0].numpy()),
                ("onnx_c", c_onnx, state[1].numpy()),
                ("onnx_buffer", b_onnx, cnn_reference.numpy()),
                ("ms_h", h_ms, state[0].numpy()),
                ("ms_c", c_ms, state[1].numpy()),
                ("ms_buffer", b_ms, cnn_reference.numpy()),
            ):
                assert np.isfinite(actual).all(), name
                errors[name] = max(errors[name], float(np.max(np.abs(actual - target))))
            if feature_model:
                history = np.zeros(output_shapes[4], dtype=np.float32)
                raw_history = buffer["raw"].numpy()
                history[:, -raw_history.shape[1]:] = raw_history
                for prefix, extra in (("onnx", extra_onnx), ("ms", extra_ms)):
                    for label, actual, target in (("feature_history", extra[0], history),
                                                  ("feature_ready", extra[1], np.ones(1))):
                        assert np.isfinite(actual).all(), label
                        key = f"{prefix}_{label}"
                        errors[key] = max(errors[key], float(np.max(np.abs(actual - target))))
            if frame in final_states:
                for label, actual, target in zip(("torch_h", "torch_c"), state, final_states[frame]):
                    errors[label] = max(errors[label], float(np.max(np.abs(actual.numpy() - target.numpy()))))

    for name, error in errors.items():
        print(f"{name}: {error:.8g}")
        assert error < (2e-3 if name.startswith("ms_") else 1e-4), f"{name} diverged by frame {frame}"
    print(f"PASS: {len(samples)} frames, all {layers} LSTM layer(s)' hidden and cell states fed back independently")
    if args.report:
        report = {"passed": True, "frames": len(samples), "reset_every": args.reset_every,
                  "model_type": ckpt["model_type"], "epoch": ckpt["epoch"], "errors": errors,
                  "recording": str(args.recording), "input_names": input_names, "output_names": output_names}
        for label in ("checkpoint", "onnx", "ms"):
            report[f"{label}_sha256"] = hashlib.sha256(getattr(args, label).read_bytes()).hexdigest()
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
