"""Export TapRecognition checkpoint to ONNX for MindSpore Lite (.ms) conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tap_recognition.model import CausalCNNGRU


def load_model(checkpoint_path: Path, device: torch.device) -> CausalCNNGRU:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = CausalCNNGRU(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    window_samples: int = 64,
    input_dim: int = 6,
    opset: int = 14,
) -> Path:
    device = torch.device("cpu")
    model = load_model(checkpoint_path, device)
    dummy_input = torch.randn(1, window_samples, input_dim, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["imu"],
        output_names=["logits", "h_n"],
        dynamic_axes={
            "imu": {0: "batch", 1: "time"},
            "logits": {0: "batch", 1: "time"},
            "h_n": {1: "batch"},
        },
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX for HarmonyOS")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="PyTorch checkpoint path",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/tap_recognition.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--window-samples", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    onnx_path = export_onnx(
        Path(args.checkpoint),
        Path(args.output),
        window_samples=args.window_samples,
        input_dim=args.input_dim,
        opset=args.opset,
    )
    print(f"Exported ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
