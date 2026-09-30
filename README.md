# IMU Double-Tap Recognition on PC

Recognize **double-tap** gestures on a laptop/PC chassis using IMU (accelerometer + gyroscope) data. This project provides:

1. A **physical/mathematical model** of tap-induced vibrations
2. A **causal CNN + GRU** neural network for streaming inference
3. **Interactive labeling tools** for collected IMU recordings
4. Training on **recorded labels** or **synthetic simulation**

## Mathematical Model (Summary)

See [docs/mathematical_model.md](docs/mathematical_model.md) for the full derivation.

A double-tap produces two impulsive mechanical responses within an inter-tap interval \(\Delta t \in [T_{\min}, T_{\max}]\). Each tap excites damped oscillations in the chassis; the IMU observes:

\[
\mathbf{y}(t) = \mathbf{R}(\boldsymbol{\theta})\mathbf{g} + \sum_{k=1}^{2} h(t - t_k; \mathbf{p}_k) + \boldsymbol{\eta}(t)
\]

where \(\mathbf{y} = [a_x, a_y, a_z, \omega_x, \omega_y, \omega_z]^\top\).

## Project Structure

```
TapRecognition/
├── docs/mathematical_model.md   # Full physical & detection model
├── data/
│   ├── train_data/              # Training recordings + sidecar .txt labels
│   └── valid_data/              # Validation recordings + sidecar .txt labels
├── tools/
│   ├── label_imu.py             # Interactive single-file labeling GUI
│   ├── label_all_imu.py         # Batch labeling for all CSVs under data/
│   ├── export_for_harmony.py    # Export checkpoint to ONNX for HarmonyOS
│   └── convert_to_ms.ps1        # ONNX → .ms via MindSpore Lite converter
├── tap_recognition/
│   ├── config.py                # Hyperparameters (TrainConfig dataclasses)
│   ├── physics.py               # Tap signal synthesis (IMUSimulator)
│   ├── labels.py                # Gaussian second-tap soft labels
│   ├── recording.py             # CSV loading, JSON annotation helpers
│   ├── dataset.py               # IMUDoubleTapDataset + RecordedIMUDataset
│   ├── model.py                 # Causal CNN + GRU
│   └── inference.py             # Streaming online detector
├── train.py                     # Training script
├── demo.py                      # End-to-end synthetic demo
├── checkpoints/                 # Saved model weights (best.pt, last.pt)
└── requirements.txt
```

## Quick Start

Requires Python 3.8+ with PyTorch.

```bash
pip install -r requirements.txt
pip install pandas   # required for tools/label_imu.py
```

### 1. Label collected IMU data

Recordings are CSV files with columns `timestamp_ms`, `acc_x/y/z`, `gyro_x/y/z`, `segment_index`, etc.

**Single file** — open the interactive GUI, drag on either plot to select each double-tap region, then close the window to save:

```bash
python tools/label_imu.py data/train_data/imu_ZR_knock_twice_20260712_103508.csv
```

**All files under `data/`** — batch process by action type:

```bash
python tools/label_all_imu.py --skip-existing
```

| Recording type | Filename pattern | Label file |
|----------------|------------------|------------|
| Double-tap | `*knock_twice*` | Sidecar `.txt` with trigger frames (interactive GUI) |
| Single-tap | `*knock_once*` | Empty `.txt` (negative class) |
| Arbitrary motion | `*arbitrary*` | Empty `.txt` (negative class) |

Labels are saved as a **sidecar text file** next to each CSV:

```
data/train_data/imu_ZR_knock_twice_20260712_103508.csv
data/train_data/imu_ZR_knock_twice_20260712_103508.txt
```

Each line in a double-tap label file is `frame_index, action_label` (e.g. `160, 1`), where `frame_index` is the global row index in the CSV and `action_label = 1` marks the second tap.

The GUI plots **ACC energy delta** and **GYRO energy**, picks the minimum normalized gyro energy within each dragged span as the trigger frame, and marks it with a red vertical line.

> **Windows path note:** use raw strings for backslash paths in `label_imu.py`, e.g. `r'data\train_data\file.csv'`, to avoid `\t` being parsed as a tab character.

### 2. Train on labeled recordings

Place training CSVs in `data/train_data/` and validation CSVs in `data/valid_data/`. Edit defaults in `tap_recognition/config.py` if needed (`TrainConfig`, `DataConfig`, etc.).

```bash
python train.py
```

For synthetic data, set `DataConfig.mode = "synthetic"` in `tap_recognition/config.py`.

Checkpoints are written to `checkpoints/best.pt` and `checkpoints/last.pt`.

#### CNN–LSTM experiment

`main_notebooks/baseline_lstm.ipynb` follows the GRU baseline with a one-layer,
unidirectional LSTM. Set `LSTM_HIDDEN_SIZE` to **64** for the same hidden width,
or **55** for approximately the same total parameter count as the GRU-64 model.
Match `LABEL_DELAY_FRAMES`, `AMPLITUDE_AUGMENTATION`, data, and training settings
to the GRU run being compared; high-pass filtering remains enabled.

The shared training script also supports the LSTM:

```bash
python train.py --recurrent-type lstm --recurrent-hidden 64 --out-dir checkpoints_lstm64
python train.py --recurrent-type lstm --recurrent-hidden 55 --out-dir checkpoints_lstm55
```

The script retains its existing augmentation and unshifted-target behavior;
use the notebook's switches for delayed-target/augmentation experiments.
LSTM checkpoints record `model_type="lstm"` and use `lstm_hidden`/`lstm_layers`
in `model_config`. `tap_recognition.model_factory.build_model` loads either
architecture, including legacy GRU checkpoints without `model_type`.
Streaming LSTM inference carries both hidden and cell state.

### 4. Run inference

```bash
python -m tap_recognition.inference --checkpoint checkpoints/best.pt
```

Or run the full synthetic pipeline demo:

```bash
python demo.py
```

## Model Architecture

```
IMU window [B, T, 6]
    │
    ▼
Causal Conv1D stack (left-padded, no future leakage)
    │
    ▼
GRU (maintains hidden state for streaming)
    │
    ▼
Sigmoid → P(double-tap at current frame)
```

**Causality** is enforced by:

- Left-only padding in all conv layers
- GRU processing strictly in temporal order
- Online inference updates one sample at a time with persistent GRU state

## Configuration

Hyperparameters are defined as dataclasses in `tap_recognition/config.py`:

| Class | Contents |
|-------|----------|
| `DataConfig` | Dataset mode, paths, window size, negative sampling |
| `LabelConfig` | Gaussian soft-label parameters |
| `ModelConfig` | CNN + GRU architecture |
| `TrainingConfig` | Epochs, batch size, learning rate, early stopping |
| `TrainConfig` | Top-level wrapper (seed, device, out_dir, nested configs) |

## Deploy to HarmonyOS (`.pth` → `.ms`)

HarmonyOS edge inference uses **MindSpore Lite** `.ms` models. The official converter is `converter_lite` ([MindSpore Lite 文档](https://www.mindspore.cn/lite/docs/zh-CN/master/converter/converter_tool.html)).

> **Note:** Training checkpoints (`best.pt`) store weights + config, not a deployable graph. Convert in two steps: **PyTorch → ONNX → MS**.

### Step 1: Export ONNX from checkpoint

```bash
python tools/export_for_harmony.py --checkpoint checkpoints/best.pt --output checkpoints/tap_recognition.onnx
```

Input shape: `[1, 64, 6]` (batch, time, IMU channels). Outputs: `logits [1, 64, 1]`, `h_n` (GRU state).

Requires: `pip install onnx` (do **not** install `onnxscript` on Python 3.8).

### Step 2: Convert ONNX to `.ms`

1. Download **MindSpore Lite Windows x64** from the [official download page](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html) (e.g. `mindspore-lite-2.9.0-win-x64.zip`).
2. Extract the package and add `tools/converter/lib` to `PATH`.
3. Run:

```powershell
# Windows
$env:PATH = "<MSLITE_ROOT>\tools\converter\lib;$env:PATH"
<MSLITE_ROOT>\tools\converter\converter\converter_lite.exe `
  --fmk=ONNX `
  --modelFile=checkpoints/tap_recognition.onnx `
  --outputFile=checkpoints/tap_recognition `
  --inputShape="imu:1,64,6"
```

Or use the helper script:

```powershell
.\tools\convert_to_ms.ps1 -MsLiteRoot "<extracted_mindspore_lite_folder>"
```

Success prints `CONVERT RESULT SUCCESS:0` and produces `checkpoints/tap_recognition.ms`.

### Alternative: direct PyTorch conversion (Linux only)

Prebuilt Windows packages **do not** support `--fmk=PYTORCH`. On Linux you must compile MindSpore Lite from source with:

```bash
export MSLITE_ENABLE_CONVERT_PYTORCH_MODEL=on
export LIB_TORCH_PATH="/path/to/libtorch"
```

Then export a TorchScript model (not a training checkpoint) and run `converter_lite --fmk=PYTORCH`.

### HarmonyOS inference

Load `tap_recognition.ms` in your HarmonyOS app via the MindSpore Lite C++/ArkTS API. See [HarmonyOS MindSpore Lite 模型转换](https://www.seaxiang.com/blog/vkzEm4) for the end-to-end deployment flow.

---
