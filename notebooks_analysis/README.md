# Experiment notebooks

Open notebooks using the repository's `.venv` Python kernel and run cells in order.
Core dependencies are listed in the repository's `requirements.txt`; additional
notebook dependencies are in `notebooks/requirements.txt`.

## 03 — Reset state versus continuous inference

[`03_state_reset_vs_continuous.ipynb`](03_state_reset_vs_continuous.ipynb) is an
executed evaluation of `checkpoints/best.pt` on all train and validation recordings.
It includes:

- Reproduction of the checkpoint's original 300-frame validation metrics.
- A paired 2×2 CNN/GRU reset-versus-carry comparison on original recording frames.
- Numerical batch/chunk/single-sample equivalence and event-matching checks.
- Window/frame/event metrics, confusion matrices, probability differences,
  recording-level bootstrap intervals, temporal plots and full-recording examples.
- Threshold and event-matching tolerance sensitivity.

The continuous arm is a checked streaming equivalent of `forward()`, **not** the
existing `model.step()` implementation, whose discrepancy is measured separately.
All inference uses frozen weights in eval mode. No retraining is performed.

Change `CHECKPOINT` in the first code cell to evaluate another saved model. Tables
and a manifest with input/checkpoint/code hashes and environment versions are
written to `notebooks/results/state_reset_<checkpoint-directory>_<checkpoint-stem>/`.
Embedded plots make the executed notebook readable without rerunning it.

The reusable numerical helpers are in [`state_reset_analysis.py`](state_reset_analysis.py).
Experimental settings, orchestration, plots and interpretation are in the notebook.

To execute and save outputs non-interactively from the repository root:

```bash
./.venv/bin/python -c "import nbformat; from nbclient import NotebookClient; from pathlib import Path; p = Path('notebooks/03_state_reset_vs_continuous.ipynb'); n = nbformat.read(p, as_version=4); NotebookClient(n, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}}).execute(); nbformat.write(n, p)"
```

The notebook resolves paths from either the repository root or `notebooks/`.
