# Hard-negative review workspace

Open [`../main_notebooks/mine_hard_negatives.ipynb`](../main_notebooks/mine_hard_negatives.ipynb) with the project Python kernel.

- `source_recordings/`: raw recordings and any existing sparse `.txt` labels, copied from `temp/`. Change the notebook input path to inspect other files.
- `crops/`: reviewed single-session raw CSVs, explicit empty `.txt` event labels, and `.mining.json` provenance. Copy the CSV **and** its matching `.txt` into the desired dataset's `train_data/` manually.

Candidates come from one selected checkpoint; additional checkpoints can be plotted alongside it. Change the recording filename, per-model thresholds, lookahead/cooldown, crop anchor, and core/context frame counts in the notebook.

The default marked hard interval is **40 frames before through 10 frames after** the chosen probability peak, inclusive (~0.5 s at 100 Hz). Export context defaults to **200 frames before through 99 frames after** (300 frames / 3 s). Set both context settings to `None` for just the short marked interval. For 5 s of context, for example, keep 200 preceding frames and use 299 following frames.

The existing `RecordedIMUDataset` creates one negative window per session: short sessions are edge-padded to `window_samples`; long sessions are truncated to their first `window_samples` frames. The exporter verifies that this fitting will retain the marked hard interval. A full minute in a single segment would otherwise contribute only its first window, which is why cropping matters here.

CSV crop bounds are half-open `[start, end)` original row indices. Timestamps restart at zero; the crop has one `segment_index=1` session. Sensor values stay raw; training applies its usual preprocessing. Each crop's JSON records the source/checkpoint hashes, thresholds, detection settings, and original frames. Re-exporting an identical selection is idempotent.
