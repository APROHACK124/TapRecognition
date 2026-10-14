# Hard-negative review workspace

Open [`../main_notebooks/mine_hard_negatives.ipynb`](../main_notebooks/mine_hard_negatives.ipynb) with the project Python kernel.

- `source_recordings/`: full raw originals, including `negative_mining`, hard-negative, and `normal_usage` recordings. Optional staging from `temp/` is enabled with `STAGE_NEW_RECORDINGS=True`.
- `crops/`: reviewed single-session raw CSVs, explicit empty `.txt` event labels, and `.mining.json` provenance. Copy **all three files** together into the desired dataset's split manually; keep the original CSV available.

The batch section uses alarms from **at least one** selected model, with per-model left/right thresholds. Select `BATCH_FILES` or enable `BATCH_SELECT_ALL`; `BATCH_MODELS` controls the model set. `BATCH_MIN_GAP_FRAMES=300` skips the next 300 anchor frames after a kept failure, across all models. Overlapping/boundary-duplicate crops are also removed by default. Per-file and overall reports show final crop count, crops/minute, total original time, and summed crop-segment time, including zero-crop originals. `BATCH_KEEP_IDS` filters the final set, `BATCH_PREVIEW_ID` browses it, and `SAVE_BATCH_CROPS=True` writes it. The original optional single-checkpoint inspection remains available, with saving separately enabled by `SAVE_REVIEWED_CROPS`.

The default marked hard interval is **40 frames before through 10 frames after** the chosen probability peak, inclusive (~0.5 s at 100 Hz). Export context defaults to **200 frames before through 99 frames after** (300 frames / 3 s). Set both context settings to `None` for just the short marked interval. For 5 s of context, for example, keep 200 preceding frames and use 299 following frames.

The existing `RecordedIMUDataset` creates one negative window per session: short sessions are edge-padded to `window_samples`; long sessions are truncated to their first `window_samples` frames. The exporter verifies that this fitting will retain the marked hard interval. A full minute in a single segment would otherwise contribute only its first window, which is why cropping matters here.

CSV crop bounds are half-open `[start, end)` original row indices. Timestamps restart at zero; the crop has one `segment_index=1` session. Sensor values stay raw; training applies its usual preprocessing. Each crop's JSON records the source/checkpoint hashes, thresholds, detection settings, and original frames. Re-exporting an identical selection is idempotent.

Normal-use crops retain their original filename prefix but use action `negative_mining` and report type `hard_negative`; they also participate in the configurable hard-negative training repetition. New JSON provenance records the original duration/frames/sample rate/type and all batch model settings. Full normal-use evaluation in `evaluate.ipynb` and `compare.ipynb` replays complete `normal_usage` originals separately, never substituting mined crops. A source whose crops were trained on is not an independent held-out test recording.
