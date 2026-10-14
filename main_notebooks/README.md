# Training, Evaluation, And Comparison Notebooks

Use these notebooks for controlled recorded-data experiments. Open them with the repository `.venv` Python kernel and run cells in order. The expected sequence is train one run, evaluate that saved run, then compare its exports with other evaluated runs.

Do not train multiple candidates simultaneously on this machine. Preserve every completed run's checkpoints, history, and evaluation export under a distinct directory.

## Canonical Workflow

```text
training_gru.ipynb or training_lstm.ipynb
                    |
                    v
          <run directory>/best.pt
          <run directory>/last.pt
          <run directory>/history.csv
                    |
                    v
              evaluate.ipynb
                    |
                    v
  <run directory>/threshold_comparison/*.csv
                    |
                    v
            compare_models.ipynb
```

All compared runs must use compatible data, preprocessing, matching rules, and threshold grid. `compare_models.ipynb` verifies those requirements before it combines results.

## Train A GRU

[`training_gru.ipynb`](training_gru.ipynb) trains the original causal CNN plus GRU architecture.

Set before training:

- seed;
- train and validation directories;
- output directory or run name;
- `LABEL_DELAY_FRAMES`;
- standard training settings in `TrainConfig` and `TrainingConfig`.

The notebook applies positive-window amplitude augmentation and writes the target delay into checkpoint metadata. Use `LABEL_DELAY_FRAMES = 0` for an unshifted baseline and a positive value only for a deliberate delayed-target experiment.

## Train An LSTM

[`training_lstm.ipynb`](training_lstm.ipynb) changes only the recurrent head. It keeps the causal CNN and classifier head aligned with the GRU experiment.

Set before training:

- `LSTM_HIDDEN_SIZE`;
- `LSTM_LAYERS`;
- `LABEL_DELAY_FRAMES`;
- `AMPLITUDE_AUGMENTATION`;
- seed, data directories, output directory, and training settings.

For a controlled GRU/LSTM comparison, keep all non-architecture settings identical. The LSTM checkpoint records `model_type="lstm"`, layer count, delay, and augmentation setting.

## Train An LSTM With Engineered IMU Features

[`training_feature_lstm.ipynb`](training_feature_lstm.ipynb) is an opt-in experiment based on `training_lstm.ipynb`. Set its split, training seed, target delay, feature groups, and optimizer settings, then run in order. It trains from scratch into a fresh timestamped directory under `testing_checkpoints/lstm64_features/`.

- Inputs remain six high-passed IMU axes. Feature engineering runs inside the model, after any amplitude augmentation.
- Default 20 channels: signed axes, acceleration/gyro magnitudes, per-sample differences and their magnitudes, and trailing RMS at 5/15 frames. All features are causal.
- Per-feature normalization is fitted on unaugmented training windows only, frozen for validation/inference, and stored in checkpoint buffers. Feature settings and channel order are also saved.
- Checkpoints include the usual delay, augmentation, training configuration, and validation metrics, plus `initial.pt`, run metadata, and an input-file manifest. Existing data and run directories are read-only inputs.
- Evaluate the printed run directory with `evaluate.ipynb`; the factory recognizes `model_type="feature_lstm"`. Restart that kernel after installing the new code. Keep `EVALUATION_SEED`, data, and alarm-matching settings identical across the feature and raw-axis runs.
- `forward()` and the analysis overlap-streaming reference accept ordinary six-channel tensors. Python `step()` uses a separate `{raw, cnn}` buffer for feature/CNN history and resets with `None`, while carrying the usual LSTM hidden/cell state.

The feature extractor and training entry point live in `tap_recognition/imu_features.py`, `tap_recognition/model_feature_lstm.py`, and `tap_recognition/feature_training.py`. The existing raw-axis training notebook and architecture remain the baseline.

## Fine-Tune LSTM64-Delay-8

[`fine_tuning_lstm64d8.ipynb`](fine_tuning_lstm64d8.ipynb) initializes from `testing_checkpoints/lstm64d8/best.pt` and fine-tunes all layers on the mixed old/new data in `data_05_10_26/`.

- Architecture, seed, target delay, preprocessing, augmentation, and loss settings come from the source checkpoint.
- Default fine-tuning learning rate is `1e-4`, with up to 30 epochs and early stopping.
- Only training windows from filenames containing both `20261005` and `switch_hand` are repeated five times per epoch. Old training windows appear once; validation is not repeated.
- Each timestamped run under `testing_checkpoints/lstm64d8_hand_switch_ft/` preserves an exact `initial.pt` source copy, new checkpoints, history, and sampling/provenance records.
- The epoch-0 starting model participates in checkpoint selection. The source model and `training_lstm.ipynb` remain available for the separate from-scratch control.

For a three-model comparison, evaluate the original, fine-tuned, and from-scratch models on the same expanded validation set at the fixed `0.50 / 0.50`, `lookahead_12f` operating point. Use a separate evaluation copy for the original so its historical exports are preserved. The new notebook explains the required data-path override and matched sampling setup for the control.

## Evaluate One Saved Run

[`evaluate.ipynb`](evaluate.ipynb) performs full-recording evaluation; it does not retrain or overwrite the selected checkpoint. Set `EVALUATION_DATA_ROOT` in its setup cell to evaluate multiple checkpoints on the same prepared data split. That override applies to window metrics, continuous metrics, and exported comparison fingerprints. Its default `EVALUATION_OUTPUT_DIR` is separate from the checkpoint when an override is active, preserving historical exports.

Set `RUN_DIR` to the completed run directory. The notebook reconstructs the correct GRU or LSTM model from checkpoint metadata, reads the target delay, and produces:

- training-history plots;
- window, frame, and matched-event metrics;
- instant and look-ahead policies;
- left/right threshold-pair sweeps;
- latency, early/late detection, false-alarm, and recording-type analysis;
- CSV exports in `RUN_DIR/threshold_comparison/`.

Evaluation data is used for model selection and threshold selection. Its results are not independent test estimates.

Threshold selection defaults to `FP_SELECTION_METRIC = 'negative_fp_per_min'`: the FP-budget modes and `recall_target_min_fp` use alarms per minute only in validation recordings with no annotated double taps. FP-rate tie-breakers use the same setting. Set it to `'fp_per_min'` to use all unmatched alarms over all validation minutes instead. `FP_PER_MIN_LIMIT` is a strict upper bound on the selected metric; the export metadata records `fp_selection_metric`. Both rates are still reported, and `max_f1` still uses event F1 as its primary objective. Rerun evaluation/export for each compared checkpoint after changing this setting.

`evaluate.ipynb` and `compare.ipynb` use true original-recording exposure for `negative_mining` crops: resolve each crop's `.mining.json`, verify the original CSV hash, measure its frames/sample rate, and count that source once per evaluated split/model. A source's minutes are divided among its evaluated crops so ordinary grouped totals do not overcount. Other recordings retain included duration. `minutes` is the allocated FP-rate denominator; `evaluated_minutes` preserves actual input duration. Evaluation exports `recording_duration_audit.csv`, `duration_basis`, and `duration_sha256`. Keep the originals in `hard_negative_mining/source_recordings/` or configure `MINING_SOURCE_DIR` in both notebooks; missing/mismatched originals fail explicitly. This also applies to comparison's training and validation replays independently.

Comparison recalculates displayed duration-dependent rates for older exports without changing saved thresholds or event counts, and warns that evaluation must be rerun for threshold reselection. **Alarms are still counted on crops, not the entire originals: source-duration-normalized crop FP/min is not a measured full-source or normal-use false-alarm rate.**

### Full Normal-Use Evaluation

Both `evaluate.ipynb` and `compare.ipynb` also evaluate complete `normal_usage` originals in a **separate population**. Configure `NORMAL_USAGE_SOURCE_DIR` (default: the mining source folder) and `NORMAL_USAGE_FILES` (`None`: all originals, `[]`: disabled, or an explicit basename list). Full originals keep report type `normal_usage`; their mined crops are `hard_negative`. Full-source inference carries CNN/recurrent state across source sessions, with chunked computation and reset only between files. Source copies are deduplicated by hash, and zero-alarm files still contribute their duration.

`normal_usage_fp_per_min = all emitted alarms on complete normal-use originals / their actual total minutes`. Set `FP_SELECTION_METRIC = 'normal_usage_fp_per_min'` in evaluation to use this rate in the existing selectors. Event recall/F1 remain measured on labeled validation data; no full-normal-use alarms/minutes are added to crop diagnostics. Without originals the rate is undefined, not zero, so normal-use budget selection is infeasible. Evaluation exports `normal_usage_metrics.csv`, `normal_usage_recording_metrics.csv`, `normal_usage_alarm_details.csv`, and `normal_usage_manifest.csv`, plus population provenance in `evaluation_context.json`. Comparison replays the selected checkpoints on the current full-normal-use population at frozen thresholds, including older exports, and warns if the population differs from export.

Keep original CSVs available; source lengths alone cannot replace full inference. For held-out estimates use a dedicated source folder/list whose originals **and crops** were excluded from training and threshold selection. The default source folder is not automatically held out.

## Compare Evaluated Runs

[`compare_models.ipynb`](compare_models.ipynb) reads completed `threshold_comparison/` exports from multiple runs.

Edit:

- `RUNS`: names and paths to each run's export directory;
- `SELECTION_MODE`: for example, `max_f1`;
- `POLICY`: for example, `lookahead_12f`.

The notebook shows operating points, window/frame/event metrics, per-type results, per-recording behavior, and saved-checkpoint examples. It does not retrain or change thresholds. It rejects incompatible exports instead of silently comparing different datasets or rules.

## Mine Hard Negatives

The GRU, raw LSTM, and feature LSTM training notebooks expose `HARD_NEGATIVE_MULTIPLIER = 5` in their settings. Negative windows from filenames containing `negative_mining` (including mined crops) appear that many times per epoch; all other training windows appear once. Set it to `1` to disable. This repetition is additional to the ordinary `neg_window_weight` loss multiplier. Validation and unique-window feature normalization remain unchanged. Each run saves `training_sampling.csv`, and the multiplier is included in checkpoint training configuration. Data paths are not changed: select a dataset containing your mined recordings for the setting to have an effect.

For Python training entry points use `cfg.training.hard_negative_multiplier = 5`; the underlying `TrainingConfig` default is `1` to retain legacy behavior. The CLI supports `python train.py --hard-negative-multiplier 5`. Filename selection checks both `negative_mining` and the window's negative class, so positive windows are never repeated by this rule.

[`mine_hard_negatives.ipynb`](mine_hard_negatives.ipynb) mines candidates from the union of all configured checkpoints: each model applies its own side thresholds, followed by one shared lookahead/cooldown/gap/overlap decision. Its single-source review plots IMU and all causal left/right model probabilities over the full recording and uses this exact union—never a separate single-model candidate list.

The notebook stages inputs under `hard_negative_mining/source_recordings/`, previews tunable hard intervals and surrounding context, and exports explicitly selected candidates under `hard_negative_mining/crops/`. Each raw crop is one session with an empty `.txt` negative label and source/checkpoint provenance. Copy the chosen CSV/label pairs manually into the appropriate training folder. See [`../hard_negative_mining/README.md`](../hard_negative_mining/README.md) for crop fitting details.

The **batch mining** section accepts full `normal_usage`, `negative_mining`, and hard-negative originals. Select `BATCH_FILES` or enable `BATCH_SELECT_ALL`; choose `BATCH_MODELS` and each model's left/right thresholds in `MODELS`. It mines the union of alarms from at least one model, applies `BATCH_MIN_GAP_FRAMES` across that union, and prohibits overlapping/slid duplicate crops by default. Three-second crop duration uses each source's sample rate. Per-file and overall tables show crop count, crops/minute, total source time, and summed crop-segment time; zero-crop sources are included. To inspect one file, set `REVIEW_FILE`: its table, full recording plot, candidate zoom, selection IDs, and export all use exactly the candidates counted for that file. Set `REVIEW_KEEP_IDS` and enable `SAVE_REVIEWED_CROPS` only when ready to write. Exported files remain browsable in the crops inventory.

Copy **CSV + empty `.txt` + `.mining.json`** together. Normal-use crops retain `normal_usage` in their filenames and use action `negative_mining`; they are hard negatives in reports and in the configurable training repetition. Full normal-use files are not repeated by that filename rule. New provenance also records original frames, sample rate, duration, source type, and all batch model hashes/thresholds.

## Historical Notebooks

| Notebook | Purpose |
| --- | --- |
| [`compare_8frames.ipynb`](compare_8frames.ipynb) | Earlier saved-run comparison focused on delayed-target experiments. |
| [`compare_GRUvsLSTM.ipynb`](compare_GRUvsLSTM.ipynb) | Earlier GRU/LSTM comparison with example inspection. |

Use `compare_models.ipynb` for new comparisons. Historical notebooks remain useful for understanding the experiment evolution but should not replace the canonical evaluation/export flow.

## Related Documentation

- [`../docs/training-and-evaluation.md`](../docs/training-and-evaluation.md): controlled experiment rules, exported files, thresholds, and external testing.
- [`../docs/data-collection-and-labeling.md`](../docs/data-collection-and-labeling.md): recording layout and strict auto-labeling window.
- [`../notebooks_analysis/README.md`](../notebooks_analysis/README.md): read-only data, label, and state-analysis notebooks.
