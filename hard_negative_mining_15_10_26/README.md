# Isolated mining workspace — 15 October 2026

Use `main_notebooks/mine_hard_negatives.ipynb` with the project Python kernel.
This workspace contains only the new negative recordings moved from `temp/`:
4 `negative_mining` originals and 30 `normal_usage` originals.
The previous `hard_negative_mining/` sources and crops are not included or changed.

- `source_recordings/`: full raw originals, retained for crop provenance and full-source replay.
- `crops/`: newly exported CSV + empty `.txt` + `.mining.json` triplets. Keep each triplet together when moving it into a dataset split.
- Models: define `MODELS` in the notebook; no old models are enabled automatically.
- Nothing is exported until `SAVE_ALL_BATCH_CROPS=True` or `SAVE_REVIEWED_CROPS=True`.

## Separate recording types and cropping styles

In the batch cell, choose one of:

```python
BATCH_SOURCE_TYPES = ['negative_mining']
BATCH_SOURCE_TYPES = ['normal_usage']
BATCH_SOURCE_TYPES = ['negative_mining', 'normal_usage']
```

The type filter applies to automatic selection and explicit `BATCH_FILES`.
To use different cropping styles, mine/export one type, change the type and
the crop/context/gap/cooldown/overlap settings, then rerun the batch cell.
Exports retain the settings in provenance; changing the filter does not remove prior exports.
The review section always shows the exact all-model union from the current batch.

## Preserved previous setup

`main_notebooks/mine_hard_negatives_before_15_10_26.ipynb` preserves the previous
notebook, model list, source/output paths, settings, and outputs.

The two new strict double-tap recordings are **not** mining inputs. They were
auto-labeled with `--strict-spike-window-ms 1900 2700` and moved, with sparse
and dense labels, to `data_15_10_26/valid_data/` (20 events per recording, zero misses).

When evaluating crops from this workspace later, configure the evaluator's
`MINING_SOURCE_DIR` (and `NORMAL_USAGE_SOURCE_DIR` for full normal-use replay)
to this workspace's `source_recordings/`. Do not merge full-source rates with
crop-based diagnostics or assume sources represented by training crops are held out.
