# Hard-negative mining — 22 October 2026

Open `main_notebooks/mine_hard_negatives.ipynb` with the project Python kernel.
Only the 14 new negative originals staged from `temp/` are present here (including
recordings collected on 23 October). Previous mining folders remain untouched.
Original files in `temp/` are retained as well.

- `source_recordings/`: 7 `negative_mining` and 7 `normal_usage` originals.
- `crops/`: future raw CSV + empty `.txt` + `.mining.json` triplets.
- Define `MODELS` with checkpoint paths and per-side thresholds before mining.
- Select `BATCH_SOURCE_TYPES` as `['negative_mining']`, `['normal_usage']`, or both.
  Run/export each type separately if crop/context/gap settings should differ.
- `SAVE_ALL_BATCH_CROPS` and `SAVE_REVIEWED_CROPS` are off by default.
- Outputs are not automatically moved to `data_22_10_26` or assigned to splits.

The prior notebook is preserved byte-for-byte as
`main_notebooks/mine_hard_negatives_before_22_10_26.ipynb`.
Keep full originals available for hash-verified provenance and full-source replay.
When later evaluating new crops, point source-directory settings at this folder;
older crops in the copied dataset retain their original source paths/workspaces.
