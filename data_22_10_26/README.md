# Dataset — 22 October 2026

Initialized as a byte-for-byte copy of `data_15_10_26`, preserving all original
train/validation assignments, crop provenance, exclusions and split manifests.
The copied manifests describe inherited data only; they do not assign new crops.

Added 388 newly mined crop triplets (294 positive, 94 negative). All new crops
were sorted together by source collection timestamp, then original frame bounds,
and assigned with a repeating **train, train, train, validation** cycle (75%/25%).
The cycle does not reset at source or positive/negative boundaries.

| New crops | Train | Validation |
|---|---:|---:|
| Negative | 71 | 23 |
| Positive left | 120 | 42 |
| Positive right | 100 | 32 |
| **Total** | **291** | **97** |

Assignments, original frame bounds and all triplet hashes are recorded in
`crop_chronological_split_22_10_26_manifest.csv`. All inherited files were
preserved unchanged. The dataset now contains 1,038 training CSVs and 317
validation CSVs (including 898 and 257 crop CSVs respectively).

This is a crop-level split, not a session-disjoint evaluation: source recordings
can contribute to both splits, and 11 negative-crop interval pairs overlap
across train/validation. The allocation intentionally does not enforce separation.

The new original recordings (16–23 October) remain available separately:

- negatives: `hard_negative_mining_22_10_26/source_recordings/`;
- strict positives: `hard_positive_mining_22_10_26/source_recordings/`, labeled
  with a 1950–2750 ms spike window (480 events, zero misses).

Use `main_notebooks/mine_hard_negatives.ipynb` and
`main_notebooks/mine_hard_positives.ipynb` to select/export difficult examples.
Only add further selected crops after choosing their train/validation allocation;
do not copy all positive originals into training, since that would include easy
segments and duplicate the selected ones. Keep every CSV/label/provenance triplet
together and retain all old/new source folders for provenance verification.
