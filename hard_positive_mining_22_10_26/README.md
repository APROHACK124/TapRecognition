# Hard-positive mining — 22 October 2026

Open `main_notebooks/mine_hard_positives.ipynb` with the project Python kernel.
`source_recordings/` contains 24 new strict double-tap originals staged from
`temp/`, including recordings collected on 23 October. The originals remain in
`temp/` too. They have **480 auto-labeled taps, zero misses**, using
`tools/auto_label_imu.py --strict-spike-window-ms 1950 2750` (10-frame soft-label
half-width). `labeling_report.csv` and each `.labeling.json` record the settings,
raw/label hashes, segment counts, events and misses.

## Selection rule

1. By default each segment is independently high-pass preprocessed (if required
   by its checkpoint) and inferred with fresh model state. Set
   `RESET_AT_SEGMENTS=False` explicitly to inspect continuous-source behavior.
2. For each unambiguously labeled segment and each selected model, take the
   maximum **correct-side** probability in the inclusive **-500 ms to +500 ms**
   window around the auto-labeled second tap, clipped to that segment.
3. If **any model** is strictly below its own side threshold, keep that segment.
   Threshold equality counts as easy. A high wrong-side peak does not count.
4. Easy segments are not selected. Unlabeled or multiply labeled segments are
   audited separately and never become hard-positive exports or negatives.

This is confidence mining, not matched-event recall or a cooldown/latency score.
The clipped score window may end at the segment boundary before +500 ms;
`window_clipped` and the exact frame bounds make that explicit.

## Review and export

- Define `MODELS` and their left/right confidence thresholds, independently of
  negative mining. Select source files, sides and model subsets in the batch cell.
- `hard_candidates` is the exact any-model union; `all_segments` includes the
  excluded easy/unlabeled/ambiguous segments. `model_scores` shows every model's
  correct-side peak and threshold per labeled segment. File stats include
  all-easy recordings.
- The source review cell shows per-segment scores. The separate detailed-review
  cell accepts a global `DETAIL_CANDIDATE_ID` and displays all raw sensor/model
  traces, label, score window, peak scores and thresholds for just that segment.
- Enable `SAVE_ALL_HARD_POSITIVES` for all hard segments, or explicitly choose
  `POSITIVE_KEEP_IDS` and enable `SAVE_SELECTED_HARD_POSITIVES`. Both are off by
  default. No mining/export is performed automatically during workspace setup.
- Each export is the **whole original segment**, normally 3 s: raw CSV with
  timestamps restarting at zero and `segment_index=1`, nonempty `.txt` with its
  remapped positive event, and `.mining.json` with hashes, all models/scores,
  scoring/reset settings and strict-label provenance. The action/side stays
  positive. The existing training loader builds dense labels from the `.txt`.
- Identical exports are reusable; conflicting or incomplete outputs are rejected.
  All triplets must stay together. Changing source labels requires remining.

The crops stay in this workspace until you choose how to add them to
`data_22_10_26`. Do not also add their full source originals; doing so would
reintroduce easy segments and duplicate the mined positives. Positive crops report
as `knock_twice`, not `hard_negative`, and use their own evaluated duration.
