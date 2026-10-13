"""Training-only repetition of filename-selected hard-negative windows."""

from __future__ import annotations

from bisect import bisect_right

import pandas as pd
from torch.utils.data import ConcatDataset, Dataset, Subset

from .dataset import RecordedIMUDataset


def _window_source(dataset: Dataset, index: int) -> tuple[str, int]:
    """Recover window provenance through explicit splits and multiple folders."""
    if isinstance(dataset, Subset):
        return _window_source(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        part = bisect_right(dataset.cumulative_sizes, index)
        offset = dataset.cumulative_sizes[part - 1] if part else 0
        return _window_source(dataset.datasets[part], index - offset)
    if isinstance(dataset, RecordedIMUDataset):
        return dataset.window_meta[index].source_file, int(dataset.samples[index][2])
    return "", int(dataset[index]["window_label"])


def repeat_hard_negatives(
    dataset: Dataset, multiplier: int = 1,
) -> tuple[Dataset, pd.DataFrame]:
    """Keep all windows once; repeat negative_mining negatives to N total copies.

    Pass only the training split. The original dataset and its metadata are
    unchanged, so evaluation and feature normalization can use unique windows.
    Counts are exact per epoch, unlike weighted sampling with replacement.
    """
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError("hard_negative_multiplier must be an integer >= 1")
    rows, hard_indices = [], []
    for index in range(len(dataset)):
        source, class_id = _window_source(dataset, index)
        hard_negative = class_id == 0 and "negative_mining" in source.lower()
        if hard_negative:
            hard_indices.append(index)
        rows.append(dict(window_index=index, source_file=source, window_class=class_id,
                         hard_negative=hard_negative,
                         appearances_per_epoch=multiplier if hard_negative else 1))
    table = pd.DataFrame(rows, columns=[
        "window_index", "source_file", "window_class", "hard_negative", "appearances_per_epoch",
    ])
    extra_indices = hard_indices * (multiplier - 1)
    sampled = ConcatDataset([dataset, Subset(dataset, extra_indices)]) if extra_indices else dataset
    return sampled, table


def print_training_sampling(table: pd.DataFrame, multiplier: int) -> None:
    hard_count = int(table.hard_negative.sum())
    epoch_count = int(table.appearances_per_epoch.sum())
    share = hard_count * multiplier / epoch_count if epoch_count else 0.0
    print(f"Training windows: {len(table)} unique; {epoch_count} appearances per epoch")
    print(f"negative_mining negatives: {hard_count} x {multiplier}; "
          f"{share:.1%} of training epoch. Other windows x 1; validation unchanged.")
    if multiplier > 1 and not hard_count:
        print("No negative training windows with 'negative_mining' in their filename; "
              "the multiplier has no effect on this split.")
