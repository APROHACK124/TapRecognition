"""Separate training entry point for feature-engineered LSTM experiments."""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig
from .imu_features import IMUFeatureConfig
from .model_factory import build_model
from train import (
    EarlyStopping, _augment_positive_windows, _recorded_dataset,
    evaluate, frame_soft_ce, resolve_device, window_class_from_probs,
)


class DelayedTargets(Dataset):
    """Shift frame targets on retrieval without changing annotations or the source dataset."""

    def __init__(self, dataset, delay: int):
        if type(delay) is not int or delay < 0:
            raise ValueError("label_delay_frames must be a non-negative integer")
        self.dataset, self.delay = dataset, delay

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        target = item["frame_labels"]
        if self.delay >= len(target):
            raise ValueError("Label delay must be smaller than the window length")
        if self.delay:
            shifted = torch.zeros_like(target)
            shifted[self.delay:, 1:] = target[:-self.delay, 1:]
            shifted[:, 0] = 1 - shifted[:, 1:].sum(dim=-1)
            item["frame_labels"] = shifted
        return item


def build_feature_datasets(cfg: TrainConfig):
    """Load explicit recorded splits without the baseline's automatic train/val fallback."""
    if cfg.data.mode != "recorded":
        raise ValueError("The feature notebook requires an explicit recorded train/validation split")
    train_dirs = [p.strip() for p in cfg.data.train_dir.split(",") if p.strip()]
    val_dirs = [p.strip() for p in cfg.data.val_dir.split(",") if p.strip()]
    if not train_dirs or not val_dirs:
        raise ValueError("Both train and validation directories are required")
    if {Path(p).resolve() for p in train_dirs} & {Path(p).resolve() for p in val_dirs}:
        raise ValueError("Train and validation directories must be separate")
    train_ds = _recorded_dataset(train_dirs, cfg, cfg.seed)
    val_ds = _recorded_dataset(val_dirs, cfg, cfg.seed + 1)
    if not any(int(item["window_label"]) > 0 for item in val_ds):
        raise ValueError("Validation has no positive windows; prepare a labeled validation split")
    return train_ds, val_ds


def _train_epoch(model, loader, optimizer, device, cfg, amplitude_augmentation):
    model.train()
    total_loss = correct_frames = total_frames = correct_windows = total_windows = 0
    for batch in loader:
        imu = batch["imu"].to(device)
        labels = batch["frame_labels"].to(device)
        window = batch["window_label"].to(device)
        if amplitude_augmentation:
            # Recompute ALL nonlinear engineered features from augmented raw axes.
            imu = _augment_positive_windows(imu, window)
        logits, _ = model(imu)
        loss = frame_soft_ce(logits, labels, cfg.event_loss_weight,
                             window_label=window, neg_window_weight=cfg.neg_window_weight)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        probs = logits.detach().softmax(-1)
        total_loss += loss.item() * len(imu)
        correct_frames += int((probs.argmax(-1) == labels.argmax(-1)).sum())
        total_frames += labels.shape[0] * labels.shape[1]
        correct_windows += int((window_class_from_probs(probs, cfg.prediction_threshold) == window).sum())
        total_windows += len(imu)
    return dict(loss=total_loss / total_windows, frame_acc=correct_frames / total_frames,
                window_acc=correct_windows / total_windows)


def _manifest(cfg):
    paths = set()
    for value in (cfg.data.train_dir, cfg.data.val_dir):
        for folder in value.split(","):
            for path in Path(folder.strip()).glob("*.csv"):
                if not path.name.endswith(".labels.csv"):
                    paths.add(path)
                    paths.add(path.with_suffix(".txt"))
    if cfg.data.session_exclusions_file:
        paths.add(Path(cfg.data.session_exclusions_file))
    return [dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(paths) if path.is_file()]


def train_feature_lstm(
    cfg: TrainConfig,
    feature_config: IMUFeatureConfig,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    label_delay_frames: int = 10,
    amplitude_augmentation: bool = True,
):
    """Train from scratch in a new timestamped directory; return (run_dir, history).

    Config, datasets, recordings and existing runs are never modified. The
    baseline loss/optimizer/scheduler and window-based selection are retained.
    Normalization is fitted once on unaugmented training windows only.
    """
    cfg = copy.deepcopy(cfg)
    if cfg.init_checkpoint or cfg.freeze_except_last:
        raise ValueError("This feature experiment trains from scratch; initialization/freezing are unsupported")
    if cfg.training.epochs < 1 or not len(train_ds) or not len(val_ds):
        raise ValueError("Training requires positive epochs and nonempty train/validation datasets")
    if cfg.model.input_dim != 6:
        raise ValueError("Pass six IMU channels; feature engineering happens inside the model")
    if not 0 <= label_delay_frames < cfg.data.window_samples:
        raise ValueError("Label delay must be within the window length")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model_kwargs = cfg.model.to_model_kwargs() | {"feature_config": feature_config.to_dict()}
    model = build_model(model_kwargs, "feature_lstm")
    stats_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=False,
                              num_workers=cfg.training.num_workers)
    normalization_frames = model.input_proj.fit_normalization(stats_loader)
    model = model.to(device)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(DelayedTargets(train_ds, label_delay_frames),
                              batch_size=cfg.training.batch_size, shuffle=True,
                              generator=generator, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(DelayedTargets(val_ds, label_delay_frames),
                            batch_size=cfg.training.batch_size, shuffle=False,
                            num_workers=cfg.training.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                   weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    early_stop = EarlyStopping(cfg.training.early_stopping)
    run_dir = Path(cfg.out_dir) / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.out_dir = str(run_dir)
    training_config = cfg.to_dict()
    training_config["model"] = model_kwargs
    metadata = dict(
        model_type="feature_lstm", model_config=model_kwargs, train_config=training_config,
        positive_label_delay_frames=label_delay_frames, amplitude_augmentation=amplitude_augmentation,
        feature_names=feature_config.names, normalization_fit_split="train",
        normalization_frames=normalization_frames,
    )
    (run_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "data_manifest.json").write_text(json.dumps(_manifest(cfg), indent=2) + "\n")

    def save_checkpoint(name, epoch, metrics=None):
        snapshot = metadata | dict(model_state=model.state_dict(), epoch=epoch)
        if metrics is not None:
            snapshot["val_metrics"] = metrics
        temporary = run_dir / f"{name}.tmp"
        torch.save(snapshot, temporary)
        temporary.replace(run_dir / name)

    save_checkpoint("initial.pt", 0)
    history, best_score = [], None
    print(f"Run: {run_dir}\nDevice: {device}; {len(feature_config.names)} input features; "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = _train_epoch(model, train_loader, optimizer, device,
                                     cfg.training, amplitude_augmentation)
        val_metrics = evaluate(model, val_loader, device, cfg.training.prediction_threshold,
                                cfg.training.event_loss_weight, cfg.training.neg_window_weight)
        if not np.isfinite(list(val_metrics.values())).all():
            raise ValueError("Non-finite validation metrics")
        score = val_metrics[early_stop.metric_key]
        improved = (best_score is None or (score > best_score if early_stop.mode == "max" else score < best_score))
        if improved:
            best_score = score
            save_checkpoint("best.pt", epoch, val_metrics)
        save_checkpoint("last.pt", epoch, val_metrics)
        history.append(dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"],
                            **{f"train_{k}": v for k, v in train_metrics.items()},
                            **{f"val_{k}": v for k, v in val_metrics.items()}))
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{cfg.training.epochs} | train loss={train_metrics['loss']:.4f} "
              f"acc={train_metrics['window_acc']:.3f} | val loss={val_metrics['loss']:.4f} "
              f"acc={val_metrics['window_acc']:.3f} P={val_metrics['precision']:.3f} "
              f"R={val_metrics['recall']:.3f}")
        if early_stop.step(val_metrics):
            print(f"Early stopping at epoch {epoch}")
            break
    return run_dir, pd.DataFrame(history)
