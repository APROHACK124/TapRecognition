"""Training script for causal CNN + GRU double-tap detector."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from tap_recognition.config import EarlyStoppingConfig, TrainConfig
from tap_recognition.dataset import IMUDoubleTapDataset, RecordedIMUDataset
from tap_recognition.model import CausalCNNGRU


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_datasets(cfg: TrainConfig) -> tuple[Dataset, Dataset]:
    data = cfg.data
    labels = cfg.labels.to_label_params()

    if data.mode == "recorded":
        train_ds = RecordedIMUDataset(
            data.train_dir,
            window_samples=data.window_samples,
            label_params=labels,
            highpass=data.highpass,
            negative_windows_per_file=data.negative_windows_per_file,
            trigger_context_samples=data.trigger_context_samples,
            exclusion_margin=data.exclusion_margin,
            dt_min=data.dt_min,
            dt_max=data.dt_max,
            seed=cfg.seed,
        )
        val_ds = RecordedIMUDataset(
            data.val_dir,
            window_samples=data.window_samples,
            label_params=labels,
            highpass=data.highpass,
            negative_windows_per_file=data.negative_windows_per_file,
            trigger_context_samples=data.trigger_context_samples,
            exclusion_margin=data.exclusion_margin,
            dt_min=data.dt_min,
            dt_max=data.dt_max,
            seed=cfg.seed + 1,
        )
        return train_ds, val_ds

    full_ds = IMUDoubleTapDataset(
        num_samples=data.synthetic_samples,
        window_samples=data.window_samples,
        sample_rate=data.sample_rate,
        seed=cfg.seed,
        highpass=data.highpass,
    )
    n_val = max(1, int(data.val_split * len(full_ds)))
    n_train = len(full_ds) - n_val
    return random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )


def train_one_epoch(
    model: CausalCNNGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    threshold: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct_frames = 0
    total_frames = 0
    correct_windows = 0
    total_windows = 0

    for batch in loader:
        imu = batch["imu"].to(device)
        frame_labels = batch["frame_labels"].to(device)

        logits, _ = model(imu)
        probs = torch.sigmoid(logits.squeeze(-1))
        loss = F.binary_cross_entropy(probs, frame_labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * imu.size(0)
        pred = (probs > threshold).float()
        correct_frames += (pred == frame_labels).sum().item()
        total_frames += frame_labels.numel()

        window_pred = (probs.max(dim=1).values > threshold).float()
        correct_windows += (window_pred == batch["window_label"].to(device)).sum().item()
        total_windows += imu.size(0)

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "frame_acc": correct_frames / total_frames,
        "window_acc": correct_windows / total_windows,
    }


@torch.no_grad()
def evaluate(
    model: CausalCNNGRU,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct_frames = 0
    total_frames = 0
    correct_windows = 0
    total_windows = 0
    tp = fp = fn = tn = 0

    for batch in loader:
        imu = batch["imu"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        window_labels = batch["window_label"].to(device)

        logits, _ = model(imu)
        probs = torch.sigmoid(logits.squeeze(-1))
        loss = F.binary_cross_entropy(probs, frame_labels)
        total_loss += loss.item() * imu.size(0)

        pred = (probs > threshold).float()
        correct_frames += (pred == frame_labels).sum().item()
        total_frames += frame_labels.numel()

        window_pred = (probs.max(dim=1).values > threshold).float()
        correct_windows += (window_pred == window_labels).sum().item()
        total_windows += imu.size(0)

        for wp, wl in zip(window_pred, window_labels):
            if wp == 1 and wl == 1:
                tp += 1
            elif wp == 1 and wl == 0:
                fp += 1
            elif wp == 0 and wl == 1:
                fn += 1
            else:
                tn += 1

    n = len(loader.dataset)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "loss": total_loss / n,
        "frame_acc": correct_frames / total_frames,
        "window_acc": correct_windows / total_windows,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class EarlyStopping:
    """Stop training when the monitored validation metric stops improving."""

    def __init__(self, config: EarlyStoppingConfig):
        self.enabled = config.enabled
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.monitor = config.monitor
        if self.monitor not in {"val_window_acc", "val_loss"}:
            raise ValueError(
                f"Unsupported early_stopping.monitor: {self.monitor!r}"
            )
        self.mode = "max" if self.monitor == "val_window_acc" else "min"
        self.metric_key = "window_acc" if self.monitor == "val_window_acc" else "loss"
        self.best: float | None = None
        self.counter = 0

    def step(self, metrics: dict[str, float]) -> bool:
        if not self.enabled:
            return False

        current = metrics[self.metric_key]
        if self.best is None:
            self.best = current
            return False

        if self.mode == "max":
            improved = current > self.best + self.min_delta
        else:
            improved = current < self.best - self.min_delta

        if improved:
            self.best = current
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def train(cfg: TrainConfig) -> float:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    print(f"Loading {cfg.data.mode} datasets...")
    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model_kwargs = cfg.model.to_model_kwargs()
    model = CausalCNNGRU(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    early_stop = EarlyStopping(cfg.training.early_stopping)
    stopped_epoch = cfg.training.epochs

    es = cfg.training.early_stopping
    print(f"Training on {device}, {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"Model receptive field: {model.receptive_field} samples")
    if es.enabled:
        print(
            f"Early stopping: monitor={es.monitor}, patience={es.patience}, "
            f"min_delta={es.min_delta}"
        )

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg.training.grad_clip,
            cfg.training.prediction_threshold,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            cfg.training.prediction_threshold,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{cfg.training.epochs} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['window_acc']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['window_acc']:.3f} "
            f"P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f}"
        )

        if val_metrics["window_acc"] > best_val_acc:
            best_val_acc = val_metrics["window_acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_kwargs,
                    "train_config": cfg.to_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                out_dir / "best.pt",
            )

        if early_stop.step(val_metrics):
            stopped_epoch = epoch
            print(
                f"Early stopping at epoch {epoch} "
                f"(no {es.monitor} improvement for {es.patience} epochs)"
            )
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_kwargs,
            "train_config": cfg.to_dict(),
            "epoch": stopped_epoch,
        },
        out_dir / "last.pt",
    )
    print(f"Done. Best val window acc: {best_val_acc:.3f} (stopped at epoch {stopped_epoch})")
    print(f"Checkpoints saved to {out_dir}/")
    return best_val_acc


def main() -> None:
    train(TrainConfig())


if __name__ == "__main__":
    main()
