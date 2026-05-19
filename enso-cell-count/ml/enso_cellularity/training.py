"""Training loop utilities for EnsoCellularity."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.utils.data

from enso_cellularity.losses import EnsoCellularityCompositeLoss
from enso_cellularity.metrics import MetricAccumulator

_log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    tiles_per_slide: int = 8192
    eval_tiles_per_slide: int = 0
    eval_tile_chunk_size: int = 8192
    slide_batch_size: int = 1
    lr: float = 3e-5
    weight_decay: float = 3e-3
    max_epochs: int = 80
    patience: int = 5
    min_delta: float = 0.0
    early_stop_metric: str = "val_mae_count"
    num_workers: int = 4
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    scheduler_patience: int = 1
    seed: int = 42


class EarlyStopping:
    """Patience-based early stopping on a validation metric."""

    def __init__(self, patience: int = 12, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.counter = 0

    def __call__(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_adamw_with_decay_exclusions(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Apply weight decay only to matrix-like weights, not norms/biases."""

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    _log.info(
        "AdamW param groups: decay=%d no_decay=%d weight_decay=%.2e lr=%.2e",
        len(decay_params),
        len(no_decay_params),
        weight_decay,
        lr,
    )
    return torch.optim.AdamW(groups, lr=lr)


def move_batch(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: EnsoCellularityCompositeLoss,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: str | torch.device,
    train: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = 1.0,
    log_every: int = 20,
) -> dict[str, float]:
    """Train or evaluate for one epoch."""

    if train:
        model.train()
    else:
        model.eval()

    use_amp = bool(scaler is not None and scaler.is_enabled())
    grad_accum_steps = max(1, int(grad_accum_steps))
    totals: dict[str, float] = {}
    total_samples = 0
    metrics = MetricAccumulator.empty()
    t0 = time.time()
    n_batches = len(loader)

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, raw_batch in enumerate(loader):
            batch = move_batch(raw_batch, device)
            amp_context = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if use_amp
                else nullcontext()
            )
            with amp_context:
                outputs = model.forward_outputs(
                    batch["x9"],
                    batch["valid9"],
                    batch["metadata"],
                    batch["exposure_mm2"],
                )
                loss, parts = criterion(
                    outputs,
                    batch["y_count"],
                    teacher_confidence=batch.get("teacher_confidence"),
                    quality_target=batch.get("quality_target"),
                )

            if train and optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss / grad_accum_steps).backward()
                    should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (
                        batch_idx + 1 == n_batches
                    )
                    if should_step:
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    (loss / grad_accum_steps).backward()
                    should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (
                        batch_idx + 1 == n_batches
                    )
                    if should_step:
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            n = int(batch["y_count"].numel())
            total_samples += n
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * n
            for name, value in parts.items():
                if name == "loss":
                    continue
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * n
            metrics.update(
                y_true=batch["y_count"],
                outputs=outputs,
                true_bin=batch.get("count_bin"),
            )

            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                running = totals["loss"] / max(total_samples, 1)
                _log.info(
                    "  [%s] batch %d/%d loss=%.4f samples=%d elapsed=%.1fs",
                    "train" if train else "val",
                    batch_idx + 1,
                    n_batches,
                    running,
                    total_samples,
                    time.time() - t0,
                )

    out = {name: value / max(total_samples, 1) for name, value in totals.items()}
    out.update(metrics.compute())
    out["samples"] = float(total_samples)
    out["elapsed_s"] = time.time() - t0
    return out


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    model_config: Any,
    train_config: TrainConfig,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint with enough metadata for inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else model_config,
        "train_config": asdict(train_config),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
