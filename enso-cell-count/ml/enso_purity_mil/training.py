"""Training utilities: epoch runner, early stopping, config."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

_log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    num_instances: int = 4096
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 0.0
    num_workers: int = 14
    seed: int = 42


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _unpack_batch(
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if len(batch) == 3:
        feats, labels, is_tumor = batch
        return feats, labels, is_tumor
    feats, labels = batch
    return feats, labels, None


def _beta_nll(labels: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    labels = labels.clamp(min=1e-4, max=1.0 - 1e-4)
    dist = torch.distributions.Beta(alpha.clamp_min(1e-4), beta.clamp_min(1e-4))
    return -dist.log_prob(labels).mean()


def _proto_entropy_penalty(proto_hist: torch.Tensor) -> torch.Tensor:
    # Minimize negative entropy -> discourages collapsed assignments.
    eps = 1e-8
    entropy = -(proto_hist * torch.log(proto_hist + eps)).sum(dim=1).mean()
    return -entropy


def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    *,
    device: str = "cpu",
    train: bool = True,
    log_every: int = 20,
    max_log_lines: Optional[int] = None,
    aux_bce_weight: float = 0.1,
    beta_nll_weight: float = 0.02,
    proto_entropy_weight: float = 0.01,
    consistency_weight: float = 0.05,
    grad_accum_steps: int = 1,
) -> Union[float, dict]:
    """Run a single training or validation epoch.

    Main tracked loss is L1 on purity mean (mu), matching prior behavior.
    """
    if criterion is None:
        criterion = nn.L1Loss()

    if train:
        model.train()
    else:
        model.eval()

    grad_accum_steps = max(1, int(grad_accum_steps))

    total_objective = 0.0
    total_l1 = 0.0
    n_samples = 0
    n_batches = len(loader)
    if max_log_lines is not None:
        if max_log_lines <= 0:
            log_every = 0
        else:
            log_every = max(1, (n_batches + max_log_lines - 1) // max_log_lines)
    t_epoch = time.time()

    all_preds: list[float] = []
    all_labels: list[float] = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loader):
            feats, labels, is_tumor = _unpack_batch(batch)
            feats = feats.to(device)
            labels = labels.to(device)
            if is_tumor is not None:
                is_tumor = is_tumor.to(device=device, dtype=torch.float32)

            if hasattr(model, "forward_outputs"):
                out = model.forward_outputs(feats)
                mu = out["mu"].squeeze(-1)
            else:
                mu = model(feats).squeeze(-1)
                out = {}

            l1_loss = criterion(mu, labels)
            loss = l1_loss

            if out and is_tumor is not None and aux_bce_weight > 0.0:
                aux_bce = F.binary_cross_entropy(
                    out["tumor_prob"].squeeze(-1), is_tumor
                )
                loss = loss + aux_bce_weight * aux_bce

            if out and beta_nll_weight > 0.0 and "alpha" in out and "beta" in out:
                beta_nll = _beta_nll(labels, out["alpha"].squeeze(-1), out["beta"].squeeze(-1))
                loss = loss + beta_nll_weight * beta_nll

            if out and proto_entropy_weight > 0.0 and "proto_hist" in out:
                proto_pen = _proto_entropy_penalty(out["proto_hist"])
                loss = loss + proto_entropy_weight * proto_pen

            if train and out and consistency_weight > 0.0 and hasattr(model, "forward_outputs"):
                out_2 = model.forward_outputs(feats)
                consistency = F.l1_loss(out["mu"], out_2["mu"].detach())
                loss = loss + consistency_weight * consistency

            if train and optimizer is not None:
                (loss / grad_accum_steps).backward()
                should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == n_batches)
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_objective += loss.item() * labels.size(0)
            total_l1 += l1_loss.item() * labels.size(0)
            n_samples += labels.size(0)

            if not train:
                all_preds.extend(mu.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                avg_obj = total_objective / max(n_samples, 1)
                avg_l1 = total_l1 / max(n_samples, 1)
                elapsed = time.time() - t_epoch
                mode = "train" if train else "val"
                _log.info(
                    "  [%s] batch %d/%d  running_loss=%.4f  running_l1=%.4f  (%.1fs)",
                    mode, batch_idx + 1, n_batches, avg_obj, avg_l1, elapsed,
                )

    avg_l1 = total_l1 / max(n_samples, 1)
    if train:
        return avg_l1

    r2 = float("nan")
    spearman = float("nan")
    try:
        from sklearn.metrics import r2_score
        from scipy.stats import spearmanr

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        r2 = float(r2_score(y_true, y_pred))
        sp_corr, _ = spearmanr(y_true, y_pred)
        spearman = float(sp_corr)
    except Exception:
        pass

    return {"loss": avg_l1, "r2": r2, "spearman": spearman}
