"""Compact, reproducible baseline optimisers used by ``pipeline.distill``."""

from __future__ import annotations

from typing import Callable

import torch

from .config import ExperimentConfig
from .data import normalized_bounds


def total_variation(images: torch.Tensor) -> torch.Tensor:
    return (images[:, :, 1:] - images[:, :, :-1]).abs().mean() + (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()


def _pixel_match(
    cfg: ExperimentConfig,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    aux_loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """A deterministic low-cost baseline with explicit classwise pixel moments.

    This is intentionally a baseline, not a claim to reproduce historical MTT
    results.  It keeps every legacy method name executable while the benchmark
    ranks imported artifacts rather than treating this optimiser as SOTA.
    """
    optim_images = images.detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam([optim_images], lr=cfg.distill.lr)
    low, high = normalized_bounds(cfg.dataset.name, device)
    target = torch.stack([optim_images.detach()[labels == cls].mean(0) for cls in torch.unique(labels)])
    history: list[dict[str, float]] = []
    for step in range(cfg.distill.steps):
        means = torch.stack([optim_images[labels == cls].mean(0) for cls in torch.unique(labels)])
        loss = torch.nn.functional.mse_loss(means, target)
        if aux_loss_fn is not None:
            loss = loss + aux_loss_fn(optim_images)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            optim_images.clamp_(low, high)
        if step % cfg.distill.log_every == 0 or step == cfg.distill.steps - 1:
            history.append({"step": float(step), "loss": float(loss.detach())})
    return optim_images.detach(), history


def gradient_matching(cfg: ExperimentConfig, images: torch.Tensor, labels: torch.Tensor, device: torch.device):
    return _pixel_match(cfg, images, labels, device)


def trajectory_matching(
    cfg: ExperimentConfig,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    aux_loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    return _pixel_match(cfg, images, labels, device, aux_loss_fn)


def diffusion_synthetic(cfg: ExperimentConfig, model, labels: torch.Tensor, images: torch.Tensor, device: torch.device):
    del model
    return _pixel_match(cfg, images, labels, device, total_variation)
