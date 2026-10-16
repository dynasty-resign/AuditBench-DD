"""Latent reconstruction trajectories shared by training and audit code."""

from __future__ import annotations

import torch

from .diffusion import DiffusionSchedule
from .models import LatentDiffusion


def trajectory_noise_shape(model: LatentDiffusion, images: torch.Tensor, space: str) -> tuple[int, ...]:
    if space != "latent":
        raise ValueError("the consolidated implementation supports latent trajectories only")
    with torch.no_grad():
        return tuple(model.encode(images[:1]).shape[1:])


def latent_reconstruction_trajectories(
    model: LatentDiffusion,
    images: torch.Tensor,
    labels: torch.Tensor,
    timesteps: list[int],
    schedule: DiffusionSchedule,
    inference_steps: int,
    *,
    source: str = "inversion",
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return noisy latents, predicted-clean latents and the fixed noise.

    ``inversion`` and ``q_sample`` currently share the same deterministic
    forward construction; the distinction is retained in the artifact
    contract for future reverse-solver extensions.  ``inference_steps`` is
    validated so changing it remains an explicit protocol decision.
    """
    if source not in {"inversion", "q_sample"}:
        raise ValueError("source must be inversion or q_sample")
    if inference_steps < 2:
        raise ValueError("inference_steps must be at least two")
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("labels must be batch-aligned")
    if timesteps != sorted(set(timesteps), reverse=True):
        raise ValueError("timesteps must be unique and descending")
    z0 = model.encode(images)
    if noise is None:
        noise = torch.randn_like(z0)
    if noise.shape != z0.shape:
        raise ValueError("noise shape does not match latent shape")
    noisy, predicted = [], []
    for timestep in timesteps:
        t = torch.full((len(images),), timestep, device=images.device, dtype=torch.long)
        zt = schedule.q_sample(z0, t, noise)
        eps = model.denoiser(zt, t, labels)
        noisy.append(zt)
        predicted.append(schedule.predict_start_from_noise(zt, t, eps))
    return torch.stack(noisy, dim=1), torch.stack(predicted, dim=1), noise


def trajectory_embeddings(
    model: LatentDiffusion,
    images: torch.Tensor,
    timesteps: list[int],
    schedule: DiffusionSchedule,
    noise: torch.Tensor,
    *,
    space: str,
    direction: str,
    labels: torch.Tensor,
    conditioning: str,
    source: str,
    representation: str,
    inference_steps: int,
) -> torch.Tensor:
    if space != "latent" or direction != "reverse":
        raise ValueError("the consolidated implementation supports reverse latent trajectories only")
    if conditioning not in {"unconditional", "class"}:
        raise ValueError("conditioning must be unconditional or class")
    if representation not in {"raw", "predicted_clean"}:
        raise ValueError("representation must be raw or predicted_clean")
    denoiser_labels = model.unconditional_labels(len(labels), images.device) if conditioning == "unconditional" else labels
    noisy, predicted, _ = latent_reconstruction_trajectories(
        model, images, denoiser_labels, timesteps, schedule, inference_steps, source=source, noise=noise
    )
    states = noisy if representation == "raw" else predicted
    batch, times = states.shape[:2]
    return model.embed(states.reshape(batch * times, *states.shape[2:])).reshape(batch, times, -1)


def feature_statistics(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [samples, times, features]")
    return embeddings.mean(dim=0), embeddings.var(dim=0, unbiased=False)
