"""Small, explicit DDPM schedule used by the latent backbone.

The project only needs forward noising and x0 reconstruction for proxy
extraction.  Keeping the schedule here avoids hidden dependencies and makes
the trajectory definition reproducible from the resolved configuration.
"""

from __future__ import annotations

import torch


class DiffusionSchedule:
    """A linear-beta DDPM schedule with batch-safe coefficient extraction."""

    def __init__(
        self,
        steps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device | str,
    ) -> None:
        if steps < 2:
            raise ValueError("diffusion schedule requires at least two steps")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
        self.steps = int(steps)
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = self.alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1.0 - self.alpha_bars).sqrt()

    def _coefficient(self, values: torch.Tensor, t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1 or len(t) != len(reference):
            raise ValueError("t must be a one-dimensional batch-aligned tensor")
        if t.dtype != torch.long:
            t = t.long()
        if int(t.min()) < 0 or int(t.max()) >= self.steps:
            raise ValueError("timestep lies outside schedule")
        coeff = values.to(reference.device)[t.to(reference.device)]
        return coeff.reshape(len(reference), *((1,) * (reference.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample q(x_t | x_0) using supplied noise for reproducibility."""
        if x0.shape != noise.shape:
            raise ValueError("x0 and noise must have identical shapes")
        return self._coefficient(self.sqrt_alpha_bars, t, x0) * x0 + self._coefficient(
            self.sqrt_one_minus_alpha_bars, t, x0
        ) * noise

    def predict_start_from_noise(
        self, xt: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        """Recover x0 from x_t and an epsilon prediction."""
        if xt.shape != predicted_noise.shape:
            raise ValueError("xt and predicted_noise must have identical shapes")
        alpha = self._coefficient(self.sqrt_alpha_bars, t, xt)
        sigma = self._coefficient(self.sqrt_one_minus_alpha_bars, t, xt)
        return (xt - sigma * predicted_noise) / alpha.clamp_min(torch.finfo(xt.dtype).eps)
