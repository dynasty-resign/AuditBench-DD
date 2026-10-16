"""Differentiable H0 signatures for the distillation objective.

For a finite point cloud, finite H0 deaths equal the edge lengths of its
minimum spanning tree.  The selected tree is piecewise constant in the input,
while the selected edge values retain useful gradients for optimisation.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _pairwise_distance(points: torch.Tensor, metric: str) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError("topology expects a [points, features] tensor")
    if len(points) < 2:
        return points.new_empty((0, 0))
    if metric == "euclidean":
        return torch.cdist(points, points)
    if metric == "cosine":
        normalised = F.normalize(points, dim=-1, eps=1e-8)
        return (1.0 - normalised @ normalised.T).clamp_min(0.0)
    raise ValueError(f"unsupported topology metric: {metric}")


def _prim_deaths(distances: torch.Tensor) -> torch.Tensor:
    """Return sorted MST edge weights using deterministic Prim traversal."""
    count = len(distances)
    if count < 2:
        return distances.new_zeros((0,))
    visited = torch.zeros(count, dtype=torch.bool, device=distances.device)
    visited[0] = True
    best = distances[0].clone()
    best[0] = float("inf")
    deaths = []
    for _ in range(count - 1):
        choice = torch.argmin(best)
        deaths.append(best[choice])
        visited[choice] = True
        best = torch.minimum(best, distances[choice])
        best[visited] = float("inf")
    return torch.sort(torch.stack(deaths)).values


def persistence_signature(
    points: torch.Tensor, signature_size: int, metric: str = "cosine"
) -> torch.Tensor:
    """Fixed-length H0 death signature, padded by zeros when necessary."""
    if signature_size < 1:
        raise ValueError("signature_size must be positive")
    deaths = _prim_deaths(_pairwise_distance(points, metric))
    if len(deaths) >= signature_size:
        return deaths[-signature_size:]
    return F.pad(deaths, (signature_size - len(deaths), 0))


def diagram_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Mean squared distance between same-length H0 signature vectors."""
    if left.shape != right.shape:
        raise ValueError("topology signatures must have the same shape")
    return F.mse_loss(left, right)
