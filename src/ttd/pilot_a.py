"""Small geometry primitives retained by the audit pipeline.

The old Pilot-A protocol was removed during repository consolidation.  These
two primitives remain part of the reproducible definition of the audit H0
score and are deliberately dependency-free.
"""

from __future__ import annotations

import numpy as np


def _pairwise(points: np.ndarray, metric: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must have shape [samples, features]")
    if metric == "euclidean":
        squared = np.maximum((points**2).sum(1)[:, None] + (points**2).sum(1)[None] - 2 * points @ points.T, 0)
        return np.sqrt(squared)
    if metric == "cosine":
        normalised = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
        return np.maximum(1.0 - normalised @ normalised.T, 0.0)
    raise ValueError(f"unsupported metric: {metric}")


def _prim_deaths(distances: np.ndarray) -> np.ndarray:
    """Sorted finite H0 deaths (MST edges) for a dense distance matrix."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be square")
    count = len(distances)
    if count < 2:
        return np.empty(0, dtype=float)
    visited = np.zeros(count, dtype=bool)
    visited[0] = True
    best = distances[0].copy()
    best[0] = np.inf
    deaths = np.empty(count - 1, dtype=float)
    for index in range(count - 1):
        choice = int(np.argmin(best))
        deaths[index] = best[choice]
        visited[choice] = True
        best = np.minimum(best, distances[choice])
        best[visited] = np.inf
    return np.sort(deaths)
