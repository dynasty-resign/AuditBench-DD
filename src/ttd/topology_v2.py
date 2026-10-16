"""Stable Betti-0 curve summaries for audit-only topology diagnostics."""

from __future__ import annotations

import numpy as np


def fit_epsilon_grid(deaths: np.ndarray, bins: int) -> np.ndarray:
    """Fit one shared non-degenerate filtration grid from reference deaths."""
    values = np.asarray(deaths, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("deaths must have shape [resamples, finite_h0_deaths]")
    if bins < 2:
        raise ValueError("bins must be at least two")
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("deaths contain non-finite values")
    if upper <= lower:
        scale = max(abs(lower), 1.0)
        lower, upper = lower - 1e-6 * scale, upper + 1e-6 * scale
    return np.linspace(lower, upper, bins, dtype=float)


def betti0_curve(deaths: np.ndarray, epsilon_grid: np.ndarray) -> np.ndarray:
    """Return H0 component counts along a filtration grid for each row."""
    values = np.asarray(deaths, dtype=float)
    grid = np.asarray(epsilon_grid, dtype=float)
    if values.ndim != 2:
        raise ValueError("deaths must have shape [resamples, finite_h0_deaths]")
    if grid.ndim != 1 or len(grid) == 0:
        raise ValueError("epsilon_grid must be a non-empty one-dimensional array")
    # A cloud with k finite H0 deaths begins with k+1 components.  A component
    # dies once its corresponding MST edge enters the filtration.
    return 1.0 + (values[:, :, None] > grid[None, None, :]).sum(axis=1)
