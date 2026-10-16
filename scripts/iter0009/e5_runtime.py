"""Runtime-only registration for the frozen Imagenette normalisation pin.

This module intentionally performs no I/O, download, model construction or
evaluation.  The caller supplies the values recorded in the E5 evidence pack.
"""

from __future__ import annotations

from collections.abc import Sequence


def register(mean: Sequence[float], std: Sequence[float]) -> dict[str, object]:
    """Register the 64px Imagenette normalisation constants for this process."""
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Imagenette normalisation requires exactly three channels")
    mean_tuple = tuple(float(value) for value in mean)
    std_tuple = tuple(float(value) for value in std)
    if any(value <= 0.0 for value in std_tuple):
        raise ValueError("Imagenette standard deviations must be positive")
    from ttd import data

    data.DATASET_STATS["imagenette"] = (mean_tuple, std_tuple)
    return {"dataset": "imagenette", "image_size": 64, "registered": True}
