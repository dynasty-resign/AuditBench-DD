"""iter-0009 redesign round: unit tests for the gradient-alignment proxy suite."""
from __future__ import annotations

import math

import torch

from src.ttd.gradient_proxy import (
    compute_all_proxies,
    gradient_alignment_score,
    importance_weighted_grad_align,
    multi_step_gradient_alignment,
    zero_cost_metrics,
)
from src.ttd.models import classifier


def _toy_data(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    syn_x = torch.randn(32, 3, 32, 32, generator=generator)
    syn_y = torch.randint(0, 10, (32,), generator=generator)
    real_x = torch.randn(64, 3, 32, 32, generator=generator)
    real_y = torch.randint(0, 10, (64,), generator=generator)
    return syn_x, syn_y, real_x, real_y


def _model_fn():
    return classifier("convnet", 10, 32)


def test_gradient_alignment_shape():
    syn_x, syn_y, real_x, real_y = _toy_data()
    out = gradient_alignment_score(syn_x, syn_y, real_x, real_y, _model_fn, K=2)
    for key in ("grad_align_mean", "grad_align_weighted", "grad_align_std"):
        assert key in out
        assert isinstance(out[key], float)
    assert isinstance(out["per_layer_means"], dict) and out["per_layer_means"]
    assert len(out["per_init_scores"]) == 2


def test_gradient_alignment_reproducibility():
    syn_x, syn_y, real_x, real_y = _toy_data()
    first = gradient_alignment_score(syn_x, syn_y, real_x, real_y, _model_fn, K=2)
    second = gradient_alignment_score(syn_x, syn_y, real_x, real_y, _model_fn, K=2)
    assert first["grad_align_mean"] == second["grad_align_mean"]
    assert first["grad_align_weighted"] == second["grad_align_weighted"]
    assert first["per_init_scores"] == second["per_init_scores"]


def test_multi_step_shape():
    syn_x, syn_y, real_x, real_y = _toy_data()
    out = multi_step_gradient_alignment(
        syn_x, syn_y, real_x, real_y, _model_fn, steps=(1, 5, 10), K=2
    )
    for key in ("step_0", "step_1", "step_5", "step_10"):
        assert key in out
        assert isinstance(out[key]["grad_align_mean"], float)
        assert isinstance(out[key]["grad_align_weighted"], float)


def test_zero_cost_metrics_shape():
    syn_x, syn_y, _, _ = _toy_data()
    out = zero_cost_metrics(syn_x, syn_y, _model_fn)
    for key in ("grad_norm", "grad_variance", "activation_diversity", "synflow_score"):
        assert key in out
        assert isinstance(out[key], float)
    assert 0.0 <= out["activation_diversity"] <= 1.0
    assert out["grad_norm"] >= 0.0
    assert out["synflow_score"] >= 0.0


def test_no_nan_inf():
    syn_x, syn_y, real_x, real_y = _toy_data()
    align = gradient_alignment_score(syn_x, syn_y, real_x, real_y, _model_fn, K=2)
    weighted = importance_weighted_grad_align(syn_x, syn_y, real_x, real_y, _model_fn, K=2)
    zero_cost = zero_cost_metrics(syn_x, syn_y, _model_fn)
    values = (
        [align[k] for k in ("grad_align_mean", "grad_align_weighted", "grad_align_std")]
        + list(align["per_layer_means"].values())
        + [
            weighted["grad_align_mean_weighted_by_importance"],
            weighted["grad_align_weighted_by_importance"],
            weighted["grad_align_std_weighted_by_importance"],
        ]
        + list(zero_cost.values())
    )
    for value in values:
        assert math.isfinite(value), f"non-finite value: {value}"


def test_gradient_alignment_range():
    syn_x, syn_y, real_x, real_y = _toy_data()
    out = gradient_alignment_score(syn_x, syn_y, real_x, real_y, _model_fn, K=3)
    for value in list(out["per_layer_means"].values()) + out["per_init_scores"]:
        assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= out["grad_align_mean"] <= 1.0 + 1e-6


def test_compute_all_proxies_missing_dir(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        compute_all_proxies(tmp_path, torch.randn(8, 3, 32, 32), torch.zeros(8).long(), None, torch.device("cpu"))
