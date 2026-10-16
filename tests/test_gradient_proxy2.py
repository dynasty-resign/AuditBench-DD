"""iter-0002 redesign #2: unit tests for conditioned gradient proxies."""
from __future__ import annotations

import math

import torch

from src.ttd.gradient_proxy2 import (
    compute_conditioned_metrics,
    grad_vector,
    metrics_from_syn_grads,
    prepare_real_side,
    random_init_reference,
    train_conditioned_reference,
    _tracked_parameters,
)
from src.ttd.gradient_proxy import _deterministic_subset


def _toy_data(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    syn_x = torch.randn(32, 3, 32, 32, generator=generator)
    syn_y = torch.randint(0, 10, (32,), generator=generator)
    real_x = torch.randn(96, 3, 32, 32, generator=generator)
    real_y = torch.randint(0, 10, (96,), generator=generator)
    held_x = torch.randn(48, 3, 32, 32, generator=generator)
    held_y = torch.randint(0, 10, (48,), generator=generator)
    return syn_x, syn_y, real_x, real_y, held_x, held_y


def _reference():
    return random_init_reference(num_classes=10, image_size=32, seed=1)


def test_train_conditioned_reference_deterministic():
    _, _, real_x, real_y, _, _ = _toy_data()
    model_a, hist_a = train_conditioned_reference(
        real_x, real_y, num_classes=10, image_size=32, epochs=1, batch_size=64
    )
    model_b, hist_b = train_conditioned_reference(
        real_x, real_y, num_classes=10, image_size=32, epochs=1, batch_size=64
    )
    assert hist_a == hist_b and len(hist_a) == 1
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
    _, hist_c = train_conditioned_reference(
        real_x, real_y, num_classes=10, image_size=32, epochs=2, batch_size=64
    )
    assert hist_c[0] == hist_a[0]


def test_metrics_finite_and_deterministic():
    syn_x, syn_y, real_x, real_y, held_x, held_y = _toy_data()
    reference = _reference()
    device = torch.device("cpu")
    real_side = prepare_real_side(
        reference, real_x, real_y, held_x, held_y, num_classes=10, device=device
    )
    first = compute_conditioned_metrics(
        syn_x, syn_y, reference, real_side, num_classes=10, device=device
    )
    second = compute_conditioned_metrics(
        syn_x, syn_y, reference, real_side, num_classes=10, device=device
    )
    assert first == second
    for key, value in first.items():
        assert math.isfinite(value), f"non-finite {key}: {value}"


def test_delta_l_real_first_order_form():
    syn_x, syn_y, real_x, real_y, held_x, held_y = _toy_data()
    reference = _reference()
    device = torch.device("cpu")
    real_side = prepare_real_side(
        reference, real_x, real_y, held_x, held_y, num_classes=10, device=device
    )
    out = compute_conditioned_metrics(
        syn_x, syn_y, reference, real_side, num_classes=10, device=device
    )
    params = _tracked_parameters(reference)
    syn_batch, syn_b_y = _deterministic_subset(syn_x, syn_y, 256, 0)
    g_syn = grad_vector(reference, params, syn_batch, syn_b_y, device)
    expected = -0.01 * float(g_syn @ real_side["g_heldout"])
    assert abs(out["delta_l_real"] - expected) < 1e-9


def test_control_pipelines_exist():
    syn_x, syn_y, real_x, real_y, _, _ = _toy_data()
    generator = torch.Generator().manual_seed(7)
    held_x = torch.randn(len(real_y), 3, 32, 32, generator=generator)
    held_y = torch.randint(0, 10, (len(real_y),), generator=generator)
    reference = _reference()
    device = torch.device("cpu")
    permutation = torch.randperm(len(real_y))
    real_obs = prepare_real_side(
        reference, real_x, real_y, held_x, held_y, num_classes=10, device=device
    )
    real_shuffled = prepare_real_side(
        reference,
        real_x,
        real_y,
        held_x,
        held_y,
        num_classes=10,
        device=device,
        label_permutation=permutation,
    )
    assert not torch.equal(real_obs["g_train"], real_shuffled["g_train"])
    syn_perm = torch.randperm(len(syn_y))
    out_obs = compute_conditioned_metrics(
        syn_x, syn_y, reference, real_obs, num_classes=10, device=device
    )
    out_null = compute_conditioned_metrics(
        syn_x,
        syn_y,
        reference,
        real_obs,
        num_classes=10,
        device=device,
        syn_label_permutation=syn_perm,
    )
    assert set(out_obs) == set(out_null)
    out_cc = compute_conditioned_metrics(
        syn_x, syn_y, _reference(), real_obs, num_classes=10, device=device
    )
    assert set(out_cc) == set(out_obs)


def test_within_class_macro_average():
    g_syn = torch.ones(4)
    syn_pc = {0: torch.ones(4), 1: torch.full((4,), 2.0)}
    real_side = {
        "g_train": torch.ones(4),
        "g_heldout": torch.full((4,), 3.0),
        "per_class": {0: torch.ones(4), 1: torch.full((4,), 0.5)},
    }
    out = metrics_from_syn_grads(g_syn, syn_pc, real_side, step_lr=0.01)
    assert out["grad_dot_cond"] == 4.0
    assert abs(out["delta_l_real"] - (-0.01 * 12.0)) < 1e-9
    assert abs(out["grad_dot_cond_within_class"] - (4.0 + 4.0) / 2) < 1e-9
    assert out["num_classes_within"] == 2.0
    empty = metrics_from_syn_grads(g_syn, {}, real_side)
    assert math.isnan(empty["grad_dot_cond_within_class"])
