"""Unit tests for the iter-0003 proxy-search library (synthetic fixtures only).

No GPU, no network, no real-dataset access. Verifies component math, the
frozen greedy forward-selection rules, and the qualification gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ttd.proxy_search import components as pc  # noqa: E402
from src.ttd.proxy_search import search as ps  # noqa: E402


def test_jsd_base2_properties():
    p = np.array([0.5, 0.5])
    assert pc.jsd_base2(p, p) == pytest.approx(0.0, abs=1e-6)
    q = np.array([1.0, 0.0])
    r = np.array([0.0, 1.0])
    assert pc.jsd_base2(q, r) == pytest.approx(1.0, abs=1e-3)
    assert pc.jsd_base2(p, q) == pytest.approx(pc.jsd_base2(q, p), abs=1e-12)


def test_hhi_orientation():
    uniform = np.tile(np.arange(8), 8)  # flat usage
    concentrated = np.zeros(64, dtype=int)  # single token
    assert pc.struct_token_hhi(uniform, n_tokens=64) > pc.struct_token_hhi(concentrated, n_tokens=64)


def test_cov_perfect_and_zero_overlap():
    syn_tokens = np.array([0, 1, 2, 3])
    real_tokens = np.array([0, 1, 2, 3])
    labels = torch.tensor([0, 0, 1, 1])
    assert pc.struct_token_cov(syn_tokens, labels, real_tokens, labels) == pytest.approx(1.0)
    disjoint = np.array([10, 11, 12, 13])
    assert pc.struct_token_cov(disjoint, labels, real_tokens, labels) == pytest.approx(0.0)


def test_synflow_per_class_finite_and_deterministic():
    torch.manual_seed(0)
    images = torch.randn(12, 3, 32, 32)
    labels = torch.tensor([0, 1, 2] * 4)
    first = pc.synflow_per_class(images, labels, num_classes=3, image_size=32)
    second = pc.synflow_per_class(images, labels, num_classes=3, image_size=32)
    assert np.isfinite(first)
    assert first == pytest.approx(second, rel=1e-9)
    perm = torch.randperm(len(labels))
    null_value = pc.synflow_per_class(images, labels[perm], num_classes=3, image_size=32)
    assert np.isfinite(null_value)


def test_moments_raw_better_for_matched_means():
    rng = np.random.default_rng(0)
    real = torch.tensor(rng.normal(0, 1, (200, 8)), dtype=torch.float32)
    real_labels = torch.tensor([0] * 100 + [1] * 100)
    matched = torch.tensor(rng.normal(0, 1, (20, 8)), dtype=torch.float32)
    shifted = matched + 3.0
    labels = torch.tensor([0] * 10 + [1] * 10)
    real_z, mean, std = pc.reference_zscore(real, real)
    matched_z = (matched - mean) / std
    shifted_z = (shifted - mean) / std
    good = pc.class_conditional_moments_raw(matched_z, labels, real_z, real_labels)
    bad = pc.class_conditional_moments_raw(shifted_z, labels, real_z, real_labels)
    assert good > bad


def _toy_search_inputs():
    rng = np.random.default_rng(0)
    utility = {"d1": np.linspace(0, 1, 12), "d2": np.linspace(0, 1, 12), "d3": np.linspace(0, 1, 12)}
    observed = {
        "good": {ds: vec + rng.normal(0, 0.01, 12) for ds, vec in utility.items()},
        "noise": {ds: rng.normal(0, 1, 12) for ds in utility},
        "grad_norm": {ds: rng.normal(0, 1, 12) for ds in utility},
    }
    nulls = {
        "good": {ds: rng.normal(0, 1, 12) for ds in utility},
        "noise": {ds: rng.normal(0, 1, 12) for ds in utility},
        "grad_norm": {ds: rng.normal(0, 1, 12) for ds in utility},
    }
    control_max = {"d1": 0.5, "d2": 0.5, "d3": 0.5}
    return observed, nulls, utility, control_max


def test_greedy_recovers_signal_and_respects_member_only():
    observed, nulls, utility, control_max = _toy_search_inputs()
    trace = ps.greedy_forward_selection(
        ["noise", "grad_norm", "good"], {"grad_norm"}, observed, nulls, utility, control_max
    )
    assert trace["steps"][0]["action"] == "add"
    assert trace["steps"][0]["component"] == "good"
    assert trace["evaluations_spent"] <= trace["budget"]
    step1_components = {c["component"] for c in trace["steps"][0]["candidates"]}
    assert "grad_norm" not in step1_components


def test_gate_and_no_qualifying_outcome():
    observed, nulls, utility, control_max = _toy_search_inputs()
    trace = ps.greedy_forward_selection(
        ["noise", "grad_norm", "good"], {"grad_norm"}, observed, nulls, utility, control_max
    )
    selection = ps.select_winner(trace["evaluations"])
    assert selection["outcome"] in {"QUALIFYING_COMBINATION_FOUND", "SEARCH_NO_QUALIFYING_COMBINATION"}
    high_bar = {ds: 2.0 for ds in utility}
    trace2 = ps.greedy_forward_selection(
        ["noise", "grad_norm", "good"], {"grad_norm"}, observed, nulls, utility, high_bar
    )
    selection2 = ps.select_winner(trace2["evaluations"])
    assert selection2["outcome"] == "SEARCH_NO_QUALIFYING_COMBINATION"
    assert selection2["winner"] is None


def test_rank_average_monotone_alignment():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 9.0, 8.0, 7.0])
    combo_aa = pc.rank_average([a, a])
    assert combo_aa[0] < combo_aa[-1]
    combo_ab = pc.rank_average([a, b])
    assert combo_ab[0] == pytest.approx(combo_ab[-1])