"""Synthetic-data tests for the pre-registered E2 analysis (auditbench_e2).

No remote host, GPU, or real bank is touched: every test builds a small
cohort with known ground truth (>=3 families, >=2 lineages, >=2 archs,
3 seeds per arch) and checks one frozen statistical caliber.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ttd.auditbench_e2 import (
    ALPHA_GRID,
    ARM_FEATURES,
    build_rows,
    check_complete_case,
    endpoint_rho,
    evaluate_predictions,
    oos_pooled_predictions,
    random_null,
    resample_lineage_clusters,
    validate_e1_report,
    zscore_targets_train_only,
)

ARCHS = ["arch_a", "arch_b"]
SEEDS = [0, 1, 2]


def _synthetic_cohort() -> list[dict]:
    """3 families x 3 lineages x {ipc10, ipc50} x 2 archs, known signal."""
    rows = []
    layout = [
        ("fam_alpha", "lin_one"),
        ("fam_beta", "lin_two"),
        ("fam_gamma", "lin_three"),
    ]
    value = 0.0
    for family, lineage in layout:
        for ipc in (10, 50):
            value += 1.0
            signal = value  # strictly increasing, no ties
            rows.append(
                {
                    "entry_id": f"{lineage}.ipc{ipc}",
                    "release_lineage": lineage,
                    "method_family": family,
                    "ipc": ipc,
                    "scores": {
                        "ridge_proxy": signal,
                        "knn_coverage": 0.1 * signal,
                        "class_moment_dist": -signal,
                        "topo_endpoint_dist": -0.5 * signal,
                        "fid_endpoint": 1.0,
                    },
                    "nulls": {},
                    "utility": {
                        arch: {"mean": signal, "std": 0.01, "n": 3} for arch in ARCHS
                    },
                    "proxy_runtime_sec": 4.0,
                    "train_wall_sec": {
                        f"{arch}|{seed}": 20.0 for arch in ARCHS for seed in SEEDS
                    },
                }
            )
    return rows


def test_zscore_fitted_on_train_fold_only():
    raw = np.array([1.0, 2.0, 3.0, 1000.0])
    arch = np.array(["a", "a", "a", "a"])
    ipc = np.array([50, 50, 50, 50])
    train_mask = np.array([True, True, True, False])
    targets, skipped = zscore_targets_train_only(raw, arch, ipc, train_mask)
    train_mean = raw[:3].mean()
    train_std = raw[:3].std()
    expected = (raw - train_mean) / train_std
    assert skipped == 0
    np.testing.assert_allclose(targets, expected)
    # the held-out extreme value (1000) must not move the parameters
    assert targets[3] == pytest.approx((1000.0 - train_mean) / train_std)


def test_zscore_skips_cells_with_fewer_than_two_train_rows():
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    arch = np.array(["a", "a", "b", "b"])
    ipc = np.array([50, 50, 50, 50])
    train_mask = np.array([True, True, True, False])
    targets, skipped = zscore_targets_train_only(raw, arch, ipc, train_mask)
    assert skipped == 1
    # skipped cell (arch b) keeps raw values
    assert targets[2] == 3.0
    assert targets[3] == 4.0
    # fitted cell is standardized with train-only parameters
    assert targets[0] == pytest.approx((1.0 - 1.5) / 0.5)


def _fold_entry_sets(rows, folds, archs):
    long_keys = [(row["entry_id"], arch) for row in rows for arch in archs]
    for fold in folds:
        train = {long_keys[i] for i in fold["train_index"]}
        test = {long_keys[i] for i in fold["test_index"]}
        yield fold, train, test


def test_lofo_loao_never_mixes_an_artifact_between_train_and_test():
    rows = _synthetic_cohort()
    result = oos_pooled_predictions(
        rows, ARM_FEATURES["base"], ARCHS, ALPHA_GRID, return_details=True
    )
    folds = result["details"]["folds"]
    assert len(folds) == 3 * 2  # 3 families x 2 archs
    for fold, train, test in _fold_entry_sets(rows, folds, ARCHS):
        assert train.isdisjoint(test), f"leakage in fold {fold['held_family']}"
        held_entries = {
            row["entry_id"]
            for row in rows
            if row["method_family"] == fold["held_family"]
        }
        assert all(entry not in held_entries for entry, _ in train)
        assert all(arch != fold["held_arch"] for _, arch in train)


def test_bootstrap_separation_and_lineage_granularity():
    rows = _synthetic_cohort()
    rng = np.random.default_rng(3)
    pseudo, counts = resample_lineage_clusters(rows, rng)
    lineages = sorted({row["release_lineage"] for row in rows})
    assert set(counts) == set(lineages)
    assert sum(counts.values()) == len(lineages)
    # lineage granularity: within one draw suffix every row shares one lineage
    by_draw: dict[str, set[str]] = {}
    for row in pseudo:
        lineage, draw = row["release_lineage"].rsplit("#", 1)
        by_draw.setdefault(draw, set()).add(lineage)
    assert all(len(lineage_set) == 1 for lineage_set in by_draw.values())
    # family labels are unchanged (a family may vanish from a replicate when
    # its lineage is never drawn, but labels are never reassigned)
    assert {row["method_family"] for row in pseudo} <= {
        row["method_family"] for row in rows
    }
    for row in pseudo:
        base_entry = row["entry_id"].rsplit("#", 1)[0]
        original = next(r for r in rows if r["entry_id"] == base_entry)
        assert row["method_family"] == original["method_family"]
    result = oos_pooled_predictions(
        pseudo, ARM_FEATURES["base"], ARCHS, ALPHA_GRID, return_details=True
    )
    for fold, train, test in _fold_entry_sets(
        pseudo, result["details"]["folds"], ARCHS
    ):
        assert train.isdisjoint(test), f"leakage in bootstrap fold {fold}"


def test_random_column_null_does_not_improve_oos_rho():
    rows = _synthetic_cohort()
    ipcs = [10, 50]
    base = oos_pooled_predictions(rows, ARM_FEATURES["base"], ARCHS, ALPHA_GRID)
    base_eval = evaluate_predictions(rows, base["predictions"], ARCHS, ipcs)
    base_primary = endpoint_rho(base_eval, ARCHS, 50)
    rng = np.random.default_rng(0)
    null = random_null(rows, ARCHS, ipcs, ALPHA_GRID, rng, reps=4)
    pure_mean = null["variants"]["pure"]["rho_mean"]["mean"]
    augmented_mean = null["variants"]["base_plus_random"]["primary_rho"]["mean"]
    assert pure_mean is not None and abs(pure_mean) < 0.6
    assert augmented_mean is not None
    assert augmented_mean <= base_primary + 0.2


def test_perfect_proxy_recovers_primary_rho_near_one():
    rows = _synthetic_cohort()
    # score is a strictly monotone transform of utility (identical per arch)
    for row in rows:
        signal = row["utility"]["arch_a"]["mean"]
        row["scores"]["perfect"] = float(np.log1p(signal))
    result = oos_pooled_predictions(rows, ("perfect",), ARCHS, ALPHA_GRID)
    evaluation = evaluate_predictions(rows, result["predictions"], ARCHS, [10, 50])
    primary = endpoint_rho(evaluation, ARCHS, 50)
    secondary = endpoint_rho(evaluation, ARCHS, 10)
    assert primary is not None and primary > 0.99
    assert secondary is not None and secondary > 0.99


def _minimal_valid_report() -> dict:
    return {
        "stage": "auditbench_e1_report",
        "generated_at": "2026-09-05T00:00:00+00:00",
        "evidence_class": "measured",
        "preregistration_sha256": "x",
        "cohort": {},
        "rows": [],
        "oos_predictions": {},
        "arms": {
            arm: {"primary_rho_ipc50": 0.0, "secondary_rho_ipc10": 0.0}
            for arm in ("base", "base_plus_topo")
        },
        "random_null": {"reps": 20},
        "bootstrap": {"reps_requested": 1000},
        "regret": {},
        "compute_break_even": {},
        "per_family_descriptive": {},
        "wall_time": {},
        "screening_statement": "s",
        "implementation_preregistration_map": [],
        "statistical_uncertainty": {},
    }


def test_validate_e1_report_accepts_minimal_valid():
    validate_e1_report(_minimal_valid_report())


@pytest.mark.parametrize("missing_key", ["bootstrap", "random_null", "arms", "rows"])
def test_validate_e1_report_rejects_incomplete(missing_key):
    report = _minimal_valid_report()
    report.pop(missing_key)
    with pytest.raises(ValueError, match="missing"):
        validate_e1_report(report)


def test_validate_e1_report_rejects_wrong_stage():
    report = _minimal_valid_report()
    report["stage"] = "something_else"
    with pytest.raises(ValueError, match="stage"):
        validate_e1_report(report)


def _write_fake_formal(formal_dir, entry_id, smoke_only=False):
    scores = {
        name: 1.0
        for name in (
            "fid_endpoint",
            "knn_precision",
            "knn_recall",
            "knn_coverage",
            "effective_rank",
            "topo_traj_dist",
            "topo_endpoint_dist",
            "betti_traj_dist",
            "class_moment_dist",
            "ridge_proxy",
            "short_run_5ep",
        )
    }
    scores["runtime_sec"] = 4.2
    scores["nulls"] = {
        name: 0.0
        for name in (
            "labelperm_class_moment_dist",
            "labelperm_topo_traj_dist",
            "timeshuffle_topo_traj_dist",
            "timeshuffle_betti_traj_dist",
            "cardinality_half_fid_endpoint",
            "cardinality_half_effective_rank",
        )
    }
    (formal_dir / f"{entry_id}.scores.json").write_text(
        json.dumps(
            {
                "entry_id": entry_id,
                "ipc": 10,
                "smoke_only": smoke_only,
                "scores": scores,
            }
        ),
        encoding="utf-8",
    )
    runs = [
        {"architecture": arch, "seed": seed, "top1": 0.4}
        for arch in ARCHS
        for seed in SEEDS
    ]
    summary = [
        {"architecture": arch, "mean": 0.4, "std": 0.01, "n": 3} for arch in ARCHS
    ]
    (formal_dir / f"{entry_id}.eval_input.evaluation.json").write_text(
        json.dumps({"runs": runs, "summary": summary}), encoding="utf-8"
    )


def _fake_preregistration(entry_ids) -> dict:
    return {
        "cohort": {
            "entries": [
                {
                    "entry_id": entry_id,
                    "release_lineage": "lin_one",
                    "method_family": "fam_alpha",
                    "ipc": 10,
                }
                for entry_id in entry_ids
            ]
        }
    }


def test_build_rows_reads_formal_artifacts(tmp_path):
    _write_fake_formal(tmp_path, "a.ipc10")
    _write_fake_formal(tmp_path, "b.ipc10")
    progress = {
        "entries": {
            "a.ipc10": {"cells": {"arch_a|0": {"wall_sec": 21.5}}},
            "b.ipc10": {"cells": {}},
        }
    }
    rows = build_rows(
        _fake_preregistration(["a.ipc10", "b.ipc10"]), tmp_path, progress
    )
    assert len(rows) == 2
    first = rows[0]
    assert first["method_family"] == "fam_alpha"
    assert first["proxy_runtime_sec"] == 4.2
    assert first["utility"]["arch_a"] == {"mean": 0.4, "std": 0.01, "n": 3}
    assert first["train_wall_sec"] == {"arch_a|0": 21.5}
    assert rows[1]["train_wall_sec"] == {}
    completeness = check_complete_case(rows, ARCHS, SEEDS)
    assert completeness["complete"] is True


def test_build_rows_rejects_smoke_artifacts(tmp_path):
    _write_fake_formal(tmp_path, "a.ipc10", smoke_only=True)
    with pytest.raises(ValueError, match="not a formal artifact"):
        build_rows(_fake_preregistration(["a.ipc10"]), tmp_path)


def test_build_rows_requires_all_artifacts(tmp_path):
    _write_fake_formal(tmp_path, "a.ipc10")
    with pytest.raises(FileNotFoundError):
        build_rows(_fake_preregistration(["a.ipc10", "b.ipc10"]), tmp_path)


def test_check_complete_case_flags_missing_utility(tmp_path):
    _write_fake_formal(tmp_path, "a.ipc10")
    rows = build_rows(_fake_preregistration(["a.ipc10"]), tmp_path)
    rows[0]["utility"]["arch_a"]["n"] = 1
    completeness = check_complete_case(rows, ARCHS, SEEDS)
    assert completeness["complete"] is False
    assert any("arch_a" in problem for problem in completeness["problems"])
