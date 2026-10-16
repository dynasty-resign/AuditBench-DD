"""Synthetic-data tests for the two-dataset pooled summary (auditbench_e2b)
and for the iter-0006 dataset parameterization of auditbench_e2.

No remote host, GPU, or real bank is touched: every test builds small
two-dataset cohorts with known ground truth and checks one frozen caliber
(fold separation, train-only dataset-aware z-score, pooled/per-dataset
endpoint aggregation, lineage x dataset bootstrap granularity, four-parameter
screening template, report schema, stage parameterization).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ttd.auditbench_e2 import (
    ALPHA_GRID,
    ARM_FEATURES,
    E2_IMPLEMENTATION_MAP,
    run_e2_analysis,
    validate_e1_report,
    validate_report,
)
from ttd.auditbench_e2b import (
    POOLED_IMPLEMENTATION_MAP,
    endpoint_rho_per_dataset,
    endpoint_rho_pooled,
    evaluate_predictions_2ds,
    oos_pooled_predictions_2ds,
    resample_lineage_dataset_clusters,
    run_e2b_pooled,
    validate_pooled_report,
    zscore_targets_train_only_2ds,
)

ARCHS = ["arch_a", "arch_b"]
SEEDS = [0, 1, 2]
DATASETS = ["cifar10", "cifar100"]


def _cohort_rows(scale_by_dataset=None, offset_by_dataset=None):
    """2 datasets x 3 families x 3 lineages x {ipc10, ipc50}, known signal."""
    scale_by_dataset = scale_by_dataset or {"cifar10": 1.0, "cifar100": 1.0}
    offset_by_dataset = offset_by_dataset or {"cifar10": 0.0, "cifar100": 0.0}
    layout = [
        ("fam_alpha", "lin_one"),
        ("fam_beta", "lin_two"),
        ("fam_gamma", "lin_three"),
    ]
    rows = []
    value = 0.0
    for dataset in DATASETS:
        for family, lineage in layout:
            for ipc in (10, 50):
                value += 1.0
                signal = value
                utility = {
                    arch: {
                        "mean": offset_by_dataset[dataset]
                        + scale_by_dataset[dataset] * signal,
                        "std": 0.01,
                        "n": 3,
                    }
                    for arch in ARCHS
                }
                rows.append(
                    {
                        "entry_id": f"{dataset}:{lineage}.ipc{ipc}",
                        "dataset": dataset,
                        "release_lineage": lineage,
                        "lineage_dataset": f"{dataset}:{lineage}",
                        "method_family": family,
                        "ipc": ipc,
                        "scores": {
                            "ridge_proxy": signal,
                            "knn_coverage": 0.1 * signal,
                            "class_moment_dist": -signal,
                            "topo_endpoint_dist": -0.5 * signal,
                            "fid_endpoint": 1.0,
                        },
                        "utility": utility,
                    }
                )
    return rows


def _fold_entry_sets(rows, folds, archs):
    long_keys = [(row["entry_id"], arch) for row in rows for arch in archs]
    for fold in folds:
        train = {long_keys[i] for i in fold["train_index"]}
        test = {long_keys[i] for i in fold["test_index"]}
        yield fold, train, test


def test_pooled_fold_separation_and_fold_count():
    rows = _cohort_rows()
    result = oos_pooled_predictions_2ds(
        rows, ARM_FEATURES["base"], ARCHS, ALPHA_GRID, return_details=True
    )
    folds = result["details"]["folds"]
    assert len(folds) == 3 * 2  # 3 pooled families x 2 archs
    for fold, train, test in _fold_entry_sets(rows, folds, ARCHS):
        assert train.isdisjoint(test), f"leakage in fold {fold['held_family']}"
        held_entries = {
            row["entry_id"]
            for row in rows
            if row["method_family"] == fold["held_family"]
        }
        assert all(entry not in held_entries for entry, _ in train)
        assert all(arch != fold["held_arch"] for _, arch in train)


def test_zscore_2ds_train_only_and_dataset_aware():
    raw = np.array([1.0, 2.0, 3.0, 101.0, 102.0, 103.0])
    cells = np.array(
        [
            "cifar10|a|50",
            "cifar10|a|50",
            "cifar10|a|50",
            "cifar100|a|50",
            "cifar100|a|50",
            "cifar100|a|50",
        ]
    )
    train_mask = np.array([True, True, False, True, True, False])
    targets, skipped = zscore_targets_train_only_2ds(raw, cells, train_mask)
    assert skipped == 0
    mean10, std10 = raw[:2].mean(), raw[:2].std()
    mean100, std100 = raw[3:5].mean(), raw[3:5].std()
    assert targets[0] == pytest.approx((1.0 - mean10) / std10)
    assert targets[2] == pytest.approx((3.0 - mean10) / std10)  # held-out, train params
    assert targets[3] == pytest.approx((101.0 - mean100) / std100)
    assert targets[5] == pytest.approx((103.0 - mean100) / std100)
    # datasets never share z-score parameters
    assert targets[2] != pytest.approx((3.0 - raw[[0, 1, 3, 4]].mean()) / raw[[0, 1, 3, 4]].std())


def test_zscore_2ds_skips_cells_with_fewer_than_two_train_rows():
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    cells = np.array(["cifar10|a|50"] * 2 + ["cifar100|a|50"] * 2)
    train_mask = np.array([True, True, True, False])
    targets, skipped = zscore_targets_train_only_2ds(raw, cells, train_mask)
    assert skipped == 1
    assert targets[2] == 3.0  # skipped cell keeps raw values
    assert targets[3] == 4.0


def test_pooled_endpoint_recovers_signal_across_dataset_offsets():
    rows = _cohort_rows(
        scale_by_dataset={"cifar10": 1.0, "cifar100": 3.0},
        offset_by_dataset={"cifar10": 0.0, "cifar100": 100.0},
    )
    result = oos_pooled_predictions_2ds(rows, ARM_FEATURES["base"], ARCHS, ALPHA_GRID)
    evaluation = evaluate_predictions_2ds(
        rows, result["predictions"], ARCHS, DATASETS, [10, 50]
    )
    pooled = endpoint_rho_pooled(evaluation, ARCHS, DATASETS, 50)
    assert pooled is not None and pooled > 0.9
    # pooled endpoint = mean over the 2 archs x 2 datasets cells at ipc50
    cell_values = [
        evaluation["rho"][f"{dataset}|{arch}|50"]
        for dataset in DATASETS
        for arch in ARCHS
    ]
    assert pooled == pytest.approx(float(np.mean(cell_values)))
    for dataset in DATASETS:
        per_ds = endpoint_rho_per_dataset(evaluation, ARCHS, dataset, 50)
        assert per_ds == pytest.approx(
            float(np.mean([evaluation["rho"][f"{dataset}|{arch}|50"] for arch in ARCHS]))
        )
    # regret of a perfect-ranking arm is zero in every cell
    assert all(value == pytest.approx(0.0) for value in evaluation["regret"].values())


def test_resample_lineage_dataset_clusters_granularity():
    rows = _cohort_rows()
    rng = np.random.default_rng(3)
    pseudo, counts = resample_lineage_dataset_clusters(rows, rng)
    clusters = sorted({row["lineage_dataset"] for row in rows})
    assert set(counts) == set(clusters)
    assert sum(counts.values()) == len(clusters) == 6
    by_draw: dict[str, set[str]] = {}
    for row in pseudo:
        lineage_dataset, draw = row["lineage_dataset"].rsplit("#", 1)
        by_draw.setdefault(draw, set()).add(lineage_dataset)
    assert all(len(found) == 1 for found in by_draw.values())
    # family labels unchanged; dataset tag preserved on entry ids
    original_families = {row["entry_id"]: row["method_family"] for row in rows}
    for row in pseudo:
        base_id = row["entry_id"].rsplit("#", 1)[0]
        assert row["method_family"] == original_families[base_id]
        assert row["dataset"] == base_id.split(":", 1)[0]

# ---------------------------------------------------------------------------
# Report-level tests (run_e2b_pooled on synthetic per-dataset reports)
# ---------------------------------------------------------------------------

_SCORE_KEYS = (
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


def _report_rows(dataset: str) -> list[dict]:
    layout = [
        ("fam_alpha", "lin_one"),
        ("fam_beta", "lin_two"),
        ("fam_gamma", "lin_three"),
    ]
    rows = []
    value = 0.0
    for family, lineage in layout:
        for ipc in (10, 50):
            value += 1.0
            signal = value + (0.0 if dataset == "cifar10" else 0.5)
            scores = {key: 0.0 for key in _SCORE_KEYS}
            scores.update(
                {
                    "ridge_proxy": signal,
                    "knn_coverage": 0.1 * signal,
                    "class_moment_dist": -signal,
                    "topo_endpoint_dist": -0.5 * signal,
                }
            )
            rows.append(
                {
                    "entry_id": f"{lineage}.ipc{ipc}",
                    "release_lineage": lineage,
                    "method_family": family,
                    "ipc": ipc,
                    "scores": scores,
                    "utility": {
                        arch: {"mean": signal, "std": 0.01, "n": 3} for arch in ARCHS
                    },
                    "proxy_runtime_sec": 4.0,
                }
            )
    return rows


def _fake_dataset_report(stage: str, dataset: str, primary: float) -> dict:
    return {
        "stage": stage,
        "generated_at": "2026-09-06T00:00:00+00:00",
        "evidence_class": "measured",
        "preregistration_sha256": "x" * 64,
        "cohort": {
            "n_entries": 6,
            "entry_ids": [row["entry_id"] for row in _report_rows(dataset)],
            "families": ["fam_alpha", "fam_beta", "fam_gamma"],
            "lineages": ["lin_one", "lin_two", "lin_three"],
            "architectures": ARCHS,
            "seeds": SEEDS,
            "ipcs": [10, 50],
            "completeness": {"complete": True, "problems": [], "n_rows": 6},
        },
        "rows": _report_rows(dataset),
        "oos_predictions": {},
        "oos_prediction_records": {},
        "arms": {
            arm: {"primary_rho_ipc50": primary, "secondary_rho_ipc10": primary}
            for arm in ("base", "base_plus_topo")
        },
        "alpha_selection": {"grid": list(ALPHA_GRID), "chosen_per_fold": {}},
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


def _fake_preregistration() -> dict:
    return {
        "statistics": {
            "screening_statement_template": (
                "E2 (n={n} artifacts, {f} families, {l} lineages, {d} datasets) "
                "is screening evidence: grouped CIs are wide by construction and "
                "findings are hypotheses for confirmatory multi-dataset follow-up, "
                "not established cross-method/cross-architecture conclusions. "
                "Per-dataset endpoints are reported alongside the pooled endpoint; "
                "the pooled endpoint alone is never reported."
            )
        }
    }


def _write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_e2b_pooled_end_to_end(tmp_path):
    e1_path = _write_json(
        tmp_path, "e1_report.json", _fake_dataset_report("auditbench_e1_report", "cifar10", 0.11)
    )
    e2_path = _write_json(
        tmp_path, "e2_report.json", _fake_dataset_report("auditbench_e2_report", "cifar100", 0.22)
    )
    prereg_path = _write_json(tmp_path, "e2_preregistration.json", _fake_preregistration())
    report = run_e2b_pooled(
        e1_path, e2_path, prereg_path, bootstrap_reps=20, random_reps=2, log_every=0
    )
    assert report["stage"] == "auditbench_e2b_pooled_report"
    validate_pooled_report(report)
    # per-dataset standalone endpoints are verbatim copies of the input reports
    assert report["per_dataset_standalone"]["cifar10"]["arms"]["base"][
        "primary_rho_ipc50"
    ] == pytest.approx(0.11)
    assert report["per_dataset_standalone"]["cifar100"]["arms"]["base"][
        "primary_rho_ipc50"
    ] == pytest.approx(0.22)
    # pooled + per-dataset endpoints are defined and consistent
    for arm in ("base", "base_plus_topo"):
        bundle = report["arms"][arm]
        assert bundle["primary_rho_ipc50_pooled"] is not None
        cell_rhos = [
            rho
            for key, rho in bundle["rho_cells"].items()
            if key.endswith("|50")
        ]
        assert bundle["primary_rho_ipc50_pooled"] == pytest.approx(
            float(np.mean(cell_rhos))
        )
        for dataset in DATASETS:
            assert bundle["per_dataset"][dataset]["primary_rho_ipc50"] is not None
    # four-parameter screening statement filled verbatim
    assert "n=12 artifacts, 3 families, 6 lineages, 2 datasets" in report[
        "screening_statement"
    ]
    # oos prediction records: one per (entry, arch) per arm
    for arm in ("base", "base_plus_topo"):
        records = report["oos_prediction_records"][arm]
        assert len(records) == 12 * len(ARCHS)
        keys = {(record["entry_id"], record["arch"]) for record in records}
        assert len(keys) == 12 * len(ARCHS)
    # lineage x dataset clusters registered
    assert len(report["cohort"]["lineage_dataset_clusters"]) == 6
    assert report["bootstrap"]["cluster_unit"] == "release_lineage x dataset"
    assert report["implementation_preregistration_map"] == POOLED_IMPLEMENTATION_MAP


def test_validate_pooled_report_rejects_wrong_stage_and_missing(tmp_path):
    e1_path = _write_json(
        tmp_path, "e1_report.json", _fake_dataset_report("auditbench_e1_report", "cifar10", 0.11)
    )
    e2_path = _write_json(
        tmp_path, "e2_report.json", _fake_dataset_report("auditbench_e2_report", "cifar100", 0.22)
    )
    prereg_path = _write_json(tmp_path, "e2_preregistration.json", _fake_preregistration())
    report = run_e2b_pooled(
        e1_path, e2_path, prereg_path, bootstrap_reps=10, random_reps=1, log_every=0
    )
    broken = dict(report, stage="something_else")
    with pytest.raises(ValueError, match="stage"):
        validate_pooled_report(broken)
    broken = dict(report)
    broken.pop("bootstrap")
    with pytest.raises(ValueError, match="missing"):
        validate_pooled_report(broken)


# ---------------------------------------------------------------------------
# auditbench_e2 stage/n_datasets/implementation_map parameterization
# ---------------------------------------------------------------------------


def _write_fake_formal(formal_dir, entry_id, mean_map, smoke_only=False):
    scores = {key: 1.0 for key in _SCORE_KEYS}
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
        {"architecture": arch, "seed": seed, "top1": mean_map[arch]}
        for arch in ARCHS
        for seed in SEEDS
    ]
    summary = [
        {"architecture": arch, "mean": mean_map[arch], "std": 0.01, "n": 3}
        for arch in ARCHS
    ]
    (formal_dir / f"{entry_id}.eval_input.evaluation.json").write_text(
        json.dumps({"runs": runs, "summary": summary}), encoding="utf-8"
    )


def _fake_e2_preregistration(entry_ids) -> dict:
    return {
        "cohort": {
            "entries": [
                {
                    "entry_id": entry_id,
                    "release_lineage": f"lin_{index}",
                    "method_family": f"fam_{index}",
                    "ipc": 10,
                }
                for index, entry_id in enumerate(entry_ids)
            ]
        },
        "student_architectures": {"set": ARCHS},
        "training_budget": {"seeds": SEEDS},
        "statistics": {
            "screening_statement_template": (
                "E2 (n={n} artifacts, {f} families, {l} lineages, {d} datasets) "
                "is screening evidence."
            )
        },
    }


def test_run_e2_analysis_stage_parameterization(tmp_path):
    entry_ids = [f"e{index}.ipc10" for index in range(6)]
    prereg = _fake_e2_preregistration(entry_ids)
    prereg_path = _write_json(tmp_path, "e2_preregistration.json", prereg)
    formal = tmp_path / "formal"
    formal.mkdir()
    for index, entry_id in enumerate(entry_ids):
        _write_fake_formal(
            formal, entry_id, {arch: float(index + 1) for arch in ARCHS}
        )
    report = run_e2_analysis(
        prereg_path,
        formal,
        bootstrap_reps=10,
        random_reps=2,
        stage="auditbench_e2_report",
        n_datasets=1,
        implementation_map=E2_IMPLEMENTATION_MAP,
    )
    assert report["stage"] == "auditbench_e2_report"
    assert report["cohort"]["n_datasets"] == 1
    validate_report(report, stage="auditbench_e2_report")
    with pytest.raises(ValueError, match="stage"):
        validate_e1_report(report)
    assert "n=6 artifacts, 6 families, 6 lineages, 1 datasets" in report[
        "screening_statement"
    ]
    assert report["implementation_preregistration_map"] == E2_IMPLEMENTATION_MAP
    for arm in ("base", "base_plus_topo"):
        records = report["oos_prediction_records"][arm]
        assert len(records) == 6 * len(ARCHS)
        keys = {(record["entry_id"], record["arch"]) for record in records}
        assert len(keys) == 6 * len(ARCHS)
        for record in records:
            assert record["held_family"] == record["method_family"]
            assert record["held_arch"] == record["arch"]
            assert record["alpha"] in ALPHA_GRID


def test_degenerate_single_cluster_fold_is_skipped_not_raised():
    """A bootstrap replicate can collapse to one cluster; folds with no train
    rows must be skipped (endpoints undefined, replicate counted skipped)
    rather than crash the ridge fit."""
    rows = [row for row in _cohort_rows() if row["dataset"] == "cifar10"]
    one_family = [dict(row, method_family="fam_only") for row in rows]
    result = oos_pooled_predictions_2ds(
        one_family, ARM_FEATURES["base"], ARCHS, ALPHA_GRID
    )
    assert all(preds == {} for preds in result["predictions"].values())
    evaluation = evaluate_predictions_2ds(
        one_family, result["predictions"], ARCHS, ["cifar10"], [10, 50]
    )
    assert endpoint_rho_pooled(evaluation, ARCHS, ["cifar10"], 50) is None
