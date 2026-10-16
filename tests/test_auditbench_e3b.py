"""Synthetic-data tests for the E3b inference round (auditbench_e3b).

No GPU, no remote host, no real E3a artifact is touched: every test builds a
small synthetic cohort with known ground truth and checks one frozen caliber
(progress merge, register cross-check, three-dataset row construction, fold
separation on the 3-dataset table, screening-statement fills, report schema).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ttd.auditbench_e2 import ALPHA_GRID, ARM_FEATURES, ARMS
from ttd.auditbench_e2b import oos_pooled_predictions_2ds
from ttd.auditbench_e3b import (
    E3_SINGLE_STAGE,
    merge_e3a_progress,
    build_e3_rows,
    pooled_rows_3ds,
    run_e3b_pooled,
    run_e3b_single,
    validate_pooled_3ds_report,
)

ARCHS = ["arch_a", "arch_b"]
SEEDS = [0, 1, 2]
TEMPLATE = (
    "E3 (n={n} artifacts, {f} families, {l} lineages, {d} datasets) is "
    "screening evidence: grouped CIs are wide by construction."
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _lane_progress(entries_cells, started="2026-01-01T00:00:00+00:00"):
    return {
        "stage": "auditbench_e3a_matrix",
        "started_at": started,
        "updated_at": started,
        "entries": entries_cells,
    }


def test_merge_e3a_progress_disjoint_and_conflict(tmp_path):
    lanes = {
        "lane1": _lane_progress({"e.a": {"cells": {"arch_a|0": {"status": "done", "wall_sec": 5.0}}}}),
        "lane2": _lane_progress({"e.b": {"cells": {"arch_a|0": {"status": "done", "wall_sec": 7.0}}}}),
        "gpu0": _lane_progress({"e.a": {"cells": {}, "proxy": {"status": "done"}}}),
        "gpu0proxy": _lane_progress({}),
    }
    for lane, payload in lanes.items():
        _write_json(tmp_path / f"e3a_progress.{lane}.json", payload)
    merged = merge_e3a_progress(tmp_path)
    assert set(merged["entries"]) == {"e.a", "e.b"}
    assert merged["entries"]["e.a"]["cells"]["arch_a|0"]["wall_sec"] == 5.0
    assert merged["entries"]["e.a"]["proxy_blocks"]["gpu0"]["status"] == "done"
    assert set(merged["lanes"]) == {"lane1", "lane2", "gpu0", "gpu0proxy"}

    lanes["lane2"] = _lane_progress({"e.a": {"cells": {"arch_a|0": {"status": "done"}}}})
    _write_json(tmp_path / "e3a_progress.lane2.json", lanes["lane2"])
    with pytest.raises(ValueError, match="conflict"):
        merge_e3a_progress(tmp_path)


def _e3_fixture(tmp_path, n_families=4, tamper_register=False):
    """Synthetic E3-like preregistration + formal_dir + register."""
    formal_dir = tmp_path / "e3a_matrix"
    entries = []
    register = {"utilities": {}, "cell_seed_stats": {}, "near_random": {}}
    value = 0.0
    for index in range(n_families):
        family = f"fam_{index}"
        lineage = f"lin_{index}"
        for ipc in (10, 50):
            value += 1.0
            signal = value
            entry_id = f"entry.{lineage}.ipc{ipc}"
            entries.append(
                {
                    "entry_id": entry_id,
                    "release_lineage": lineage,
                    "method_family": family,
                    "ipc": ipc,
                }
            )
            runs = []
            summary = []
            seed_maps = {}
            for arch_i, arch in enumerate(ARCHS):
                base = signal + 0.1 * arch_i
                tops = [base - 0.01, base, base + 0.01]
                seed_map = {}
                for seed, top1 in zip(SEEDS, tops):
                    runs.append({"architecture": arch, "seed": seed, "top1": top1})
                    seed_map[str(seed)] = top1
                seed_maps[arch] = seed_map
                mean = float(np.mean(tops))
                std = float(np.std(tops))  # ddof=0 register caliber
                summary.append({"architecture": arch, "mean": mean, "std": std, "n": 3})
                register["cell_seed_stats"][f"{entry_id}|{arch}"] = {
                    "n_seeds": 3,
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "values": seed_map,
                }
            register["utilities"][entry_id] = seed_maps
            if tamper_register and value == 1.0:
                register["utilities"][entry_id]["arch_a"]["0"] += 1.0
            scores = {
                "ridge_proxy": signal,
                "knn_coverage": 0.1 * signal,
                "class_moment_dist": -signal,
                "topo_endpoint_dist": -0.5 * signal,
                "fid_endpoint": 1.0,
                "nulls": {},
                "runtime_sec": 3.0,
            }
            _write_json(
                formal_dir / f"{entry_id}.scores.json",
                {"entry_id": entry_id, "ipc": ipc, "smoke_only": False, "scores": scores},
            )
            _write_json(
                formal_dir / f"{entry_id}.eval_input.evaluation.json",
                {"runs": runs, "summary": summary},
            )
    preregistration = {
        "cohort": {"entries": entries},
        "student_architectures": {"set": ARCHS},
        "training_budget": {"seeds": SEEDS},
        "statistics": {"screening_statement_template": TEMPLATE},
    }
    for arch in ARCHS:
        for ipc in (10, 50):
            register["near_random"][f"{arch}|ipc{ipc}"] = {
                "status": "evaluated",
                "std_across_entries": 1.0,
                "mean_seed_std": 0.01,
                "near_random_descriptive": False,
                "rule": "synthetic",
            }
    _write_json(
        formal_dir / "e3a_progress.lane1.json",
        _lane_progress(
            {
                entry["entry_id"]: {
                    "cells": {
                        f"{arch}|{seed}": {"status": "done", "wall_sec": 10.0}
                        for arch in ARCHS
                        for seed in SEEDS
                    }
                }
                for entry in entries
            }
        ),
    )
    for lane in ("lane2", "gpu0", "gpu0proxy"):
        _write_json(formal_dir / f"e3a_progress.{lane}.json", _lane_progress({}))
    return preregistration, formal_dir, register


def test_build_e3_rows_register_cross_check(tmp_path):
    preregistration, formal_dir, register = _e3_fixture(tmp_path)
    rows = build_e3_rows(
        preregistration, formal_dir, register, merge_e3a_progress(formal_dir)
    )
    assert len(rows) == 8
    assert all(row["train_wall_sec"] for row in rows)

    _, formal_dir_bad, register_bad = _e3_fixture(tmp_path / "bad", tamper_register=True)
    with pytest.raises(ValueError, match="divergence"):
        build_e3_rows(preregistration, formal_dir_bad, register_bad)


def test_run_e3b_single_schema_and_screening(tmp_path):
    preregistration, formal_dir, register = _e3_fixture(tmp_path)
    report = run_e3b_single(
        preregistration, formal_dir, register,
        bootstrap_reps=6, random_reps=2, seed=0,
    )
    assert report["stage"] == E3_SINGLE_STAGE
    assert report["cohort"]["n_entries"] == 8
    assert report["screening_statement"] == TEMPLATE.format(n=8, f=4, l=4, d=1)
    for arm in ARMS:
        assert report["arms"][arm]["primary_rho_ipc50"] is not None
        assert report["arms"][arm]["secondary_rho_ipc10"] is not None
    assert report["compute_break_even"]["n_train_cells_measured"] == 8 * 2 * 3
    assert report["bootstrap"]["cluster_unit"] == "release_lineage"
    # known monotone signal: base arm recovers perfect ranking in every fold
    assert report["arms"]["base"]["primary_rho_ipc50"] == pytest.approx(1.0)


def _report_fixture(dataset, stage, scale=1.0):
    rows = []
    value = 0.0
    for index in range(3):
        for ipc in (10, 50):
            value += 1.0
            signal = value
            rows.append(
                {
                    "entry_id": f"{dataset}.lin_{index}.ipc{ipc}",
                    "release_lineage": f"{dataset}_lin_{index}",
                    "method_family": f"fam_{index}",
                    "ipc": ipc,
                    "scores": {
                        "ridge_proxy": signal,
                        "knn_coverage": 0.1 * signal,
                        "class_moment_dist": -signal,
                        "topo_endpoint_dist": -0.5 * signal,
                    },
                    "utility": {
                        arch: {"mean": scale * signal, "std": 0.01, "n": 3}
                        for arch in ARCHS
                    },
                }
            )
    arms = {
        arm: {"primary_rho_ipc50": 0.5, "secondary_rho_ipc10": 0.4} for arm in ARMS
    }
    return {
        "stage": stage,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "evidence_class": "measured",
        "preregistration_sha256": "0" * 64,
        "cohort": {"architectures": ARCHS, "seeds": SEEDS},
        "rows": rows,
        "oos_predictions": {},
        "arms": arms,
        "random_null": {"reps": 2},
        "bootstrap": {"reps_requested": 6},
        "regret": {},
        "compute_break_even": {},
        "per_family_descriptive": {},
        "wall_time": {},
        "screening_statement": "synthetic",
        "implementation_preregistration_map": [],
        "statistical_uncertainty": {},
    }


def test_pooled_rows_3ds_qualification():
    e1 = _report_fixture("cifar10", "auditbench_e1_report")
    e2 = _report_fixture("cifar100", "auditbench_e2_report")
    e3 = _report_fixture("tiny_imagenet", E3_SINGLE_STAGE)
    rows = pooled_rows_3ds(e1, e2, e3)
    assert len(rows) == 18
    assert all(":" in row["entry_id"] for row in rows)
    clusters = {row["lineage_dataset"] for row in rows}
    assert len(clusters) == 9
    assert {row["dataset"] for row in rows} == {"cifar10", "cifar100", "tiny_imagenet"}


def test_run_e3b_pooled_schema_and_per_dataset_alongside():
    e1 = _report_fixture("cifar10", "auditbench_e1_report", scale=1.0)
    e2 = _report_fixture("cifar100", "auditbench_e2_report", scale=2.0)
    e3 = _report_fixture("tiny_imagenet", E3_SINGLE_STAGE, scale=3.0)
    preregistration = {"statistics": {"screening_statement_template": TEMPLATE}}
    report = run_e3b_pooled(
        e1, e2, e3, preregistration, bootstrap_reps=6, random_reps=2, seed=0, log_every=0,
    )
    validate_pooled_3ds_report(report)
    assert set(report["per_dataset_standalone"]) == {"cifar10", "cifar100", "tiny_imagenet"}
    assert report["screening_statement"] == TEMPLATE.format(n=18, f=3, l=9, d=3)
    for arm in ARMS:
        block = report["arms"][arm]
        assert block["primary_rho_ipc50_pooled"] is not None
        assert set(block["per_dataset"]) == {"cifar10", "cifar100", "tiny_imagenet"}
        for dataset in ("cifar10", "cifar100", "tiny_imagenet"):
            assert block["per_dataset"][dataset]["primary_rho_ipc50"] is not None
    # known monotone signal inside every dataset: perfect primary rho per cell
    assert report["arms"]["base"]["primary_rho_ipc50_pooled"] == pytest.approx(1.0)


def test_oos_2ds_fold_separation_three_datasets():
    e1 = _report_fixture("cifar10", "auditbench_e1_report")
    e2 = _report_fixture("cifar100", "auditbench_e2_report")
    e3 = _report_fixture("tiny_imagenet", E3_SINGLE_STAGE)
    rows = pooled_rows_3ds(e1, e2, e3)
    result = oos_pooled_predictions_2ds(
        rows, ARM_FEATURES["base"], ARCHS, ALPHA_GRID, return_details=True
    )
    long_keys = [(row["entry_id"], arch) for row in rows for arch in ARCHS]
    folds = result["details"]["folds"]
    assert len(folds) == 3 * 2  # 3 pooled families x 2 archs
    for fold in folds:
        train = {long_keys[i] for i in fold["train_index"]}
        test = {long_keys[i] for i in fold["test_index"]}
        assert train.isdisjoint(test), f"leakage in fold {fold['held_family']}"


def test_validate_pooled_3ds_report_catches_missing_dataset():
    e1 = _report_fixture("cifar10", "auditbench_e1_report")
    e2 = _report_fixture("cifar100", "auditbench_e2_report")
    e3 = _report_fixture("tiny_imagenet", E3_SINGLE_STAGE)
    preregistration = {"statistics": {"screening_statement_template": TEMPLATE}}
    report = run_e3b_pooled(
        e1, e2, e3, preregistration, bootstrap_reps=4, random_reps=2, seed=0, log_every=0,
    )
    broken = dict(report)
    broken["per_dataset_standalone"] = {"cifar10": {}, "cifar100": {}}
    with pytest.raises(ValueError, match="tiny_imagenet"):
        validate_pooled_3ds_report(broken)
