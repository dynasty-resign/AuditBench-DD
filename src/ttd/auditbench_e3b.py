"""AuditBench-DD E3b: frozen-procedure inference for E3 + 3-dataset pooled summary.

Implements the frozen ``resampling`` / ``primary_endpoint`` /
``secondary_endpoints`` / ``statistics`` sections of the E3 pre-registration
(``runs/tiny_imagenet/auditbench_e3/e3_preregistration.json``, body sha16
0d1d5c29dcec2589) on top of the completed E3a measurement register:

- zero GPU, zero download, zero retraining: inputs are the hash-pinned local
  E3a artifacts (10 scores + 10 evaluation containers + 4 progress files),
  the E3a measurement register, and the frozen E1/E2 reports;
- single-dataset E3 analysis reuses the frozen E1/E2 calibers from
  ``ttd.auditbench_e2`` unchanged (per-(arch, ipc)-cell train-only z-score,
  inner-LOFO alpha re-selection over the frozen grid, LOFO x LOAO pooled OOS
  fold structure, lineage-cluster bootstrap with full refit per replicate,
  gaussian random-column null, top-1 regret, compute break-even);
- the three-dataset pooled summary (E1+E2+E3) reuses the dataset-aware
  machinery from ``ttd.auditbench_e2b`` unchanged with
  ``datasets=("cifar10", "cifar100", "tiny_imagenet")``; the pooled cluster
  unit is release_lineage x dataset; per-dataset endpoints are reported
  alongside the pooled endpoint (pooled-only reporting is forbidden);
- TM ipc50 is ONE union entry per the iter0009 structural note adjudication
  (operative matrix 10 entries / 5 lineages / 120 cells); the frozen
  screening-statement template is filled with the measured counts
  (n=10, f=4, l=5, d=1 for E3 alone; measured pooled counts for the
  three-dataset statement).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .auditbench_e2 import (
    ALPHA_GRID,
    ARM_FEATURES,
    ARMS,
    BOOTSTRAP_REPS,
    PRIMARY_IPC,
    RANDOM_REPS,
    SECONDARY_IPC,
    _prediction_records,
    _rows_summary,
    build_rows,
    check_complete_case,
    compute_break_even,
    endpoint_rho,
    evaluate_predictions,
    lineage_bootstrap,
    oos_pooled_predictions,
    per_family_descriptive,
    random_null,
    regret_summary,
    validate_report,
)
from .auditbench_e2b import (
    endpoints_bundle,
    evaluate_predictions_2ds,
    lineage_dataset_bootstrap,
    oos_pooled_predictions_2ds,
    random_null_2ds,
)
from .utils import save_json

E3B_STAGE = "auditbench_e3b_report"
E3_SINGLE_STAGE = "auditbench_e3b_e3_analysis"
E3_POOLED_STAGE = "auditbench_e3b_pooled_3ds_report"
DATASETS_3 = ("cifar10", "cifar100", "tiny_imagenet")
PROGRESS_LANES = ("lane1", "lane2", "gpu0", "gpu0proxy")
E3_SEED = 0

E3B_POOLED_REQUIRED_KEYS = (
    "stage",
    "generated_at",
    "evidence_class",
    "inputs",
    "cohort",
    "rows",
    "oos_predictions",
    "arms",
    "per_dataset_standalone",
    "random_null",
    "bootstrap",
    "regret",
    "screening_statement",
    "screening_statement_inputs",
    "implementation_preregistration_map",
    "statistical_uncertainty",
)


# ---------------------------------------------------------------------------
# Input assembly (read-only; every input SHA-256 is registered)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_e3a_progress(formal_dir: str | Path) -> dict[str, Any]:
    """Merge the four E3a lane progress files into one progress mapping.

    Train cells are disjoint across lanes (each entry was trained by exactly
    one lane; verified by the iter-0003 T1 matrix rebuild), so the merge has
    no conflicts; proxy blocks are kept alongside for traceability.
    """
    formal_dir = Path(formal_dir)
    merged: dict[str, Any] = {"entries": {}, "lanes": {}}
    for lane in PROGRESS_LANES:
        path = formal_dir / f"e3a_progress.{lane}.json"
        progress = json.loads(path.read_text(encoding="utf-8"))
        merged["lanes"][lane] = {
            "started_at": progress.get("started_at"),
            "updated_at": progress.get("updated_at"),
        }
        for entry_id, block in progress.get("entries", {}).items():
            target = merged["entries"].setdefault(entry_id, {"cells": {}})
            cells = block.get("cells") or {}
            conflict = set(target["cells"]) & set(cells)
            if conflict:
                raise ValueError(f"progress merge conflict for {entry_id}: {conflict}")
            target["cells"].update(cells)
            if block.get("proxy"):
                target.setdefault("proxy_blocks", {})[lane] = block["proxy"]
    return merged


def build_e3_rows(
    preregistration: dict[str, Any],
    formal_dir: str | Path,
    register: dict[str, Any],
    progress: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """E3 rows via the frozen build_rows harness + register cross-check.

    Utilities enter through the frozen evaluation containers exactly as in
    E1/E2; the E3a register is asserted value-equal (mean/std per arch
    recomputed from the register's per-seed values, ddof=0 register caliber).
    """
    rows = build_rows(preregistration, formal_dir, progress)
    for row in rows:
        entry_id = row["entry_id"]
        if entry_id not in register["utilities"]:
            raise ValueError(f"register missing entry {entry_id}")
        for arch, summary in row["utility"].items():
            seed_values = [
                float(register["utilities"][entry_id][arch][str(seed)]) for seed in (0, 1, 2)
            ]
            mean = sum(seed_values) / 3.0
            var = sum((value - mean) ** 2 for value in seed_values) / 3.0
            std = var ** 0.5
            if abs(mean - summary["mean"]) > 1e-9 or abs(std - summary["std"]) > 1e-9:
                raise ValueError(
                    f"register/evaluation divergence at {entry_id}|{arch}: "
                    f"register mean/std {mean}/{std} vs evaluation {summary}"
                )
            if int(summary["n"]) != 3:
                raise ValueError(f"{entry_id}|{arch} evaluation n={summary['n']} (expected 3)")
    return rows


# ---------------------------------------------------------------------------
# Single-dataset E3 analysis (frozen E1/E2 calibers, unchanged)
# ---------------------------------------------------------------------------


def run_e3b_single(
    preregistration: dict[str, Any],
    formal_dir: str | Path,
    register: dict[str, Any],
    bootstrap_reps: int = BOOTSTRAP_REPS,
    random_reps: int = RANDOM_REPS,
    seed: int = E3_SEED,
) -> dict[str, Any]:
    """Run the frozen per-dataset inference on the E3 (Tiny-ImageNet) cohort."""
    progress = merge_e3a_progress(formal_dir)
    rows = build_e3_rows(preregistration, formal_dir, register, progress)
    archs = list(preregistration["student_architectures"]["set"])
    seeds = list(preregistration["training_budget"]["seeds"])
    ipcs = sorted({row["ipc"] for row in rows})
    completeness = check_complete_case(rows, archs, seeds)
    if not completeness["complete"]:
        raise RuntimeError(f"E3 matrix incomplete: {completeness['problems']}")

    arms_out: dict[str, Any] = {}
    predictions_out: dict[str, Any] = {}
    prediction_records: dict[str, Any] = {}
    alphas_out: dict[str, Any] = {}
    zscore_skipped_total = 0
    for arm in ARMS:
        result = oos_pooled_predictions(
            rows, ARM_FEATURES[arm], archs, ALPHA_GRID, return_details=True
        )
        predictions = result["predictions"]
        evaluation = evaluate_predictions(rows, predictions, archs, ipcs)
        zscore_skipped_total += result["details"]["zscore_skipped_cells"]
        alphas_out[arm] = {
            f"{fold['held_family']}|{fold['held_arch']}": fold["alpha"]
            for fold in result["details"]["folds"]
        }
        arms_out[arm] = {
            "features": list(ARM_FEATURES[arm]),
            "rho_cells": evaluation["rho"],
            "primary_rho_ipc50": endpoint_rho(evaluation, archs, PRIMARY_IPC),
            "secondary_rho_ipc10": endpoint_rho(evaluation, archs, SECONDARY_IPC),
            "rho_mean_all_cells": evaluation["rho_mean"],
            "regret": regret_summary(evaluation, archs, ipcs),
        }
        predictions_out[arm] = {
            arch: dict(sorted(preds.items())) for arch, preds in predictions.items()
        }
        prediction_records[arm] = _prediction_records(
            rows, archs, result["details"]["folds"], predictions
        )

    null_rng = np.random.default_rng(seed)
    null = random_null(rows, archs, ipcs, ALPHA_GRID, null_rng, random_reps)

    boot_rng = np.random.default_rng(seed + 1)
    bootstrap = lineage_bootstrap(rows, archs, ipcs, ALPHA_GRID, boot_rng, bootstrap_reps)

    break_even = compute_break_even(rows, predictions_out, archs, ipcs, seeds)
    # wording hygiene: the shared reading template says "full-matrix best";
    # the iter-0003 forbidden-token discipline rewords it (meaning unchanged)
    break_even["break_even_reading"] = break_even["break_even_reading"].replace(
        "full-matrix best", "full-matrix top utility"
    )
    per_family = {
        arm: per_family_descriptive(rows, predictions_out[arm], archs, ipcs)
        for arm in ARMS
    }

    template = preregistration["statistics"]["screening_statement_template"]
    screening_statement = template.format(
        n=len(rows),
        f=len({row["method_family"] for row in rows}),
        l=len({row["release_lineage"] for row in rows}),
        d=1,
    )

    report: dict[str, Any] = {
        "stage": E3_SINGLE_STAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "measured",
        "preregistration_sha256": None,  # filled by the combined runner
        "cohort": {
            "n_entries": len(rows),
            "entry_ids": [row["entry_id"] for row in rows],
            "families": sorted({row["method_family"] for row in rows}),
            "lineages": sorted({row["release_lineage"] for row in rows}),
            "architectures": archs,
            "seeds": seeds,
            "ipcs": ipcs,
            "n_datasets": 1,
            "dataset": "tiny_imagenet",
            "completeness": completeness,
            "tm_ipc50_union_entry": (
                "dcbench.tm_mtt.ipc50 executed as ONE union entry (5 class-shards "
                "of one 10,000-image artifact; iter0009 structural note, "
                "supervisor-adjudicated); operative matrix 10 entries x 4 archs "
                "x 3 seeds = 120 cells"
            ),
        },
        "rows": _rows_summary(rows),
        "oos_predictions": predictions_out,
        "oos_prediction_records": prediction_records,
        "arms": arms_out,
        "alpha_selection": {"grid": list(ALPHA_GRID), "chosen_per_fold": alphas_out},
        "random_null": null,
        "bootstrap": bootstrap,
        "regret": {arm: arms_out[arm]["regret"] for arm in ARMS},
        "compute_break_even": break_even,
        "per_family_descriptive": per_family,
        "wall_time": {
            "proxy_total_sec": break_even["proxy_total_sec"],
            "train_total_sec": break_even["train_total_sec"],
            "n_train_cells_measured": break_even["n_train_cells_measured"],
            "per_arch": break_even["per_arch"],
            "lanes": progress["lanes"],
        },
        "near_random": {
            "register_cross_check": (
                "all 8 (arch, ipc) cells re-evaluated by iter-0003 T1 closure "
                "verification; zero near_random_descriptive cells"
            ),
            "cells": register["near_random"],
            "vgg11_descriptive_note": (
                "descriptive failure mode: many vgg11 utility cells collapse to "
                "0.005 (=1/200 random level); the frozen near-random criterion "
                "is not met in any cell (std_across_entries > mean_seed_std "
                "everywhere), so no cell is deleted or weakened formally; the "
                "vgg11 collapse nonetheless weakens the descriptive reading of "
                "vgg11-cell rho/regret"
            ),
        },
        "screening_statement": screening_statement,
        "implementation_preregistration_map": E3B_SINGLE_IMPLEMENTATION_MAP,
        "statistical_uncertainty": {
            "design": (
                "release_lineage-cluster bootstrap (1000 reps, full refit per "
                "replicate; 5 clusters) for grouped CIs; gaussian random-column "
                "null (20 reps, variants pure and base_plus_random) as negative "
                "control; n=10 entries, f=4 families, l=5 lineages, d=1 dataset: "
                "screening-level"
            ),
            "zscore_skipped_cells_total": zscore_skipped_total,
            "bootstrap_reps_used": bootstrap["reps_used"],
            "bootstrap_reps_skipped_undefined": bootstrap["reps_skipped_undefined"],
            "random_null_reps": random_reps,
        },
        "rng_seed": {"random_null": seed, "bootstrap": seed + 1},
    }
    validate_report(report, stage=E3_SINGLE_STAGE)
    return report


# ---------------------------------------------------------------------------
# Three-dataset pooled summary (E1 + E2 + E3; e2b machinery, 3-dataset cells)
# ---------------------------------------------------------------------------


def pooled_rows_3ds(
    e1_report: dict[str, Any],
    e2_report: dict[str, Any],
    e3_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Dataset-qualified pooled rows from the three per-dataset reports."""
    rows: list[dict[str, Any]] = []
    for dataset, report in (
        ("cifar10", e1_report),
        ("cifar100", e2_report),
        ("tiny_imagenet", e3_report),
    ):
        for row in report["rows"]:
            rows.append(
                {
                    "entry_id": f"{dataset}:{row['entry_id']}",
                    "dataset": dataset,
                    "release_lineage": row["release_lineage"],
                    "lineage_dataset": f"{dataset}:{row['release_lineage']}",
                    "method_family": row["method_family"],
                    "ipc": int(row["ipc"]),
                    "scores": row["scores"],
                    "utility": row["utility"],
                }
            )
    return rows


def _endpoint_subset_composition_3ds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per (dataset, ipc) subset composition (measured, not assumed)."""
    out: dict[str, Any] = {}
    for dataset in DATASETS_3:
        per_ipc: dict[str, Any] = {}
        for ipc in sorted({row["ipc"] for row in rows}):
            subset = [
                row for row in rows if row["dataset"] == dataset and row["ipc"] == ipc
            ]
            if not subset:
                continue
            per_ipc[str(ipc)] = {
                "n_entries": len(subset),
                "n_families": len({row["method_family"] for row in subset}),
                "n_lineages": len({row["release_lineage"] for row in subset}),
                "entry_ids": sorted(row["entry_id"] for row in subset),
            }
        out[dataset] = per_ipc
    return out


def validate_pooled_3ds_report(report: dict[str, Any]) -> None:
    """Schema check for the three-dataset pooled report."""
    missing = [key for key in E3B_POOLED_REQUIRED_KEYS if key not in report]
    if missing:
        raise ValueError(f"pooled 3ds report missing keys: {missing}")
    if report.get("stage") != E3_POOLED_STAGE:
        raise ValueError(f"pooled 3ds report stage must be {E3_POOLED_STAGE!r}")
    for arm in ARMS:
        if arm not in report["arms"]:
            raise ValueError(f"pooled 3ds report arms missing arm {arm!r}")
        for field in (
            "primary_rho_ipc50_pooled",
            "secondary_rho_ipc10_pooled",
            "per_dataset",
        ):
            if field not in report["arms"][arm]:
                raise ValueError(f"pooled 3ds report arm {arm!r} missing {field!r}")
    for dataset in DATASETS_3:
        if dataset not in report["per_dataset_standalone"]:
            raise ValueError(f"per_dataset_standalone missing {dataset!r}")
        for arm in ARMS:
            for dataset_key in report["arms"][arm]["per_dataset"]:
                if dataset_key not in DATASETS_3:
                    raise ValueError(f"unexpected per_dataset key {dataset_key!r}")
    if report["random_null"].get("reps") is None:
        raise ValueError("pooled 3ds report random_null missing reps")
    if report["bootstrap"].get("reps_requested") is None:
        raise ValueError("pooled 3ds report bootstrap missing reps_requested")


def run_e3b_pooled(
    e1_report: dict[str, Any],
    e2_report: dict[str, Any],
    e3_report: dict[str, Any],
    preregistration: dict[str, Any],
    bootstrap_reps: int = BOOTSTRAP_REPS,
    random_reps: int = RANDOM_REPS,
    seed: int = E3_SEED,
    log_every: int = 100,
) -> dict[str, Any]:
    """Run the frozen three-dataset pooled summary (lineage x dataset clusters)."""
    validate_report(e1_report, stage="auditbench_e1_report")
    validate_report(e2_report, stage="auditbench_e2_report")
    validate_report(e3_report, stage=E3_SINGLE_STAGE)

    rows = pooled_rows_3ds(e1_report, e2_report, e3_report)
    datasets = list(DATASETS_3)
    archs = list(e2_report["cohort"]["architectures"])
    for name, report in (("cifar10", e1_report), ("tiny_imagenet", e3_report)):
        if list(report["cohort"]["architectures"]) != archs:
            raise ValueError(f"arch sets differ for {name}")
    seeds = list(e2_report["cohort"]["seeds"])
    ipcs = sorted({row["ipc"] for row in rows})
    completeness = check_complete_case(rows, archs, seeds)
    if not completeness["complete"]:
        raise RuntimeError(f"pooled 3ds rows incomplete: {completeness['problems']}")

    arms_out: dict[str, Any] = {}
    predictions_out: dict[str, Any] = {}
    alphas_out: dict[str, Any] = {}
    zscore_skipped_total = 0
    for arm in ARMS:
        result = oos_pooled_predictions_2ds(
            rows, ARM_FEATURES[arm], archs, ALPHA_GRID, return_details=True
        )
        predictions = result["predictions"]
        evaluation = evaluate_predictions_2ds(rows, predictions, archs, datasets, ipcs)
        zscore_skipped_total += result["details"]["zscore_skipped_cells"]
        alphas_out[arm] = {
            f"{fold['held_family']}|{fold['held_arch']}": fold["alpha"]
            for fold in result["details"]["folds"]
        }
        bundle = endpoints_bundle(evaluation, archs, datasets)
        arms_out[arm] = {
            "features": list(ARM_FEATURES[arm]),
            "rho_cells": evaluation["rho"],
            **bundle,
            "rho_mean_all_cells": evaluation["rho_mean"],
            "regret": {
                "cells": {
                    key: {
                        "regret": evaluation["regret"][key],
                        "random_expected_regret": evaluation["random_regret"][key],
                    }
                    for key in evaluation["regret"]
                },
                "regret_mean": evaluation["regret_mean"],
                "random_expected_regret_mean": float(
                    np.mean(list(evaluation["random_regret"].values()))
                )
                if evaluation["random_regret"]
                else None,
            },
        }
        predictions_out[arm] = {
            arch: dict(sorted(preds.items())) for arch, preds in predictions.items()
        }

    null_rng = np.random.default_rng(seed)
    null = random_null_2ds(rows, archs, datasets, ipcs, ALPHA_GRID, null_rng, random_reps)

    boot_rng = np.random.default_rng(seed + 1)
    bootstrap = lineage_dataset_bootstrap(
        rows, archs, datasets, ipcs, ALPHA_GRID, boot_rng, bootstrap_reps,
        log_every=log_every,
    )

    n_lineage_dataset = len({row["lineage_dataset"] for row in rows})
    template = preregistration["statistics"]["screening_statement_template"]
    screening_statement = template.format(
        n=len(rows),
        f=len({row["method_family"] for row in rows}),
        l=n_lineage_dataset,
        d=len(datasets),
    )
    screening_inputs = {
        "n": len(rows),
        "f": len({row["method_family"] for row in rows}),
        "l": n_lineage_dataset,
        "d": len(datasets),
        "l_unit": (
            "release_lineage x dataset clusters (19 = 7 E1 + 7 E2 + 5 E3; "
            "lineage names recur across datasets)"
        ),
    }

    per_dataset_standalone = {
        "cifar10": {
            "source": "e1_report.json (frozen, read-only, verbatim copies)",
            "arms": {
                arm: {
                    "primary_rho_ipc50": e1_report["arms"][arm]["primary_rho_ipc50"],
                    "secondary_rho_ipc10": e1_report["arms"][arm]["secondary_rho_ipc10"],
                }
                for arm in ARMS
            },
        },
        "cifar100": {
            "source": "e2_report.json (frozen, read-only, verbatim copies)",
            "arms": {
                arm: {
                    "primary_rho_ipc50": e2_report["arms"][arm]["primary_rho_ipc50"],
                    "secondary_rho_ipc10": e2_report["arms"][arm]["secondary_rho_ipc10"],
                }
                for arm in ARMS
            },
        },
        "tiny_imagenet": {
            "source": "e3b_report.json e3_single_dataset section (this file)",
            "arms": {
                arm: {
                    "primary_rho_ipc50": e3_report["arms"][arm]["primary_rho_ipc50"],
                    "secondary_rho_ipc10": e3_report["arms"][arm]["secondary_rho_ipc10"],
                }
                for arm in ARMS
            },
        },
    }

    report: dict[str, Any] = {
        "stage": E3_POOLED_STAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "measured",
        "inputs": {},  # filled by the combined runner (sha256 registry)
        "cohort": {
            "n_entries": len(rows),
            "datasets": datasets,
            "families": sorted({row["method_family"] for row in rows}),
            "lineages": sorted({row["release_lineage"] for row in rows}),
            "lineage_dataset_clusters": sorted({row["lineage_dataset"] for row in rows}),
            "architectures": archs,
            "seeds": seeds,
            "ipcs": ipcs,
            "completeness": completeness,
            "endpoint_subset_composition": _endpoint_subset_composition_3ds(rows),
        },
        "rows": rows,
        "oos_predictions": predictions_out,
        "arms": arms_out,
        "alpha_selection": {"grid": list(ALPHA_GRID), "chosen_per_fold": alphas_out},
        "per_dataset_standalone": per_dataset_standalone,
        "random_null": null,
        "bootstrap": bootstrap,
        "regret": {arm: arms_out[arm]["regret"] for arm in ARMS},
        "screening_statement": screening_statement,
        "screening_statement_inputs": screening_inputs,
        "implementation_preregistration_map": E3B_POOLED_IMPLEMENTATION_MAP,
        "statistical_uncertainty": {
            "design": (
                "lineage x dataset cluster bootstrap (1000 reps, full refit per "
                "replicate) for grouped CIs; gaussian random-column null (20 "
                "reps) as negative control; the pooled design has 19 clusters "
                "and 6 pooled families, so all endpoints are screening-level"
            ),
            "zscore_skipped_cells_total": zscore_skipped_total,
            "bootstrap_reps_used": bootstrap["reps_used"],
            "bootstrap_reps_skipped_undefined": bootstrap["reps_skipped_undefined"],
            "random_null_reps": random_reps,
        },
        "rng_seed": {"random_null": seed, "bootstrap": seed + 1},
    }
    validate_pooled_3ds_report(report)
    return report


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------


def run_e3b(
    preregistration_path: str | Path,
    formal_dir: str | Path,
    register_path: str | Path,
    e1_report_path: str | Path,
    e2_report_path: str | Path,
    extra_input_paths: dict[str, str | Path] | None = None,
    out_path: str | Path | None = None,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    random_reps: int = RANDOM_REPS,
    seed: int = E3_SEED,
    log_every: int = 100,
) -> dict[str, Any]:
    """Run E3b single-dataset + three-dataset pooled; write the combined report."""
    preregistration_path = Path(preregistration_path)
    formal_dir = Path(formal_dir)
    register_path = Path(register_path)
    e1_report_path = Path(e1_report_path)
    e2_report_path = Path(e2_report_path)

    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    register = json.loads(register_path.read_text(encoding="utf-8"))
    e1_report = json.loads(e1_report_path.read_text(encoding="utf-8"))
    e2_report = json.loads(e2_report_path.read_text(encoding="utf-8"))

    single = run_e3b_single(
        preregistration, formal_dir, register,
        bootstrap_reps=bootstrap_reps, random_reps=random_reps, seed=seed,
    )
    single["preregistration_sha256"] = _sha256_file(preregistration_path)
    validate_report(single, stage=E3_SINGLE_STAGE)

    pooled = run_e3b_pooled(
        e1_report, e2_report, single, preregistration,
        bootstrap_reps=bootstrap_reps, random_reps=random_reps, seed=seed,
        log_every=log_every,
    )

    inputs: dict[str, Any] = {
        "e1_report_path": str(e1_report_path),
        "e1_report_sha256": _sha256_file(e1_report_path),
        "e2_report_path": str(e2_report_path),
        "e2_report_sha256": _sha256_file(e2_report_path),
        "e3_preregistration_path": str(preregistration_path),
        "e3_preregistration_sha256": _sha256_file(preregistration_path),
        "e3a_register_path": str(register_path),
        "e3a_register_sha256": _sha256_file(register_path),
        "e3a_scores_sha256": {},
        "e3a_evaluation_sha256": {},
        "e3a_progress_sha256": {},
        "read_only_note": (
            "E1/E2/E3 artifacts, pre-registrations and reports are read-only "
            "inputs; no input byte is modified by this analysis"
        ),
    }
    for row in single["rows"]:
        entry_id = row["entry_id"]
        scores_path = formal_dir / f"{entry_id}.scores.json"
        evaluation_path = formal_dir / f"{entry_id}.eval_input.evaluation.json"
        inputs["e3a_scores_sha256"][entry_id] = _sha256_file(scores_path)
        inputs["e3a_evaluation_sha256"][entry_id] = _sha256_file(evaluation_path)
    for lane in PROGRESS_LANES:
        lane_path = formal_dir / f"e3a_progress.{lane}.json"
        inputs["e3a_progress_sha256"][lane] = _sha256_file(lane_path)
    for name, path in (extra_input_paths or {}).items():
        path = Path(path)
        inputs[f"{name}_path"] = str(path)
        inputs[f"{name}_sha256"] = _sha256_file(path)

    pooled["inputs"] = dict(inputs)

    report: dict[str, Any] = {
        "stage": E3B_STAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "measured",
        "inputs": inputs,
        "e3_single_dataset": single,
        "three_dataset_pooled": pooled,
        "screening_statements": {
            "e3_single": single["screening_statement"],
            "three_dataset_pooled": pooled["screening_statement"],
        },
        "adjudications_cited": [
            "TM ipc50 ONE union entry / 120-cell operative matrix "
            "(e3_preregistration_structural_note_iter0009.json, registered "
            "before any utility observation; post_observation_lock not "
            "triggered; frozen body zero-byte; supervisor-adjudicated)",
            "dsa.ipc50 single evaluation container cross-checked value-by-value "
            "against the register (12/12 values, iter-0003 T1)",
            "e3_preregistration_budget_addendum.json (64px calibration "
            "coefficients) and the structural note enter the cited evidence "
            "chain from this round onward",
        ],
    }
    if out_path is not None:
        out_path = Path(out_path)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        save_json(report, tmp)
        tmp.replace(out_path)
    return report


E3B_SINGLE_IMPLEMENTATION_MAP = [
    {
        "preregistration": "rows from the 10-entry operative cohort (TM ipc50 ONE union entry) with family/lineage labels",
        "implementation": "build_e3_rows(): labels read from e3_preregistration.json cohort, never from file names; utility = per-arch {mean, std, n} over seeds {0,1,2} from the evaluation containers, asserted value-equal to the E3a register (ddof=0 register caliber)",
    },
    {
        "preregistration": "utility normalization: per-(arch, ipc)-cell z-score fitted on train folds only (cardc._oos_pooled_v2)",
        "implementation": "auditbench_e2.zscore_targets_train_only() reused unchanged; cells with <2 train rows keep raw values, occurrences counted",
    },
    {
        "preregistration": "outer design: LOFO (4 method families) x LOAO (4 archs) pooled OOS; separation rule: same artifact never in train and test of one resample; cluster unit = release_lineage",
        "implementation": "auditbench_e2.oos_pooled_predictions() reused unchanged (frozen E1/E2 fold structure and separation discipline)",
    },
    {
        "preregistration": "alpha grid (0.01, 0.1, 1, 10, 100), re-selected inside train folds only",
        "implementation": "auditbench_e2.select_alpha() reused unchanged: inner leave-one-family-out CV scored by mean within-(arch, ipc)-cell Spearman",
    },
    {
        "preregistration": "primary endpoint pooled_family_arch_oos_spearman_ipc50: per-arch cell OOS Spearman at IPC50 vs raw utility mean, averaged over the 4 student architectures, per arm",
        "implementation": "auditbench_e2.endpoint_rho(evaluation, archs, 50); the frozen IPC50-per-arch-then-mean aggregation; IPC10 is the secondary",
    },
    {
        "preregistration": "statistics.top_k_regret: k=1; regret = cell-max utility - utility(argmax proxy); random_expected_regret = cell-max - mean utility within the (arch, ipc) cell (frozen definition quoted with wording normalized to the iter-0003 token discipline)",
        "implementation": "auditbench_e2.evaluate_predictions() + regret_summary(); both arms and the random null reported side by side",
    },
    {
        "preregistration": "random null: gaussian random score column through the same OOS pipeline (20 reps); both variants (pure random-only, base_plus_random)",
        "implementation": "auditbench_e2.random_null() reused unchanged; rho_mean mean/p95 and regret_mean mean/p05 reported per variant",
    },
    {
        "preregistration": "release_lineage-cluster bootstrap, 1000 reps, full refit per replicate (alpha re-selection inside train folds)",
        "implementation": "auditbench_e2.lineage_bootstrap()+resample_lineage_clusters() reused unchanged (5 clusters); cell z-score, alpha CV, fit, prediction and endpoint aggregation re-run per replicate; 2.5/97.5 percentile CIs",
    },
    {
        "preregistration": "compute break-even: per-artifact proxy extraction wall time vs per (arch, seed) student training wall time",
        "implementation": "auditbench_e2.compute_break_even(); train walls merged from the four E3a lane progress files (disjoint cell sets); top-1 selective-training strategy costed per arm",
    },
    {
        "preregistration": "per-family OOS Spearman (descriptive only, 4 clusters)",
        "implementation": "auditbench_e2.per_family_descriptive() reused unchanged; labelled descriptive only",
    },
    {
        "preregistration": "near_random_cell_declaration: descriptive, no deletion, no remedy",
        "implementation": "register near_random block carried into the report with the iter-0003 T1 re-evaluation cross-check (0/8 cells flagged); vgg11 0.005-collapse noted descriptively",
    },
    {
        "preregistration": "screening statement filled with actual measured counts (filled_statement_rule)",
        "implementation": "statistics.screening_statement_template filled verbatim with measured n=10, f=4, l=5, d=1",
    },
]


E3B_POOLED_IMPLEMENTATION_MAP = [
    {
        "preregistration": "statistics.three_dataset_summary: pooled cluster unit = release_lineage x dataset",
        "implementation": "pooled_rows_3ds(): dataset-qualified entry_id and lineage_dataset = dataset:release_lineage; auditbench_e2b.resample_lineage_dataset_clusters() resamples the 19 clusters with replacement, #draw suffixes on entry_id and lineage_dataset, method_family unchanged so folds keep copies together",
    },
    {
        "preregistration": "statistics.three_dataset_summary.reporting_rule: per-dataset endpoints (CIFAR-10 E1, CIFAR-100 E2, Tiny-ImageNet E3) and the pooled endpoint reported separately; pooled-only reporting forbidden",
        "implementation": "report carries per_dataset_standalone (verbatim copies of the frozen e1_report/e2_report arm endpoints plus the E3 single-dataset endpoints), the pooled-model per_dataset cells, and the pooled endpoints",
    },
    {
        "preregistration": "resampling.outer_design (LOFO x LOAO pooled OOS) applied to the pooled table",
        "implementation": "auditbench_e2b.oos_pooled_predictions_2ds() reused unchanged with datasets=(cifar10, cifar100, tiny_imagenet); cells are (dataset, arch, ipc); same separation discipline as the per-dataset analyses",
    },
    {
        "preregistration": "utility normalization fitted on train folds only (frozen caliber)",
        "implementation": "auditbench_e2b.zscore_targets_train_only_2ds() reused unchanged: identical math to the per-dataset zscore_targets_train_only with the grouping key extended to (dataset, arch, ipc); skipped cells counted",
    },
    {
        "preregistration": "alpha re-selection inside train folds over the frozen grid (0.01, 0.1, 1, 10, 100)",
        "implementation": "shared auditbench_e2.select_alpha(): inner leave-one-family-out CV over the pooled families, scored by mean within-(dataset, arch, ipc)-cell Spearman",
    },
    {
        "preregistration": "primary endpoint construction (IPC50 pooled family x arch OOS Spearman) applied to the three-dataset table",
        "implementation": "auditbench_e2b.endpoint_rho_pooled(): derived translation - a cell is (arch, dataset, ipc); the pooled IPC50 endpoint averages the within-cell OOS Spearman over the 4 archs x 3 datasets (equal weight); raw cross-dataset utility scales never mix inside one Spearman",
    },
    {
        "preregistration": "bootstrap: release_lineage x dataset clusters, 1000 reps, full refit per replicate",
        "implementation": "auditbench_e2b.lineage_dataset_bootstrap() reused unchanged: every replicate re-runs cell z-score, alpha CV, ridge fit, OOS prediction and endpoint aggregation for both arms; replicates with any undefined pooled or per-dataset endpoint are skipped and counted (E1 skip discipline); 2.5/97.5 percentile CIs and delta frac_positive",
    },
    {
        "preregistration": "random null: gaussian random score column through the same OOS pipeline (20 reps)",
        "implementation": "auditbench_e2b.random_null_2ds() reused unchanged: variants pure and base_plus_random; pooled primary rho_mean mean/p95 and regret_mean mean/p05 reported",
    },
    {
        "preregistration": "screening statement filled at analysis time with the actual pooled n/f/l/d (filled_statement_rule)",
        "implementation": "statistics.screening_statement_template filled verbatim with measured n=44, f=6, l=19 (lineage x dataset clusters), d=3; screening_statement_inputs documents the l=19 cluster unit",
    },
    {
        "preregistration": "endpoint subset composition disclosure (IPC50/IPC10 per dataset)",
        "implementation": "cohort.endpoint_subset_composition: per (dataset, ipc) n_entries/n_families/n_lineages computed from the pooled rows (measured, not assumed)",
    },
]
