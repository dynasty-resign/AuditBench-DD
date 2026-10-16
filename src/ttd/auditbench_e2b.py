"""AuditBench-DD E2b: two-dataset pooled summary (E1 CIFAR-10 + E2 CIFAR-100).

Implements the frozen ``statistics.two_dataset_summary`` section of the E2
pre-registration (``runs/cifar100/auditbench_e2/e2_preregistration.json``)
on top of the two already-frozen per-dataset analyses:

- inputs are the on-disk ``e1_report.json`` (read-only) and ``e2_report.json``
  rows (scores + utility); no E1 artifact, pre-registration or report byte is
  touched, and no model is retrained - the pooled statistics refit only the
  ridge proxy model on the stored proxy scores;
- pooled cluster unit is ``release_lineage x dataset`` (bootstrap resamples
  the 14 clusters with replacement, full refit per replicate);
- the outer design stays LOFO (pooled method families) x LOAO (4 student
  architectures); utility z-scores are per-(dataset, arch, ipc) cell fitted
  on train folds only; alpha is re-selected inside every train fold by inner
  leave-one-family-out CV over the frozen grid;
- per-dataset endpoints (standalone, copied verbatim from the two input
  reports) and the pooled endpoint are reported separately; pooled-only
  reporting is forbidden by the pre-registration;
- the pooled endpoint translation (derived, registered here and in the
  report's implementation map): a cell is ``(arch, dataset, ipc)``; the
  pooled IPC50 endpoint is the within-cell OOS Spearman averaged over the
  4 archs x 2 datasets, which preserves the frozen per-arch-then-mean
  aggregation with dataset as the stratification dimension and never mixes
  raw cross-dataset utility scales inside one Spearman.

The E1/E2 statistical calibers are reused from ``ttd.auditbench_e2``
(``select_alpha``, ``_spearman`` orientation, random-null variants,
#draw-suffix bootstrap discipline); only the dataset-aware pieces are new.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audit import _oriented, _spearman
from .auditbench_e2 import (
    ALPHA_GRID,
    ARM_FEATURES,
    ARMS,
    PRIMARY_IPC,
    SECONDARY_IPC,
    check_complete_case,
    select_alpha,
    validate_report,
)
from .utils import save_json

E2B_STAGE = "auditbench_e2b_pooled_report"
DATASETS = ("cifar10", "cifar100")

POOLED_REPORT_REQUIRED_KEYS = (
    "stage",
    "generated_at",
    "evidence_class",
    "inputs",
    "cohort",
    "rows",
    "oos_predictions",
    "oos_prediction_records",
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
# Row construction from the two frozen per-dataset reports
# ---------------------------------------------------------------------------


def pooled_rows_from_reports(
    e1_report: dict[str, Any], e2_report: dict[str, Any]
) -> list[dict[str, Any]]:
    """Dataset-qualified rows from the stored report rows (read-only inputs)."""
    rows: list[dict[str, Any]] = []
    for dataset, report in (("cifar10", e1_report), ("cifar100", e2_report)):
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


# ---------------------------------------------------------------------------
# Dataset-aware OOS machinery (mirrors auditbench_e2 calibers)
# ---------------------------------------------------------------------------


def _long_table_2ds(
    rows: list[dict[str, Any]],
    names: tuple[str, ...] | list[str],
    archs: list[str],
    extra_column: np.ndarray | None = None,
) -> dict[str, Any]:
    """Entry-major long table with dataset-aware (dataset, arch, ipc) cells."""
    long_rows = [(row, arch) for row in rows for arch in archs]
    features = np.array(
        [[_oriented(row["scores"], name) for name in names] for row, _ in long_rows],
        dtype=float,
    ).reshape(len(long_rows), len(names))
    if extra_column is not None:
        features = np.concatenate([features, extra_column.reshape(-1, 1)], axis=1)
    return {
        "rows": long_rows,
        "features": features,
        "targets": np.array(
            [row["utility"][arch]["mean"] for row, arch in long_rows], dtype=float
        ),
        "families": np.array([row["method_family"] for row, _ in long_rows]),
        "archs": np.array([arch for _, arch in long_rows]),
        "cells": np.array(
            [f"{row['dataset']}|{arch}|{row['ipc']}" for row, arch in long_rows]
        ),
    }


def zscore_targets_train_only_2ds(
    raw_targets: np.ndarray,
    cell_keys: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Per-cell z-score fitted on train rows only (dataset-aware cell keys).

    Cells with fewer than 2 train rows keep raw values; occurrences counted.
    Identical caliber to auditbench_e2.zscore_targets_train_only with the
    (arch, ipc) grouping replaced by the (dataset, arch, ipc) cell key.
    """
    targets = raw_targets.copy()
    skipped = 0
    for cell in np.unique(cell_keys):
        cell_mask = cell_keys == cell
        cell_train = cell_mask & train_mask
        if cell_train.sum() < 2:
            skipped += 1
            continue
        mean = float(raw_targets[cell_train].mean())
        std = max(float(raw_targets[cell_train].std()), 1e-8)
        targets[cell_mask] = (raw_targets[cell_mask] - mean) / std
    return targets, skipped


def oos_pooled_predictions_2ds(
    rows: list[dict[str, Any]],
    names: tuple[str, ...] | list[str],
    archs: list[str],
    alphas,
    extra_column: np.ndarray | None = None,
    return_details: bool = False,
) -> dict[str, Any]:
    """LOFO (pooled method families) x LOAO (archs) OOS, E1/E2 fold caliber.

    Same separation discipline as the per-dataset analyses: for each
    (held_family, held_arch) fold the train set is every row whose family is
    not held and whose architecture is not held, so the same
    (dataset, entry, arch) never appears in train and test of one fold.
    Utility z-scores are per-(dataset, arch, ipc) cell fitted on train rows
    only; alpha is re-selected by inner leave-one-family-out CV (the shared
    ``select_alpha``) inside the train fold.
    """
    table = _long_table_2ds(rows, names, archs, extra_column=extra_column)
    features = table["features"]
    raw_targets = table["targets"]
    families = table["families"]
    arch_arr = table["archs"]
    cells = table["cells"]
    predictions: dict[str, dict[str, float]] = {arch: {} for arch in archs}
    details: dict[str, Any] = {"folds": [], "zscore_skipped_cells": 0}
    for held_family in sorted(np.unique(families)):
        for held_arch in archs:
            test_mask = (families == held_family) & (arch_arr == held_arch)
            if not test_mask.any():
                continue
            train_mask = (families != held_family) & (arch_arr != held_arch)
            if not train_mask.any():
                # degenerate resample (single-cluster bootstrap replicate):
                # no train rows exist for this fold, so it contributes no
                # predictions; the endpoint is then undefined and the
                # replicate is skipped and counted by the bootstrap driver.
                continue
            targets, skipped = zscore_targets_train_only_2ds(
                raw_targets, cells, train_mask
            )
            details["zscore_skipped_cells"] += skipped
            alpha = select_alpha(
                features[train_mask],
                targets[train_mask],
                families[train_mask],
                cells[train_mask],
                alphas,
            )
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[train_mask], targets[train_mask])
            fold_predictions = model.predict(features[test_mask])
            test_rows = [pair for pair, keep in zip(table["rows"], test_mask) if keep]
            for (row, _), pred in zip(test_rows, fold_predictions):
                predictions[held_arch][row["entry_id"]] = float(pred)
            if return_details:
                details["folds"].append(
                    {
                        "held_family": str(held_family),
                        "held_arch": held_arch,
                        "alpha": alpha,
                        "train_index": sorted(np.flatnonzero(train_mask).tolist()),
                        "test_index": sorted(np.flatnonzero(test_mask).tolist()),
                    }
                )
    result: dict[str, Any] = {"predictions": predictions}
    if return_details:
        result["details"] = details
    else:
        result["details"] = {"zscore_skipped_cells": details["zscore_skipped_cells"]}
    return result

# ---------------------------------------------------------------------------
# Endpoints (dataset-aware cells; per-dataset and pooled reported separately)
# ---------------------------------------------------------------------------


def evaluate_predictions_2ds(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, float]],
    archs: list[str],
    datasets: tuple[str, ...] | list[str],
    ipcs: list[int],
) -> dict[str, Any]:
    """Per-(dataset, arch, ipc) cell Spearman and top-1 regret vs raw utility."""
    out: dict[str, Any] = {"rho": {}, "regret": {}, "random_regret": {}}
    for dataset in datasets:
        for arch in archs:
            preds = predictions.get(arch, {})
            for ipc in ipcs:
                cell_key = f"{dataset}|{arch}|{ipc}"
                cell_rows = [
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["ipc"] == ipc
                ]
                valid = [row for row in cell_rows if row["entry_id"] in preds]
                if len(valid) < 3:
                    continue
                utilities = np.array(
                    [row["utility"][arch]["mean"] for row in valid], dtype=float
                )
                values = np.array([preds[row["entry_id"]] for row in valid])
                rho = _spearman(values, utilities)
                if rho is not None:
                    out["rho"][cell_key] = rho
                if utilities.std() > 0:
                    chosen = int(np.argmax(values))
                    best = float(utilities.max())
                    out["regret"][cell_key] = best - float(utilities[chosen])
                    out["random_regret"][cell_key] = best - float(utilities.mean())
    out["rho_mean"] = (
        float(np.mean(list(out["rho"].values()))) if out["rho"] else None
    )
    out["regret_mean"] = (
        float(np.mean(list(out["regret"].values()))) if out["regret"] else None
    )
    return out


def endpoint_rho_per_dataset(
    evaluation: dict[str, Any], archs: list[str], dataset: str, ipc: int
) -> float | None:
    """Per-arch cell Spearman at ``ipc`` within one dataset, then arch mean."""
    values = [
        evaluation["rho"].get(f"{dataset}|{arch}|{ipc}")
        for arch in archs
        if evaluation["rho"].get(f"{dataset}|{arch}|{ipc}") is not None
    ]
    return float(np.mean(values)) if values else None


def endpoint_rho_pooled(
    evaluation: dict[str, Any],
    archs: list[str],
    datasets: tuple[str, ...] | list[str],
    ipc: int,
) -> float | None:
    """Pooled endpoint: within-cell Spearman averaged over (arch, dataset).

    Derived translation of the frozen per-arch-then-mean aggregation to the
    two-dataset cell structure: each of the 4 archs x 2 datasets cells enters
    with equal weight, identical to the per-arch mean of per-dataset per-arch
    means; raw cross-dataset utility scales never mix inside one Spearman.
    """
    values = [
        evaluation["rho"].get(f"{dataset}|{arch}|{ipc}")
        for dataset in datasets
        for arch in archs
        if evaluation["rho"].get(f"{dataset}|{arch}|{ipc}") is not None
    ]
    return float(np.mean(values)) if values else None


def endpoints_bundle(
    evaluation: dict[str, Any],
    archs: list[str],
    datasets: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Primary/secondary endpoints, pooled and per-dataset, from one model."""
    return {
        "primary_rho_ipc50_pooled": endpoint_rho_pooled(
            evaluation, archs, datasets, PRIMARY_IPC
        ),
        "secondary_rho_ipc10_pooled": endpoint_rho_pooled(
            evaluation, archs, datasets, SECONDARY_IPC
        ),
        "per_dataset": {
            dataset: {
                "primary_rho_ipc50": endpoint_rho_per_dataset(
                    evaluation, archs, dataset, PRIMARY_IPC
                ),
                "secondary_rho_ipc10": endpoint_rho_per_dataset(
                    evaluation, archs, dataset, SECONDARY_IPC
                ),
            }
            for dataset in datasets
        },
    }


# ---------------------------------------------------------------------------
# Random null (negative control), same two variants as the per-dataset runs
# ---------------------------------------------------------------------------


def _distribution_summary(values: list[float], tail: str) -> dict[str, Any]:
    if not values:
        return {"mean": None, tail: None, "n": 0}
    percentile = 95 if tail == "p95" else 5
    return {
        "mean": float(np.mean(values)),
        tail: float(np.percentile(values, percentile)),
        "n": len(values),
    }


def random_null_2ds(
    rows: list[dict[str, Any]],
    archs: list[str],
    datasets: tuple[str, ...] | list[str],
    ipcs: list[int],
    alphas,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    """Gaussian random column through the same pooled OOS pipeline."""
    records = {
        variant: {"rho_mean": [], "regret_mean": [], "pooled_primary": []}
        for variant in ("pure", "base_plus_random")
    }
    for _ in range(reps):
        column = rng.normal(size=len(rows))
        long_column = np.repeat(column, len(archs))
        for variant, names in (
            ("pure", ()),
            ("base_plus_random", ARM_FEATURES["base"]),
        ):
            result = oos_pooled_predictions_2ds(
                rows, names, archs, alphas, extra_column=long_column
            )
            evaluation = evaluate_predictions_2ds(
                rows, result["predictions"], archs, datasets, ipcs
            )
            if evaluation["rho_mean"] is not None:
                records[variant]["rho_mean"].append(evaluation["rho_mean"])
            if evaluation["regret_mean"] is not None:
                records[variant]["regret_mean"].append(evaluation["regret_mean"])
            pooled = endpoint_rho_pooled(evaluation, archs, datasets, PRIMARY_IPC)
            if pooled is not None:
                records[variant]["pooled_primary"].append(pooled)
    summary: dict[str, Any] = {"reps": reps, "variants": {}}
    for variant, values in records.items():
        summary["variants"][variant] = {
            "rho_mean": _distribution_summary(values["rho_mean"], "p95"),
            "regret_mean": _distribution_summary(values["regret_mean"], "p05"),
            "pooled_primary": _distribution_summary(values["pooled_primary"], "p95"),
        }
    return summary

# ---------------------------------------------------------------------------
# lineage x dataset cluster bootstrap (full refit per replicate)
# ---------------------------------------------------------------------------


def resample_lineage_dataset_clusters(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Resample (release_lineage x dataset) clusters with replacement.

    Copies get a ``#draw`` suffix on entry_id and lineage_dataset while
    method_family stays unchanged, so LOFO folds keep every copy of an
    artifact on the same side of a fold (the E1 #draw discipline).
    """
    clusters = sorted({row["lineage_dataset"] for row in rows})
    by_cluster = {
        cluster: [row for row in rows if row["lineage_dataset"] == cluster]
        for cluster in clusters
    }
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    counts = {cluster: int((sampled == cluster).sum()) for cluster in clusters}
    pseudo: list[dict[str, Any]] = []
    for draw, cluster in enumerate(sampled):
        for row in by_cluster[cluster]:
            pseudo.append(
                dict(
                    row,
                    entry_id=f"{row['entry_id']}#{draw}",
                    lineage_dataset=f"{row['lineage_dataset']}#{draw}",
                )
            )
    return pseudo, counts


def lineage_dataset_bootstrap(
    rows: list[dict[str, Any]],
    archs: list[str],
    datasets: tuple[str, ...] | list[str],
    ipcs: list[int],
    alphas,
    rng: np.random.Generator,
    reps: int,
    log_every: int = 100,
) -> dict[str, Any]:
    """lineage x dataset cluster bootstrap with a full refit per replicate.

    Every replicate re-runs: per-(dataset, arch, ipc)-cell train-only
    z-score, inner-CV alpha selection, ridge fit, OOS prediction and the
    endpoint aggregation, for both arms. Replicates where any arm's pooled
    or per-dataset primary/secondary endpoint is undefined are skipped and
    counted (the E1 skip discipline).
    """
    per_arm = {
        arm: {
            "pooled_primary": [],
            "pooled_secondary": [],
            "per_dataset": {
                dataset: {"primary": [], "secondary": []} for dataset in datasets
            },
        }
        for arm in ARMS
    }
    delta = {"primary": [], "secondary": []}
    skipped = 0
    for rep in range(reps):
        pseudo, _counts = resample_lineage_dataset_clusters(rows, rng)
        endpoints: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            result = oos_pooled_predictions_2ds(
                pseudo, ARM_FEATURES[arm], archs, alphas
            )
            evaluation = evaluate_predictions_2ds(
                pseudo, result["predictions"], archs, datasets, ipcs
            )
            endpoints[arm] = endpoints_bundle(evaluation, archs, datasets)
        definable = True
        for arm in ARMS:
            bundle = endpoints[arm]
            values = [
                bundle["primary_rho_ipc50_pooled"],
                bundle["secondary_rho_ipc10_pooled"],
            ] + [
                bundle["per_dataset"][dataset][key]
                for dataset in datasets
                for key in ("primary_rho_ipc50", "secondary_rho_ipc10")
            ]
            if any(value is None for value in values):
                definable = False
        if not definable:
            skipped += 1
            continue
        for arm in ARMS:
            bundle = endpoints[arm]
            per_arm[arm]["pooled_primary"].append(bundle["primary_rho_ipc50_pooled"])
            per_arm[arm]["pooled_secondary"].append(
                bundle["secondary_rho_ipc10_pooled"]
            )
            for dataset in datasets:
                per_arm[arm]["per_dataset"][dataset]["primary"].append(
                    bundle["per_dataset"][dataset]["primary_rho_ipc50"]
                )
                per_arm[arm]["per_dataset"][dataset]["secondary"].append(
                    bundle["per_dataset"][dataset]["secondary_rho_ipc10"]
                )
        for key, pooled_key in (
            ("primary", "primary_rho_ipc50_pooled"),
            ("secondary", "secondary_rho_ipc10_pooled"),
        ):
            delta[key].append(
                endpoints["base_plus_topo"][pooled_key]
                - endpoints["base"][pooled_key]
            )
        if log_every and (rep + 1) % log_every == 0:
            print(f"[e2b-bootstrap] {rep + 1}/{reps} replicates", flush=True)

    def _ci(values: list[float]) -> dict[str, Any]:
        if len(values) < 10:
            return {
                "mean": float(np.mean(values)) if values else None,
                "ci95": [None, None],
                "n": len(values),
            }
        return {
            "mean": float(np.mean(values)),
            "ci95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
            "n": len(values),
        }

    return {
        "cluster_unit": "release_lineage x dataset",
        "reps_requested": reps,
        "reps_used": len(delta["primary"]),
        "reps_skipped_undefined": skipped,
        "arms": {
            arm: {
                "pooled_primary": _ci(per_arm[arm]["pooled_primary"]),
                "pooled_secondary": _ci(per_arm[arm]["pooled_secondary"]),
                "per_dataset": {
                    dataset: {
                        "primary": _ci(per_arm[arm]["per_dataset"][dataset]["primary"]),
                        "secondary": _ci(
                            per_arm[arm]["per_dataset"][dataset]["secondary"]
                        ),
                    }
                    for dataset in datasets
                },
            }
            for arm in ARMS
        },
        "delta_base_plus_topo_minus_base": {
            key: {
                **_ci(values),
                "frac_positive": (
                    float(np.mean([value > 0 for value in values])) if values else None
                ),
            }
            for key, values in delta.items()
        },
    }

# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def validate_pooled_report(report: dict[str, Any]) -> None:
    """Schema check for the machine-readable two-dataset pooled report."""
    missing = [key for key in POOLED_REPORT_REQUIRED_KEYS if key not in report]
    if missing:
        raise ValueError(f"pooled report missing keys: {missing}")
    if report.get("stage") != E2B_STAGE:
        raise ValueError(f"pooled report stage must be {E2B_STAGE!r}")
    for arm in ARMS:
        if arm not in report["arms"]:
            raise ValueError(f"pooled report arms missing arm {arm!r}")
        for field in (
            "primary_rho_ipc50_pooled",
            "secondary_rho_ipc10_pooled",
            "per_dataset",
        ):
            if field not in report["arms"][arm]:
                raise ValueError(f"pooled report arm {arm!r} missing {field!r}")
    for dataset in DATASETS:
        if dataset not in report["per_dataset_standalone"]:
            raise ValueError(f"pooled report standalone missing {dataset!r}")
    if report["random_null"].get("reps") is None:
        raise ValueError("pooled report random_null missing reps")
    if report["bootstrap"].get("reps_requested") is None:
        raise ValueError("pooled report bootstrap missing reps_requested")


def _endpoint_subset_composition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per (dataset, ipc) subset composition (the D2 disclosure, measured)."""
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        per_ipc: dict[str, Any] = {}
        for ipc in sorted({row["ipc"] for row in rows}):
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset and row["ipc"] == ipc
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


POOLED_IMPLEMENTATION_MAP = [
    {
        "preregistration": "statistics.two_dataset_summary: pooled cluster unit = release_lineage x dataset",
        "implementation": "pooled_rows_from_reports(): dataset-qualified entry_id and lineage_dataset = dataset:release_lineage; resample_lineage_dataset_clusters() resamples the 14 clusters with replacement, #draw suffixes on entry_id and lineage_dataset, method_family unchanged so folds keep copies together",
    },
    {
        "preregistration": "statistics.two_dataset_summary.reporting_rule: per-dataset endpoints (CIFAR-10 E1, CIFAR-100 E2) and the pooled endpoint are reported separately; pooled-only reporting is forbidden",
        "implementation": "report carries per_dataset_standalone (verbatim copies of the frozen e1_report/e2_report arm endpoints), the pooled-model per_dataset cells, and the pooled endpoints; E1 inputs are read-only (sha256 registered in inputs)",
    },
    {
        "preregistration": "resampling.outer_design (LOFO x LOAO pooled OOS) applied to the pooled table",
        "implementation": "oos_pooled_predictions_2ds(): same fold structure and separation discipline as the per-dataset analyses; cells are (dataset, arch, ipc); oos_prediction_records dumped for local <1e-12 endpoint recomputation",
    },
    {
        "preregistration": "utility normalization fitted on train folds only (frozen caliber)",
        "implementation": "zscore_targets_train_only_2ds(): identical math to the per-dataset zscore_targets_train_only with the grouping key extended to (dataset, arch, ipc); skipped cells counted",
    },
    {
        "preregistration": "alpha re-selection inside train folds over the frozen grid (0.01, 0.1, 1, 10, 100)",
        "implementation": "shared select_alpha(): inner leave-one-family-out CV over the pooled families, scored by mean within-(dataset, arch, ipc)-cell Spearman",
    },
    {
        "preregistration": "primary endpoint construction (IPC50 pooled family x arch OOS Spearman) applied to the two-dataset table",
        "implementation": "endpoint_rho_pooled(): derived translation - a cell is (arch, dataset, ipc); the pooled IPC50 endpoint averages the within-cell OOS Spearman over the 4 archs x 2 datasets (equal weight, identical to the per-arch mean of per-dataset per-arch means); raw cross-dataset utility scales never mix inside one Spearman",
    },
    {
        "preregistration": "bootstrap: release_lineage x dataset clusters, 1000 reps, full refit per replicate",
        "implementation": "lineage_dataset_bootstrap(): every replicate re-runs cell z-score, alpha CV, ridge fit, OOS prediction and endpoint aggregation for both arms; replicates with any undefined pooled or per-dataset endpoint are skipped and counted (E1 skip discipline); 2.5/97.5 percentile CIs and delta frac_positive",
    },
    {
        "preregistration": "random null: gaussian random score column through the same OOS pipeline (20 reps)",
        "implementation": "random_null_2ds(): variants pure and base_plus_random; pooled primary rho_mean mean/p95 and regret_mean mean/p05 reported",
    },
    {
        "preregistration": "screening statement template (four-parameter version)",
        "implementation": "statistics.screening_statement_template filled verbatim with n=34, f=6, l=14 (lineage x dataset clusters), d=2; screening_statement_inputs documents the l=14 cluster unit",
    },
    {
        "preregistration": "D2 endpoint subset composition disclosure (IPC50/IPC10 per dataset)",
        "implementation": "cohort.endpoint_subset_composition: per (dataset, ipc) n_entries/n_families/n_lineages computed from the pooled rows (measured, not assumed)",
    },
]

def run_e2b_pooled(
    e1_report_path: str | Path,
    e2_report_path: str | Path,
    preregistration_path: str | Path,
    out_path: str | Path | None = None,
    bootstrap_reps: int = 1000,
    random_reps: int = 20,
    seed: int = 0,
    log_every: int = 100,
) -> dict[str, Any]:
    """Run the frozen two-dataset pooled summary and write the report."""
    e1_report_path = Path(e1_report_path)
    e2_report_path = Path(e2_report_path)
    preregistration_path = Path(preregistration_path)
    e1_report = json.loads(e1_report_path.read_text(encoding="utf-8"))
    e2_report = json.loads(e2_report_path.read_text(encoding="utf-8"))
    validate_report(e1_report, stage="auditbench_e1_report")
    validate_report(e2_report, stage="auditbench_e2_report")
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))

    rows = pooled_rows_from_reports(e1_report, e2_report)
    datasets = list(DATASETS)
    archs_e1 = list(e1_report["cohort"]["architectures"])
    archs_e2 = list(e2_report["cohort"]["architectures"])
    if archs_e1 != archs_e2:
        raise ValueError(f"arch sets differ: {archs_e1} vs {archs_e2}")
    archs = archs_e1
    seeds = list(e2_report["cohort"]["seeds"])
    ipcs = sorted({row["ipc"] for row in rows})
    completeness = check_complete_case(rows, archs, seeds)
    if not completeness["complete"]:
        raise RuntimeError(f"pooled rows incomplete: {completeness['problems']}")

    arms_out: dict[str, Any] = {}
    predictions_out: dict[str, Any] = {}
    prediction_records: dict[str, Any] = {}
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
                "regret_mean": (
                    float(np.mean(list(evaluation["regret"].values())))
                    if evaluation["regret"]
                    else None
                ),
                "random_expected_regret_mean": (
                    float(np.mean(list(evaluation["random_regret"].values())))
                    if evaluation["random_regret"]
                    else None
                ),
            },
        }
        predictions_out[arm] = {
            arch: dict(sorted(preds.items()))
            for arch, preds in predictions.items()
        }
        long_rows = [(row, arch) for row in rows for arch in archs]
        records: list[dict[str, Any]] = []
        for fold in result["details"]["folds"]:
            for index in fold["test_index"]:
                row, arch = long_rows[index]
                records.append(
                    {
                        "entry_id": row["entry_id"],
                        "dataset": row["dataset"],
                        "arch": arch,
                        "ipc": row["ipc"],
                        "method_family": row["method_family"],
                        "release_lineage": row["release_lineage"],
                        "lineage_dataset": row["lineage_dataset"],
                        "held_family": fold["held_family"],
                        "held_arch": fold["held_arch"],
                        "alpha": fold["alpha"],
                        "prediction": predictions[arch][row["entry_id"]],
                        "utility_mean": row["utility"][arch]["mean"],
                    }
                )
        prediction_records[arm] = records

    null_rng = np.random.default_rng(seed)
    null = random_null_2ds(
        rows, archs, datasets, ipcs, ALPHA_GRID, null_rng, random_reps
    )
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
            "release_lineage x dataset clusters (14 = 7 lineage names x 2 "
            "datasets; the two datasets share all 7 lineage names)"
        ),
    }

    per_dataset_standalone = {
        "cifar10": {
            "source": "e1_report.json (frozen, read-only, verbatim copies)",
            "arms": {
                arm: {
                    "primary_rho_ipc50": e1_report["arms"][arm]["primary_rho_ipc50"],
                    "secondary_rho_ipc10": e1_report["arms"][arm][
                        "secondary_rho_ipc10"
                    ],
                }
                for arm in ARMS
            },
        },
        "cifar100": {
            "source": "e2_report.json (stage A of this iteration, verbatim copies)",
            "arms": {
                arm: {
                    "primary_rho_ipc50": e2_report["arms"][arm]["primary_rho_ipc50"],
                    "secondary_rho_ipc10": e2_report["arms"][arm][
                        "secondary_rho_ipc10"
                    ],
                }
                for arm in ARMS
            },
        },
    }

    report: dict[str, Any] = {
        "stage": E2B_STAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "measured",
        "inputs": {
            "e1_report_path": str(e1_report_path),
            "e1_report_sha256": hashlib.sha256(
                e1_report_path.read_bytes()
            ).hexdigest(),
            "e2_report_path": str(e2_report_path),
            "e2_report_sha256": hashlib.sha256(
                e2_report_path.read_bytes()
            ).hexdigest(),
            "e2_preregistration_path": str(preregistration_path),
            "e2_preregistration_sha256": hashlib.sha256(
                preregistration_path.read_bytes()
            ).hexdigest(),
            "e1_inputs_note": (
                "E1 artifacts/preregistration/report are read-only inputs; "
                "no E1 byte is modified by this analysis"
            ),
        },
        "cohort": {
            "n_entries": len(rows),
            "datasets": datasets,
            "families": sorted({row["method_family"] for row in rows}),
            "lineages": sorted({row["release_lineage"] for row in rows}),
            "lineage_dataset_clusters": sorted(
                {row["lineage_dataset"] for row in rows}
            ),
            "architectures": archs,
            "seeds": seeds,
            "ipcs": ipcs,
            "completeness": completeness,
            "endpoint_subset_composition": _endpoint_subset_composition(rows),
        },
        "rows": rows,
        "oos_predictions": predictions_out,
        "oos_prediction_records": prediction_records,
        "arms": arms_out,
        "alpha_selection": {
            "grid": list(ALPHA_GRID),
            "chosen_per_fold": alphas_out,
        },
        "per_dataset_standalone": per_dataset_standalone,
        "random_null": null,
        "bootstrap": bootstrap,
        "regret": {arm: arms_out[arm]["regret"] for arm in ARMS},
        "screening_statement": screening_statement,
        "screening_statement_inputs": screening_inputs,
        "implementation_preregistration_map": POOLED_IMPLEMENTATION_MAP,
        "statistical_uncertainty": {
            "design": (
                "lineage x dataset cluster bootstrap (1000 reps, full refit "
                "per replicate) for grouped CIs; gaussian random-column null "
                "(20 reps) as negative control; the pooled design has 14 "
                "clusters and 6 pooled families, so all endpoints are "
                "screening-level"
            ),
            "zscore_skipped_cells_total": zscore_skipped_total,
            "bootstrap_reps_used": bootstrap["reps_used"],
            "bootstrap_reps_skipped_undefined": bootstrap[
                "reps_skipped_undefined"
            ],
            "random_null_reps": random_reps,
        },
        "rng_seed": {"random_null": seed, "bootstrap": seed + 1},
    }
    validate_pooled_report(report)
    if out_path is not None:
        out_path = Path(out_path)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        save_json(report, tmp)
        tmp.replace(out_path)
    return report
