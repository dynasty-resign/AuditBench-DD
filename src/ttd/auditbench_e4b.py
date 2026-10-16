"""AuditBench-DD iter-0008 E4b: degenerate endpoint family + POOLED4 extension.

Pure analysis round: zero training, zero download, zero GPU, zero retrieval.
Inputs are the frozen E1/E2/E3 reports, the E4a measurement register and the
E4a matrix artifacts (all read-only, sha256-registered).

T1 (E4b degenerate endpoint family): the measured E4 cohort is n=2 artifacts /
f=1 selection_family / l=1 lineage (inhouse_random_baseline_iter0007e4a) /
d=1 dataset (imagewoof). Under f=1 the LOFO hold-out structure does not exist
and under l=1 the lineage-cluster bootstrap has no resampling unit, so every
E4-standalone inferential endpoint of the frozen E4 pre-registration is
registered ``structurally_inapplicable_f1_l1`` (a structural-layer
registration: the criterion structure does not exist; distinct from the E3
vgg11 descriptive-collapse precedent where the criterion existed and the
measured values failed it). Single-cluster bootstrap is pseudo-inference and
is never computed. The pre-registered degenerate endpoint family is computed
instead: in-cell descriptive Spearman (mechanical definition, G6 annotation,
descriptive_with_degenerate_cohort), top-1 regret degenerate registration
(regret=0 by construction), compute break-even (measured), and per-cell
seed-level uncertainty description (mean +/- std, n=3, no CI constructed).

T2 (POOLED4): the E4 pre-registration ``resampling.pooled4_extension_rule``
is executed mechanically: four-dataset pooled LOFO x LOAO OOS with
cluster = release_lineage x dataset, reusing the dataset-aware machinery of
``ttd.auditbench_e2b`` unchanged with
``datasets=("cifar10", "cifar100", "tiny_imagenet", "imagewoof")``.
Per-dataset endpoints are reported alongside the pooled endpoint
(pooled-only reporting is forbidden). POOLED3 stays the frozen primary
pooled endpoint; POOLED4 vs POOLED3 differences are described at the
composition layer only (no secondary inference).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .audit import _spearman
from .auditbench_e2 import (
    ALPHA_GRID,
    ARM_FEATURES,
    ARMS,
    BOOTSTRAP_REPS,
    RANDOM_REPS,
    check_complete_case,
    validate_report,
)
from .auditbench_e2b import (
    endpoints_bundle,
    evaluate_predictions_2ds,
    lineage_dataset_bootstrap,
    oos_pooled_predictions_2ds,
    random_null_2ds,
    resample_lineage_dataset_clusters,
)
from .auditbench_e3b import E3_SINGLE_STAGE, pooled_rows_3ds

E4B_STAGE = "auditbench_e4b_report"
E4B_DEGENERACY_STAGE = "auditbench_e4b_degeneracy_note"
POOLED4_STAGE = "auditbench_pooled4_report"
DATASETS_4 = ("cifar10", "cifar100", "tiny_imagenet", "imagewoof")
E4_DATASET = "imagewoof"
E4_LINEAGE = "inhouse_random_baseline_iter0007e4a"
E4_CLUSTER = f"{E4_DATASET}:{E4_LINEAGE}"
E4_ENTRY_IDS = ("inhouse.random.ipc10", "inhouse.random.ipc50")
E4B_SEED = 0

DEGENERATE_VERDICT = "structurally_inapplicable_f1_l1"
G6_ANNOTATION = {
    "descriptive": True,
    "distinct": True,
    "frozen": True,
    "LOFO": True,
    "LOAO": True,
    "endpoint": True,
}

POOLED4_REQUIRED_KEYS = (
    "stage",
    "generated_at",
    "evidence_class",
    "inputs",
    "composition_verification",
    "cohort",
    "rows",
    "oos_predictions",
    "arms",
    "per_dataset_standalone",
    "random_null",
    "bootstrap",
    "cluster_fragility",
    "pooled3_frozen_reference",
    "pooled4_vs_pooled3_composition_only",
    "screening_statement",
    "screening_statement_inputs",
    "implementation_preregistration_map",
    "statistical_uncertainty",
    "rng_seed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_e4_rows(
    cohort_fill: dict[str, Any],
    matrix_dir: str | Path,
    register: dict[str, Any],
    progress: dict[str, Any],
) -> list[dict[str, Any]]:
    """E4 rows from the filled cohort + E4a matrix artifacts + register check.

    Same construction caliber as auditbench_e2.build_rows: scores from the
    per-entry scores containers (nulls/runtime_sec separated), utility from
    the per-entry evaluation containers (mean/std/n per arch), train walls
    from the E4a progress cells. The E4a measurement register is asserted
    value-equal (mean/std recomputed from the register per-seed runs,
    ddof=0 register caliber).
    """
    matrix_dir = Path(matrix_dir)
    rows: list[dict[str, Any]] = []
    for entry in cohort_fill["cohort"]["primary_entries"]:
        entry_id = entry["entry_id"]
        scores_path = matrix_dir / f"{entry_id}.scores.json"
        evaluation_path = matrix_dir / f"{entry_id}.eval_input.evaluation.json"
        record = load_json(scores_path)
        if record.get("smoke_only") is not False:
            raise ValueError(f"scores record is not a formal artifact: {scores_path}")
        evaluation = load_json(evaluation_path)
        utility = {
            item["architecture"]: {
                "mean": float(item["mean"]),
                "std": float(item["std"]),
                "n": int(item["n"]),
            }
            for item in evaluation["summary"]
        }
        reg_runs = register["utilities"][entry_id]["runs"]
        for arch, summary in utility.items():
            seed_values = [
                float(run["top1"])
                for run in reg_runs
                if run["architecture"] == arch
            ]
            if len(seed_values) != 3:
                raise ValueError(f"register {entry_id}|{arch} n={len(seed_values)}")
            mean = sum(seed_values) / 3.0
            var = sum((value - mean) ** 2 for value in seed_values) / 3.0
            std = var**0.5
            if abs(mean - summary["mean"]) > 1e-9 or abs(std - summary["std"]) > 1e-9:
                raise ValueError(
                    f"register/evaluation divergence at {entry_id}|{arch}: "
                    f"register mean/std {mean}/{std} vs evaluation {summary}"
                )
            if int(summary["n"]) != 3:
                raise ValueError(
                    f"{entry_id}|{arch} evaluation n={summary['n']} (expected 3)"
                )
        scores = {
            key: value
            for key, value in record["scores"].items()
            if key not in ("nulls", "runtime_sec")
        }
        train_wall: dict[str, float] = {}
        cells = progress.get("entries", {}).get(entry_id, {}).get("cells", {})
        for cell_key, cell in cells.items():
            if isinstance(cell.get("wall_sec"), (int, float)):
                train_wall[cell_key] = float(cell["wall_sec"])
        rows.append(
            {
                "entry_id": entry_id,
                "release_lineage": entry["release_lineage"],
                "method_family": entry["method_family"],
                "ipc": int(entry["ipc"]),
                "scores": scores,
                "nulls": record["scores"]["nulls"],
                "utility": utility,
                "proxy_runtime_sec": float(record["scores"]["runtime_sec"]),
                "train_wall_sec": train_wall,
            }
        )
    return rows

# ---------------------------------------------------------------------------
# T1: E4b degenerate endpoint family
# ---------------------------------------------------------------------------


def degeneracy_registrations() -> list[dict[str, Any]]:
    """Per-endpoint structural registrations (no numeric value is fabricated).

    Measured cohort counts: n=2 artifacts, f=1 selection_family,
    l=1 inhouse_random_baseline_iter0007e4a, d=1 imagewoof. At f=1 the LOFO
    hold-out has no held-out family; at l=1 the lineage-cluster bootstrap has
    no resampling unit. Computing a single-cluster bootstrap or any
    LOFO-based value here would be pseudo-inference and is never done.
    """
    reason_lofo = (
        "measured cohort f=1 (selection_family only): leave-one-family-out has "
        "no hold-out family, so the frozen family x arch pooled OOS fold "
        "structure does not exist on the E4-standalone cohort"
    )
    reason_boot = (
        "measured cohort l=1 (inhouse_random_baseline_iter0007e4a only): the "
        "release_lineage-cluster bootstrap has no resampling unit; a "
        "single-cluster bootstrap would be pseudo-inference and is never computed"
    )
    counts = {"n": 2, "f": 1, "l": 1, "d": 1}
    return [
        {
            "endpoint": "primary_endpoint: pooled_family_arch_oos_spearman_ipc50",
            "registration": DEGENERATE_VERDICT,
            "measured_counts": counts,
            "reason": reason_lofo,
        },
        {
            "endpoint": "secondary_endpoint: pooled family x arch OOS Spearman at IPC10 (same construction)",
            "registration": DEGENERATE_VERDICT,
            "measured_counts": counts,
            "reason": reason_lofo,
        },
        {
            "endpoint": "resampling.outer_design: LOFO x LOAO pooled OOS fold structure (E4-standalone)",
            "registration": DEGENERATE_VERDICT,
            "measured_counts": counts,
            "reason": reason_lofo
            + "; LOAO over 4 architectures alone cannot instantiate the frozen joint LOFO x LOAO design",
        },
        {
            "endpoint": "statistics.confidence_intervals: release_lineage-cluster bootstrap CI (reps=1000, full refit)",
            "registration": DEGENERATE_VERDICT,
            "measured_counts": counts,
            "reason": reason_boot,
        },
        {
            "endpoint": "secondary_endpoint: per-family OOS Spearman (descriptive only)",
            "registration": DEGENERATE_VERDICT,
            "measured_counts": counts,
            "reason": reason_lofo
            + "; a single-family cohort offers no per-family hold-out structure at all",
        },
        {
            "endpoint": "secondary_endpoint: IPC50-style in-cell structure reproduction (in-cell Spearman inside each (arch, ipc50) cell)",
            "registration": "structurally_not_applicable_single_lineage",
            "measured_counts": counts,
            "reason": (
                "single lineage: no cross-method in-cell structure exists to "
                "reproduce; each (arch, ipc) cell contains exactly 1 measured "
                "entry and no cross-method cell-internal Spearman is computed "
                "on 1-2 entries"
            ),
        },
    ]


def incell_descriptive_spearman(
    rows: list[dict[str, Any]], archs: list[str]
) -> dict[str, Any]:
    """In-cell descriptive Spearman under the mechanical frozen definition.

    Mechanical definition (E4 pre-registration secondary endpoints): in-cell
    Spearman between proxy scores and utility across entries within the same
    (arch, ipc) cell. The measured cohort has exactly 1 entry per (arch, ipc)
    cell, so the Spearman is undefined in every cell (n=1 < 3, the frozen
    _spearman guard); this near-zero discriminative reach is registered, not
    hidden, and is never restated as a method-level conclusion. G6 annotation
    applies verbatim: the in-cell descriptive value is descriptive and
    distinct from the frozen LOFO x LOAO endpoint.
    """
    cells: dict[str, Any] = {}
    for arch in archs:
        for ipc in sorted({row["ipc"] for row in rows}):
            cell_rows = [row for row in rows if row["ipc"] == ipc]
            utilities = np.array(
                [row["utility"][arch]["mean"] for row in cell_rows], dtype=float
            )
            proxies = np.array(
                [row["scores"]["ridge_proxy"] for row in cell_rows], dtype=float
            )
            rho = _spearman(proxies, utilities)
            assert rho is None, "measured cohort must stay degenerate (n=1 per cell)"
            cells[f"{arch}|ipc{ipc}"] = {
                "n_entries_in_cell": len(cell_rows),
                "entry_ids": [row["entry_id"] for row in cell_rows],
                "spearman": None,
                "undefined_reason": (
                    "n=1 entry per (arch, ipc) cell (measured cohort); the "
                    "frozen _spearman guard (len < 3) returns undefined; the "
                    "near-zero discriminative reach of a one-entry cell is "
                    "registered, not hidden"
                ),
                "mechanical_definition": (
                    "in-cell Spearman between proxy scores and utility across "
                    "entries within the same (arch, ipc) cell"
                ),
                "label": "descriptive_with_degenerate_cohort",
                "g6_annotation": dict(G6_ANNOTATION),
            }
    return {
        "definition_source": "e4_preregistration.json secondary_endpoints (IPC50-style in-cell construction, mechanical definition)",
        "cells": cells,
        "note": (
            "descriptive registration only; never a method-level conclusion; "
            "the E3 vgg11 in-cell descriptive phenomenon (e.g. the ipc50 cell "
            "0.7071 four-way tie) is a cell-internal descriptive value, while "
            "the frozen endpoint -0.3536 is a LOFO x LOAO endpoint phenomenon; "
            "the two are never conflated and never cross-cited"
        ),
    }


def regret_degenerate_register(
    rows: list[dict[str, Any]], archs: list[str]
) -> dict[str, Any]:
    """Top-1 regret degenerate registration (regret=0 by construction).

    Every (arch, ipc) cell of the measured cohort contains exactly 1 entry,
    so the argmax over proxy scores is the only entry and top utility equals
    its utility: regret = 0 by construction; random_expected_regret =
    top - mean = 0 by the same caliber. This is never restated as 'proxy
    selection has no regret' or any method-level conclusion.
    """
    cells: dict[str, Any] = {}
    for arch in archs:
        for ipc in sorted({row["ipc"] for row in rows}):
            cell_rows = [row for row in rows if row["ipc"] == ipc]
            utilities = [row["utility"][arch]["mean"] for row in cell_rows]
            top = float(max(utilities))
            chosen = utilities[0]
            cells[f"{arch}|ipc{ipc}"] = {
                "n_entries_in_cell": len(cell_rows),
                "regret": top - float(chosen),
                "random_expected_regret": top - float(np.mean(utilities)),
                "registration": "regret=0 by construction (argmax over a one-entry cell is the only entry)",
                "frozen_definition": "regret = top utility - utility(argmax proxy); random_expected_regret = top utility - mean utility within the (arch, ipc) cell",
            }
    return {
        "cells": cells,
        "note": (
            "degenerate registration only: with 1 entry per cell regret is 0 "
            "by construction; this carries no method-level meaning and is "
            "never restated as evidence about proxy selection quality"
        ),
    }


def compute_break_even_measured(
    rows: list[dict[str, Any]],
    archs: list[str],
    seeds: list[int],
    budget_addendum: dict[str, Any],
) -> dict[str, Any]:
    """Compute break-even (measured): proxy extraction wall vs per-cell train wall.

    Proxy extraction walls (5.1s ipc10 / 7.1s ipc50) are the measured E4a
    per-entry runtime_sec; student training walls are the 24 measured
    per-(entry, arch, seed) cells of the E4a progress register. The ratio
    proxy_wall / train_cell_wall is computed per cell and summarized as
    min/max/mean. Per-step coefficients keep their measured/derived evidence
    classes verbatim from the frozen budget addendum.
    """
    per_cell: list[dict[str, Any]] = []
    for row in rows:
        proxy_wall = float(row["proxy_runtime_sec"])
        for arch in archs:
            for seed in seeds:
                cell_key = f"{arch}|{seed}"
                train_wall = row["train_wall_sec"].get(cell_key)
                if train_wall is None:
                    raise ValueError(f"missing train wall {row['entry_id']}|{cell_key}")
                per_cell.append(
                    {
                        "entry_id": row["entry_id"],
                        "architecture": arch,
                        "seed": seed,
                        "proxy_extraction_wall_sec": proxy_wall,
                        "train_cell_wall_sec": float(train_wall),
                        "ratio_proxy_over_train": proxy_wall / float(train_wall),
                        "evidence_class": "measured",
                    }
                )
    ratios = [cell["ratio_proxy_over_train"] for cell in per_cell]
    coefficients = budget_addendum["coefficients_64px"]
    per_step_evidence = {
        arch: {
            "b64_per_step_sec": coefficients["b64_per_step_sec"][arch],
            "evidence_class": (
                "measured"
                if arch in coefficients["probed"]
                else "derived (frozen escalation r_b_max=2.1462 from the measured probes)"
            ),
        }
        for arch in archs
    }
    return {
        "definition": "proxy extraction wall time per artifact vs per (arch, seed) student training wall time at the frozen budget (e4_preregistration statistics.compute_break_even)",
        "per_cell": per_cell,
        "ratio_min": float(min(ratios)),
        "ratio_max": float(max(ratios)),
        "ratio_mean": float(np.mean(ratios)),
        "n_cells": len(per_cell),
        "per_step_coefficients_64px": per_step_evidence,
        "train_walls_note": "all 24 (entry x arch x seed) train walls are measured E4a progress values",
    }


def seed_level_uncertainty(register: dict[str, Any]) -> dict[str, Any]:
    """Per-cell seed-level uncertainty description (mean +/- std, n=3).

    Registration of existence form only: no CI or inferential interval is
    constructed anywhere.
    """
    cells = register["cell_seed_stats"]
    return {
        "cells": {key: dict(value) for key, value in cells.items()},
        "n_seeds": 3,
        "note": (
            "descriptive seed-level mean +/- std (n=3 seeds) registered as an "
            "existence form; no confidence interval and no inferential "
            "interval is constructed"
        ),
    }

# ---------------------------------------------------------------------------
# T2: POOLED4 (four-dataset pooled, lineage x dataset clusters)
# ---------------------------------------------------------------------------


def pooled4_rows(
    e1_report: dict[str, Any],
    e2_report: dict[str, Any],
    e3_single: dict[str, Any],
    e4_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dataset-qualified pooled rows: E1+E2+E3 verbatim + E4 appended."""
    rows = pooled_rows_3ds(e1_report, e2_report, e3_single)
    for row in e4_rows:
        rows.append(
            {
                "entry_id": f"{E4_DATASET}:{row['entry_id']}",
                "dataset": E4_DATASET,
                "release_lineage": row["release_lineage"],
                "lineage_dataset": E4_CLUSTER,
                "method_family": row["method_family"],
                "ipc": int(row["ipc"]),
                "scores": row["scores"],
                "utility": row["utility"],
            }
        )
    return rows


def cluster_composition_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset x lineage x entry-count x ipc-coverage composition table."""
    clusters: dict[str, Any] = {}
    for row in rows:
        entry = clusters.setdefault(
            row["lineage_dataset"],
            {
                "dataset": row["dataset"],
                "release_lineage": row["release_lineage"],
                "method_families": set(),
                "entry_ids": set(),
                "ipcs": set(),
            },
        )
        entry["method_families"].add(row["method_family"])
        entry["entry_ids"].add(row["entry_id"])
        entry["ipcs"].add(int(row["ipc"]))
    table = {
        cluster: {
            "dataset": value["dataset"],
            "release_lineage": value["release_lineage"],
            "n_entries": len(value["entry_ids"]),
            "method_families": sorted(value["method_families"]),
            "ipcs": sorted(value["ipcs"]),
            "entry_ids": sorted(value["entry_ids"]),
        }
        for cluster, value in sorted(clusters.items())
    }
    per_dataset: dict[str, Any] = {}
    for dataset in DATASETS_4:
        subset = [row for row in rows if row["dataset"] == dataset]
        per_dataset[dataset] = {
            "n_entries": len(subset),
            "n_clusters": len({row["lineage_dataset"] for row in subset}),
            "n_families": len({row["method_family"] for row in subset}),
            "ipcs": sorted({row["ipc"] for row in subset}),
        }
    return {
        "stage": "auditbench_pooled4_cluster_composition",
        "cluster_unit": "release_lineage x dataset",
        "n_rows": len(rows),
        "n_clusters": len(table),
        "clusters": table,
        "per_dataset": per_dataset,
    }


def composition_verification(
    rows3: list[dict[str, Any]],
    rows4: list[dict[str, Any]],
    e3b_pooled: dict[str, Any],
) -> dict[str, Any]:
    """Row-level reconstruction check BEFORE any pooled computation.

    E1+E2+E3 rows must reconstruct the frozen e3b_report
    three_dataset_pooled cohort declaration exactly (n_entries, cluster unit,
    cluster names); E4 appends exactly 1 cluster and 2 entries.
    """
    declared = e3b_pooled["cohort"]
    expected_clusters = sorted(declared["lineage_dataset_clusters"])
    actual_clusters3 = sorted({row["lineage_dataset"] for row in rows3})
    actual_clusters4 = sorted({row["lineage_dataset"] for row in rows4})
    new_clusters = sorted(set(actual_clusters4) - set(actual_clusters3))
    new_entries = sorted(
        {row["entry_id"] for row in rows4} - {row["entry_id"] for row in rows3}
    )
    return {
        "e3b_reconstruction": {
            "expected_n_entries": declared["n_entries"],
            "actual_n_entries": len(rows3),
            "n_entries_match": declared["n_entries"] == len(rows3),
            "expected_cluster_unit": "release_lineage x dataset",
            "expected_n_clusters": len(expected_clusters),
            "actual_n_clusters": len(actual_clusters3),
            "cluster_names_match": expected_clusters == actual_clusters3,
            "expected_clusters": expected_clusters,
        },
        "e4_extension": {
            "new_clusters": new_clusters,
            "expected_new_clusters": [E4_CLUSTER],
            "new_clusters_match": new_clusters == [E4_CLUSTER],
            "new_entries": new_entries,
            "n_new_entries": len(new_entries),
            "expected_n_new_entries": 2,
        },
        "pooled4_totals": {
            "n_rows": len(rows4),
            "n_clusters": len(actual_clusters4),
        },
        "task_text_deviation_note": (
            "the iter-0008 task text parenthetically expects 20 POOLED3 "
            "clusters and 21 POOLED4 clusters; the frozen e3b_report "
            "three_dataset_pooled cohort declares 19 lineage x dataset "
            "clusters (7 E1 + 7 E2 + 5 E3, measured), so the mechanically "
            "reconstructed POOLED4 cohort is 20 clusters (19 + exactly 1 E4 "
            "cluster) over 46 entries (44 + 2). The file-declared counts are "
            "authoritative; this deviation is registered, not reconciled by "
            "any fabrication"
        ),
    }


def _endpoint_subset_composition_4ds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in DATASETS_4:
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


def cluster_fragility_diagnostic(
    rows: list[dict[str, Any]], reps: int, seed: int
) -> dict[str, Any]:
    """Resampling-mechanics diagnostic for the single-entry E4 cluster.

    Replays the exact cluster-draw sequence the bootstrap consumes (same rng
    seed and same draw order: one rng.choice call per replicate, no other rng
    consumption inside the driver) and counts how often the E4 cluster is
    drawn >= 3 times. Derived descriptive mechanics, not an endpoint and not
    an inference.
    """
    rng = np.random.default_rng(seed)
    counts = {"drawn_0": 0, "drawn_1": 0, "drawn_2": 0, "drawn_ge3": 0}
    for _ in range(reps):
        _pseudo, draw_counts = resample_lineage_dataset_clusters(rows, rng)
        n = draw_counts.get(E4_CLUSTER, 0)
        key = "drawn_ge3" if n >= 3 else f"drawn_{n}"
        counts[key] += 1
    return {
        "e4_cluster": E4_CLUSTER,
        "single_entry_cluster": True,
        "structural_fact": (
            "E4 contributes exactly 1 cluster of 2 single-entry cells; in the "
            "cluster bootstrap the E4 cluster is either wholly inside or "
            "wholly outside each replicate; even when drawn >= 3 times, the "
            "#draw copies share identical scores and utility, so within-cell "
            "Spearman is undefined (the frozen _spearman guard returns None "
            "on zero-variance inputs) and the imagewoof per-dataset endpoint "
            "stays undefined in every replicate"
        ),
        "reps_replayed": reps,
        "e4_cluster_draw_frequency": counts,
        "evidence_class": "derived (resampling-mechanics diagnostic replay; not an endpoint, not an inference)",
    }


def validate_pooled4_report(report: dict[str, Any]) -> None:
    missing = [key for key in POOLED4_REQUIRED_KEYS if key not in report]
    if missing:
        raise ValueError(f"pooled4 report missing keys: {missing}")
    if report.get("stage") != POOLED4_STAGE:
        raise ValueError(f"pooled4 report stage must be {POOLED4_STAGE!r}")
    for arm in ARMS:
        if arm not in report["arms"]:
            raise ValueError(f"pooled4 report arms missing arm {arm!r}")
        for field in (
            "primary_rho_ipc50_pooled",
            "secondary_rho_ipc10_pooled",
            "per_dataset",
        ):
            if field not in report["arms"][arm]:
                raise ValueError(f"pooled4 report arm {arm!r} missing {field!r}")
    for dataset in DATASETS_4:
        if dataset not in report["arms"][ARMS[0]]["per_dataset"]:
            raise ValueError(f"per_dataset missing {dataset!r}")
        if dataset not in report["per_dataset_standalone"]:
            raise ValueError(f"per_dataset_standalone missing {dataset!r}")
    if report["random_null"].get("reps") is None:
        raise ValueError("pooled4 report random_null missing reps")
    if report["bootstrap"].get("reps_requested") is None:
        raise ValueError("pooled4 report bootstrap missing reps_requested")

POOLED4_IMPLEMENTATION_MAP = [
    {
        "preregistration": "e4_preregistration resampling.pooled4_extension_rule: four-dataset pooled resampling, cluster = release_lineage x dataset (isomorphic to the E2b/E3 pooled construction)",
        "implementation": "pooled4_rows(): dataset-qualified entry_id and lineage_dataset = dataset:release_lineage for E1/E2/E3 (pooled_rows_3ds verbatim) plus the E4 rows under imagewoof:inhouse_random_baseline_iter0007e4a; auditbench_e2b.resample_lineage_dataset_clusters() resamples the 20 clusters with replacement, #draw suffixes keep copies together",
    },
    {
        "preregistration": "statistics.pooled4_reporting_rule: per-dataset endpoints (CIFAR-10 E1, CIFAR-100 E2, Tiny-ImageNet E3, ImageWoof E4) and the pooled endpoint reported separately; pooled-only reporting forbidden",
        "implementation": "report carries per_dataset_standalone (verbatim copies of the frozen e1_report/e2_report arms, the E3 single-dataset endpoints, and the E4 structurally_inapplicable registration), the pooled-model per_dataset cells for all four datasets, and the pooled endpoints",
    },
    {
        "preregistration": "resampling.outer_design (LOFO x LOAO pooled OOS) applied to the pooled table",
        "implementation": "auditbench_e2b.oos_pooled_predictions_2ds() reused unchanged with datasets=(cifar10, cifar100, tiny_imagenet, imagewoof); cells are (dataset, arch, ipc); same separation discipline (the same artifact never appears in both train and test of one resample)",
    },
    {
        "preregistration": "utility normalization fitted on train folds only (frozen caliber)",
        "implementation": "auditbench_e2b.zscore_targets_train_only_2ds() reused unchanged: per-(dataset, arch, ipc)-cell z-score fitted on train rows only; cells with <2 train rows keep raw values and are counted",
    },
    {
        "preregistration": "alpha re-selection inside train folds over the frozen grid (0.01, 0.1, 1, 10, 100)",
        "implementation": "shared auditbench_e2.select_alpha(): inner leave-one-family-out CV over the pooled families, scored by mean within-(dataset, arch, ipc)-cell Spearman",
    },
    {
        "preregistration": "POOLED4 extension endpoint construction (pooled family x arch OOS Spearman, IPC50 primary / IPC10 secondary)",
        "implementation": "auditbench_e2b.endpoint_rho_pooled(): within-cell OOS Spearman averaged over the 4 archs x 4 datasets cells with equal weight; raw cross-dataset utility scales never mix inside one Spearman; imagewoof cells hold 1 entry each and contribute no defined within-cell Spearman (registered)",
    },
    {
        "preregistration": "bootstrap: release_lineage x dataset clusters, 1000 reps, full refit per replicate",
        "implementation": "auditbench_e2b.lineage_dataset_bootstrap() reused unchanged: every replicate re-runs cell z-score, alpha CV, ridge fit, OOS prediction and endpoint aggregation for both arms; replicates with any undefined pooled or per-dataset endpoint are skipped and counted (E1 skip discipline); 2.5/97.5 percentile CIs and delta frac_positive",
    },
    {
        "preregistration": "random null: gaussian random score column through the same OOS pipeline (20 reps, variants pure and base_plus_random)",
        "implementation": "auditbench_e2b.random_null_2ds() reused unchanged with the four-dataset row set; pooled primary rho_mean mean/p95 and regret_mean mean/p05 reported per variant",
    },
    {
        "preregistration": "screening statement filled at analysis time with the actual pooled n/f/l/d (filled_statement_rule)",
        "implementation": "e4_preregistration statistics.screening_statement_template filled verbatim with measured n=46, f=6, l=20 (lineage x dataset clusters), d=4; screening_statement_inputs documents the l=20 cluster unit",
    },
    {
        "preregistration": "endpoint subset composition disclosure (IPC50/IPC10 per dataset)",
        "implementation": "cohort.endpoint_subset_composition: per (dataset, ipc) n_entries/n_families/n_lineages computed from the pooled rows (measured, not assumed)",
    },
    {
        "preregistration": "POOLED3 remains the frozen primary pooled endpoint; POOLED4 is a pre-registered extension endpoint; pooled4 vs pooled3 differences are composition-layer descriptions only",
        "implementation": "pooled3_frozen_reference carries the frozen POOLED3 values verbatim from e3b_report.json (read-only); pooled4_vs_pooled3_composition_only registers only entry/cluster counts and forbids any secondary inference on the difference",
    },
]


def run_pooled4(
    e1_report: dict[str, Any],
    e2_report: dict[str, Any],
    e3_single: dict[str, Any],
    e3b_pooled: dict[str, Any],
    e4_rows: list[dict[str, Any]],
    preregistration: dict[str, Any],
    bootstrap_reps: int = BOOTSTRAP_REPS,
    random_reps: int = RANDOM_REPS,
    seed: int = E4B_SEED,
    log_every: int = 100,
) -> dict[str, Any]:
    """Run the frozen POOLED4 extension (lineage x dataset clusters, 4 datasets)."""
    validate_report(e1_report, stage="auditbench_e1_report")
    validate_report(e2_report, stage="auditbench_e2_report")
    validate_report(e3_single, stage=E3_SINGLE_STAGE)

    rows3 = pooled_rows_3ds(e1_report, e2_report, e3_single)
    rows = pooled4_rows(e1_report, e2_report, e3_single, e4_rows)
    composition = composition_verification(rows3, rows, e3b_pooled)
    if not composition["e3b_reconstruction"]["n_entries_match"]:
        raise RuntimeError("E1+E2+E3 reconstruction does not match the frozen e3b cohort")
    if not composition["e3b_reconstruction"]["cluster_names_match"]:
        raise RuntimeError("E1+E2+E3 cluster names do not match the frozen e3b cohort")
    if not composition["e4_extension"]["new_clusters_match"]:
        raise RuntimeError("E4 extension cluster mismatch")
    composition_table = cluster_composition_table(rows)

    datasets = list(DATASETS_4)
    archs = list(e2_report["cohort"]["architectures"])
    for name, report in (("cifar10", e1_report), ("tiny_imagenet", e3_single)):
        if list(report["cohort"]["architectures"]) != archs:
            raise ValueError(f"arch sets differ for {name}")
    seeds = list(e2_report["cohort"]["seeds"])
    ipcs = sorted({row["ipc"] for row in rows})
    completeness = check_complete_case(rows, archs, seeds)
    if not completeness["complete"]:
        raise RuntimeError(f"pooled4 rows incomplete: {completeness['problems']}")

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
        per_dataset = {}
        for dataset in datasets:
            values = bundle["per_dataset"][dataset]
            entry = dict(values)
            if dataset == E4_DATASET and (
                entry["primary_rho_ipc50"] is None
                or entry["secondary_rho_ipc10"] is None
            ):
                entry["undefined_reason"] = (
                    "each imagewoof (arch, ipc) cell holds exactly 1 measured "
                    "entry; the frozen evaluate guard requires >= 3 rows per "
                    "cell, so the E4 per-dataset endpoint is undefined; this "
                    "is the same structural fact as the E4-standalone "
                    "structurally_inapplicable_f1_l1 registration and is "
                    "reported alongside, never hidden"
                )
            per_dataset[dataset] = entry
        arms_out[arm] = {
            "features": list(ARM_FEATURES[arm]),
            "rho_cells": evaluation["rho"],
            "primary_rho_ipc50_pooled": bundle["primary_rho_ipc50_pooled"],
            "secondary_rho_ipc10_pooled": bundle["secondary_rho_ipc10_pooled"],
            "per_dataset": per_dataset,
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
    fragility = cluster_fragility_diagnostic(rows, bootstrap_reps, seed + 1)

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
            "release_lineage x dataset clusters (20 = 7 E1 + 7 E2 + 5 E3 + 1 E4; "
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
            "source": "e3b_report.json e3_single_dataset section (frozen, read-only, verbatim copies)",
            "arms": {
                arm: {
                    "primary_rho_ipc50": e3_single["arms"][arm]["primary_rho_ipc50"],
                    "secondary_rho_ipc10": e3_single["arms"][arm]["secondary_rho_ipc10"],
                }
                for arm in ARMS
            },
        },
        "imagewoof": {
            "source": "e4b_degeneracy_note.json (this round)",
            "registration": DEGENERATE_VERDICT,
            "note": (
                "the E4-standalone primary/secondary endpoints are registered "
                "structurally_inapplicable_f1_l1 (measured n=2 / f=1 / l=1 / "
                "d=1); no E4-standalone endpoint value exists to copy; this "
                "entry is the degenerate registration, reported alongside"
            ),
        },
    }

    pooled3 = e3b_pooled
    pooled3_reference = {
        "status": "frozen primary pooled endpoint (zero byte change; POOLED4 is a pre-registered extension endpoint only)",
        "source": "runs/tiny_imagenet/auditbench_e3/e3b_report.json three_dataset_pooled (read-only)",
        "primary_rho_ipc50_pooled": pooled3["arms"]["base"]["primary_rho_ipc50_pooled"],
        "bootstrap": {
            "reps_requested": pooled3["bootstrap"]["reps_requested"],
            "reps_used": pooled3["bootstrap"]["reps_used"],
            "reps_skipped_undefined": pooled3["bootstrap"]["reps_skipped_undefined"],
            "base_pooled_primary": pooled3["bootstrap"]["arms"]["base"]["pooled_primary"],
        },
        "random_null_pooled_primary_p95": {
            variant: pooled3["random_null"]["variants"][variant]["pooled_primary"]["p95"]
            for variant in ("pure", "base_plus_random")
        },
    }

    report: dict[str, Any] = {
        "stage": POOLED4_STAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "measured",
        "inputs": {},
        "composition_verification": composition,
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
            "endpoint_subset_composition": _endpoint_subset_composition_4ds(rows),
            "e4_cohort_structural_fact": (
                "the 2 E4 entries share one method family (selection_family) "
                "and one lineage; E4-internal folds do not constitute a "
                "cross-method contrast. This structural fact is registered in "
                "the cohort section; it does not alter the mechanical "
                "computation of the joint endpoints (the joint fold "
                "structure is supplied by cross-dataset family diversity), "
                "and no post-hoc weighting or exclusion of the E4 "
                "contribution is applied"
            ),
        },
        "rows": rows,
        "cluster_composition": composition_table,
        "oos_predictions": predictions_out,
        "arms": arms_out,
        "alpha_selection": {"grid": list(ALPHA_GRID), "chosen_per_fold": alphas_out},
        "per_dataset_standalone": per_dataset_standalone,
        "random_null": null,
        "bootstrap": bootstrap,
        "cluster_fragility": fragility,
        "pooled3_frozen_reference": pooled3_reference,
        "pooled4_vs_pooled3_composition_only": {
            "pooled4_entries": len(rows),
            "pooled3_entries": pooled3["cohort"]["n_entries"],
            "pooled4_clusters": n_lineage_dataset,
            "pooled3_clusters": len(pooled3["cohort"]["lineage_dataset_clusters"]),
            "note": (
                "composition-layer description only: POOLED4 adds exactly 2 "
                "entries and exactly 1 cluster (the single-entry E4 lineage "
                "cluster) to the frozen POOLED3 cohort. Any numeric endpoint "
                "difference between POOLED4 and POOLED3 is not subject to "
                "secondary inference and no superiority or inferiority "
                "statement is made"
            ),
        },
        "screening_statement": screening_statement,
        "screening_statement_inputs": screening_inputs,
        "implementation_preregistration_map": POOLED4_IMPLEMENTATION_MAP,
        "statistical_uncertainty": {
            "design": (
                "lineage x dataset cluster bootstrap (1000 reps, full refit "
                "per replicate; 20 clusters) for grouped CIs; gaussian "
                "random-column null (20 reps, variants pure and "
                "base_plus_random) as negative control; n=46 entries, f=6 "
                "families, l=20 clusters, d=4 datasets: screening-level"
            ),
            "zscore_skipped_cells_total": zscore_skipped_total,
            "bootstrap_reps_used": bootstrap["reps_used"],
            "bootstrap_reps_skipped_undefined": bootstrap["reps_skipped_undefined"],
            "bootstrap_usable_reps_note": (
                "usable replicate count and the cluster-structure fragility "
                "are registered as measured (cluster_fragility section): the "
                "single-entry E4 cluster leaves the imagewoof per-dataset "
                "endpoint undefined in replicates, and the frozen skip "
                "discipline requires every per-dataset endpoint defined"
            ),
            "random_null_reps": random_reps,
        },
        "rng_seed": {"random_null": seed, "bootstrap": seed + 1},
    }
    validate_pooled4_report(report)
    return report

# ---------------------------------------------------------------------------
# T1 driver: E4b report assembly
# ---------------------------------------------------------------------------


def run_e4b(
    preregistration: dict[str, Any],
    cohort_fill: dict[str, Any],
    register: dict[str, Any],
    progress: dict[str, Any],
    budget_addendum: dict[str, Any],
    matrix_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Assemble the E4b report + degeneracy note; returns (report, note, rows)."""
    rows = build_e4_rows(cohort_fill, matrix_dir, register, progress)
    archs = list(preregistration["student_architectures"]["set"])
    seeds = list(preregistration["training_budget"]["seeds"])
    fill = register["screening_fill"]
    counts = {"n": fill["n"], "f": fill["f"], "l": fill["l"], "d": fill["d"]}

    registrations = degeneracy_registrations()
    incell = incell_descriptive_spearman(rows, archs)
    regret = regret_degenerate_register(rows, archs)
    break_even = compute_break_even_measured(rows, archs, seeds, budget_addendum)
    seed_stats = seed_level_uncertainty(register)

    now = datetime.now(timezone.utc).isoformat()
    note = {
        "stage": E4B_DEGENERACY_STAGE,
        "generated_at": now,
        "evidence_class": "measured (structural registration over measured cohort counts)",
        "measured_counts": counts,
        "registrations": registrations,
        "e3_precedent_distinction": (
            "E3 vgg11 registered a descriptive collapse: the criterion "
            "structure existed and the measured values failed it (criteria "
            "not met, honestly registered). This E4 registration is a "
            "structural-layer adjudication: at f=1/l=1 the criterion "
            "structure itself (LOFO hold-out, cluster resampling unit) does "
            "not exist, so the endpoints are registered "
            "structurally_inapplicable_f1_l1. The two layers are never "
            "conflated"
        ),
        "pseudo_inference_boundary": (
            "no single-cluster bootstrap, no LOFO/LOAO fold, no CI and no "
            "E4-standalone inferential value is computed anywhere in this "
            "round; forcing one is a void-level protocol violation"
        ),
        "degenerate_endpoint_family": {
            "incell_descriptive_spearman": "computed mechanically, undefined in all 8 cells (n=1 per cell), G6-annotated, labelled descriptive_with_degenerate_cohort",
            "top1_regret": "registered regret=0 by construction per cell (no method-level meaning)",
            "compute_break_even": "measured per-cell ratios with min/max/mean",
            "seed_level_uncertainty": "per-cell mean +/- std (n=3) registered as existence form; no CI constructed",
        },
    }

    template = preregistration["statistics"]["screening_statement_template"]
    screening_statement = template.format(**counts)
    report = {
        "stage": E4B_STAGE,
        "generated_at": now,
        "evidence_class": "measured (degenerate-cohort analysis; descriptive registrations only)",
        "inputs": {},
        "cohort_degeneracy": {
            "measured_counts": counts,
            "families": fill["families"],
            "lineages": fill["lineages"],
            "source": "e4a_measurement_report.json screening_fill (measured)",
        },
        "degeneracy_registrations": registrations,
        "e3_precedent_distinction": note["e3_precedent_distinction"],
        "incell_descriptive_spearman": incell,
        "top1_regret_degenerate": regret,
        "compute_break_even": break_even,
        "seed_level_uncertainty": seed_stats,
        "near_random": {
            "reference": "e4a_measurement_report.json near_random block (measured registration)",
            "first_tier": "8/8 cells single_entry_first_tier_not_applicable (n=1 entry per cell)",
            "second_tier": (
                "the frozen >= 2/5 second-tier clause (threshold 0.4) fires "
                "by construction on the random-baseline-only measured cohort "
                "(fraction=1.0): the random entry sits at its own reference "
                "level by construction. The trigger carries no empirical "
                "collapse meaning; it only weakens interpretation and never "
                "alters data, endpoints, or the frozen clause; it exposes a "
                "discriminative limit of the second-tier clause on "
                "single-lineage cohorts (descriptive registration, the "
                "clause itself is frozen and untouched)"
            ),
            "g6_annotation": dict(G6_ANNOTATION),
            "appendix": "runs/imagewoof/auditbench_e4/e4b_statistical_appendix_note.md",
        },
        "screening_statement": screening_statement,
        "statistical_uncertainty": {
            "design": (
                "no inferential statistics on the E4-standalone cohort "
                "(structurally_inapplicable_f1_l1); uncertainty is carried "
                "by the seed-level descriptive mean +/- std (n=3) existence "
                "form and by the POOLED4 lineage x dataset cluster bootstrap "
                "in the pooled4 report"
            ),
            "no_ci_constructed": True,
        },
    }
    return report, note, rows