"""AuditBench-DD E2: pre-registered statistical analysis of the E1 formal round.

This module implements the frozen E1 pre-registration
(``runs/cifar10/auditbench_e1/e1_preregistration.json``) literally:

- rows are built from the formal-round artifacts (19 primary entries), with
  family/lineage labels taken from the pre-registration cohort (never from
  file names);
- pooled leave-one-family-out (LOFO, 6 method families) x
  leave-one-architecture-out (LOAO, 4 student architectures) OOS prediction
  with per-(arch, ipc)-cell utility z-scores fitted on train folds only
  (cells with <2 train rows keep raw values; occurrences are counted);
- ridge alpha is re-selected inside every train fold by inner
  leave-one-family-out CV over the frozen grid;
- primary endpoint: IPC50 pooled family x arch OOS Spearman, computed per
  architecture cell against raw utility means and averaged over the 4
  architectures, per arm (base / base_plus_topo); IPC10 is the secondary;
- top-1 selection regret per (arch, ipc) cell vs random expected regret;
- gaussian random-column null (20 reps) through the same OOS pipeline;
- release_lineage-cluster bootstrap (1000 reps, full refit per replicate);
- compute break-even from recorded wall times; per-family descriptive
  Spearman; the frozen screening-statement template.

The structure follows ``cardc`` but no ``cardc`` internal function is called:
the three caliber differences (per-cell train-only z-score, lineage cluster
unit, IPC50-per-arch-then-mean aggregation) are implemented here directly and
listed in the report's implementation/pre-registration map.

Dataset parameterization (iter-0006): ``run_e2_analysis`` accepts optional
``stage``/``n_datasets``/``implementation_map`` arguments so the same frozen
harness executes the E2 (CIFAR-100) pre-registration with unchanged
statistical calibers; the E1 defaults keep the E1 behavior byte-identical.
The ``oos_prediction_records`` report block (entry x fold prediction plus
measured utility) is additive and supports the local <1e-12 endpoint
recomputation required by the E2 task.

"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audit import _oriented, _spearman
from .utils import save_json

ARMS = ("base", "base_plus_topo")
ARM_FEATURES = {
    "base": ("ridge_proxy", "knn_coverage", "class_moment_dist"),
    "base_plus_topo": (
        "ridge_proxy",
        "knn_coverage",
        "class_moment_dist",
        "topo_endpoint_dist",
    ),
}
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_REPS = 1000
RANDOM_REPS = 20
PRIMARY_IPC = 50
SECONDARY_IPC = 10
E2_SEED = 0

E1_REPORT_REQUIRED_KEYS = (
    "stage",
    "generated_at",
    "evidence_class",
    "preregistration_sha256",
    "cohort",
    "rows",
    "oos_predictions",
    "arms",
    "random_null",
    "bootstrap",
    "regret",
    "compute_break_even",
    "per_family_descriptive",
    "wall_time",
    "screening_statement",
    "implementation_preregistration_map",
    "statistical_uncertainty",
)


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def build_rows(
    preregistration: dict[str, Any],
    formal_dir: str | Path,
    progress: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build E2 rows from formal artifacts; labels come from the cohort JSON."""
    formal_dir = Path(formal_dir)
    rows: list[dict[str, Any]] = []
    for entry in preregistration["cohort"]["entries"]:
        entry_id = entry["entry_id"]
        scores_path = formal_dir / f"{entry_id}.scores.json"
        evaluation_path = formal_dir / f"{entry_id}.eval_input.evaluation.json"
        if not scores_path.exists():
            raise FileNotFoundError(f"formal scores missing: {scores_path}")
        if not evaluation_path.exists():
            raise FileNotFoundError(f"formal evaluation missing: {evaluation_path}")
        record = json.loads(scores_path.read_text(encoding="utf-8"))
        if record.get("smoke_only") is not False:
            raise ValueError(f"scores record is not a formal artifact: {scores_path}")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        utility = {
            item["architecture"]: {
                "mean": float(item["mean"]),
                "std": float(item["std"]),
                "n": int(item["n"]),
            }
            for item in evaluation["summary"]
        }
        scores = {
            key: value
            for key, value in record["scores"].items()
            if key not in ("nulls", "runtime_sec")
        }
        train_wall: dict[str, float] = {}
        if progress is not None:
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


def check_complete_case(
    rows: list[dict[str, Any]], archs: list[str], seeds: list[int]
) -> dict[str, Any]:
    """Completeness gate: all arm features non-None, all archs with n seeds."""
    required = sorted(set(ARM_FEATURES["base_plus_topo"]))
    problems = []
    for row in rows:
        missing = [name for name in required if row["scores"].get(name) is None]
        if missing:
            problems.append(f"{row['entry_id']}: missing scores {missing}")
        for arch in archs:
            utility = row["utility"].get(arch)
            if utility is None:
                problems.append(f"{row['entry_id']}: missing utility for {arch}")
            elif utility["n"] != len(seeds):
                problems.append(
                    f"{row['entry_id']}: {arch} has n={utility['n']} "
                    f"(expected {len(seeds)})"
                )
    return {"complete": not problems, "problems": problems, "n_rows": len(rows)}


# ---------------------------------------------------------------------------
# OOS machinery (implements the frozen resampling design; no cardc calls)
# ---------------------------------------------------------------------------


def _long_table(
    rows: list[dict[str, Any]],
    names: tuple[str, ...] | list[str],
    archs: list[str],
    extra_column: np.ndarray | None = None,
) -> dict[str, Any]:
    """Entry-major long table: one row per (entry, arch)."""
    long_rows = [(row, arch) for row in rows for arch in archs]
    features = np.array(
        [[_oriented(row["scores"], name) for name in names] for row, _ in long_rows],
        dtype=float,
    ).reshape(len(long_rows), len(names))
    if extra_column is not None:
        features = np.concatenate([features, extra_column.reshape(-1, 1)], axis=1)
    targets = np.array(
        [row["utility"][arch]["mean"] for row, arch in long_rows], dtype=float
    )
    return {
        "rows": long_rows,
        "features": features,
        "targets": targets,
        "families": np.array([row["method_family"] for row, _ in long_rows]),
        "archs": np.array([arch for _, arch in long_rows]),
        "ipcs": np.array([row["ipc"] for row, _ in long_rows]),
        "cells": np.array(
            [f"{arch}|{row['ipc']}" for row, arch in long_rows]
        ),
    }


def zscore_targets_train_only(
    raw_targets: np.ndarray,
    arch_arr: np.ndarray,
    ipc_arr: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Per-(arch, ipc)-cell z-score with parameters fitted on train rows only.

    Cells with fewer than 2 train rows keep raw target values for every row in
    the cell (train and held-out alike); the number of skipped cells is
    returned alongside the transformed targets.
    """
    targets = raw_targets.copy()
    skipped = 0
    for arch in np.unique(arch_arr):
        for ipc in np.unique(ipc_arr):
            cell = (arch_arr == arch) & (ipc_arr == ipc)
            cell_train = cell & train_mask
            if cell_train.sum() < 2:
                skipped += 1
                continue
            mean = float(raw_targets[cell_train].mean())
            std = max(float(raw_targets[cell_train].std()), 1e-8)
            targets[cell] = (raw_targets[cell] - mean) / std
    return targets, skipped


def _mean_cell_spearman(
    predictions: np.ndarray, targets: np.ndarray, cells: np.ndarray
) -> float | None:
    rhos = []
    for cell in np.unique(cells):
        mask = cells == cell
        if mask.sum() < 3:
            continue
        rho = _spearman(predictions[mask], targets[mask])
        if rho is not None:
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else None


def select_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    families: np.ndarray,
    cells: np.ndarray,
    alphas,
) -> float:
    """Inner leave-one-family-out CV: pick alpha by within-cell OOS Spearman."""
    best_alpha, best_score = alphas[0], -np.inf
    for alpha in alphas:
        oos = np.full(len(features), np.nan)
        for family in np.unique(families):
            train_mask = families != family
            test_mask = families == family
            if train_mask.sum() < 3 or not test_mask.any():
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[train_mask], targets[train_mask])
            oos[test_mask] = model.predict(features[test_mask])
        valid = ~np.isnan(oos)
        score = _mean_cell_spearman(oos[valid], targets[valid], cells[valid])
        if score is not None and score > best_score:
            best_alpha, best_score = alpha, score
    return float(best_alpha)


def oos_pooled_predictions(
    rows: list[dict[str, Any]],
    names: tuple[str, ...] | list[str],
    archs: list[str],
    alphas,
    extra_column: np.ndarray | None = None,
    return_details: bool = False,
) -> dict[str, Any]:
    """Frozen E1 outer design: LOFO (method family) x LOAO (architecture) pooled.

    For each (held_family, held_arch) fold the train set is every row whose
    family is not held and whose architecture is not held; the same artifact
    therefore never appears in train and test of one fold. Utility z-scores
    are per-(arch, ipc) cell and fitted on train rows only; alpha is selected
    by inner leave-one-family-out CV inside the train fold.
    """
    table = _long_table(rows, names, archs, extra_column=extra_column)
    features = table["features"]
    raw_targets = table["targets"]
    families = table["families"]
    arch_arr = table["archs"]
    ipc_arr = table["ipcs"]
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
            targets, skipped = zscore_targets_train_only(
                raw_targets, arch_arr, ipc_arr, train_mask
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
# Endpoints
# ---------------------------------------------------------------------------


def evaluate_predictions(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, float]],
    archs: list[str],
    ipcs: list[int],
) -> dict[str, Any]:
    """Per-(arch, ipc) cell Spearman and top-1 regret vs raw utility means."""
    out: dict[str, Any] = {"rho": {}, "regret": {}, "random_regret": {}}
    for arch in archs:
        preds = predictions.get(arch, {})
        for ipc in ipcs:
            cell_key = f"{arch}|{ipc}"
            cell_rows = [row for row in rows if row["ipc"] == ipc]
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


def endpoint_rho(
    evaluation: dict[str, Any], archs: list[str], ipc: int
) -> float | None:
    """Frozen aggregation: per-arch cell Spearman at ``ipc``, then arch mean."""
    values = [
        evaluation["rho"].get(f"{arch}|{ipc}")
        for arch in archs
        if evaluation["rho"].get(f"{arch}|{ipc}") is not None
    ]
    return float(np.mean(values)) if values else None


def regret_summary(
    evaluation: dict[str, Any], archs: list[str], ipcs: list[int]
) -> dict[str, Any]:
    """Top-1 regret vs random expected regret, per cell and aggregated."""
    cells = {}
    for arch in archs:
        for ipc in ipcs:
            cell_key = f"{arch}|{ipc}"
            if cell_key in evaluation["regret"]:
                cells[cell_key] = {
                    "regret": evaluation["regret"][cell_key],
                    "random_expected_regret": evaluation["random_regret"][cell_key],
                }
    regrets = [cell["regret"] for cell in cells.values()]
    randoms = [cell["random_expected_regret"] for cell in cells.values()]
    return {
        "cells": cells,
        "regret_mean": float(np.mean(regrets)) if regrets else None,
        "random_expected_regret_mean": float(np.mean(randoms)) if randoms else None,
    }


# ---------------------------------------------------------------------------
# Random null (negative control)
# ---------------------------------------------------------------------------


def random_null(
    rows: list[dict[str, Any]],
    archs: list[str],
    ipcs: list[int],
    alphas,
    rng: np.random.Generator,
    reps: int,
) -> dict[str, Any]:
    """Gaussian random column through the same OOS pipeline (negative control).

    Two variants per replicate: the random column alone (pure negative
    control, expected rho ~ 0) and the base arm plus the random column
    (checks that noise does not systematically improve the base arm). Alpha
    is re-selected inside every train fold, exactly as for the real arms.
    """
    records = {
        "pure": {"rho_mean": [], "regret_mean": [], "primary_rho": []},
        "base_plus_random": {"rho_mean": [], "regret_mean": [], "primary_rho": []},
    }
    for _ in range(reps):
        column = rng.normal(size=len(rows))
        long_column = np.repeat(column, len(archs))
        for variant, names in (
            ("pure", ()),
            ("base_plus_random", ARM_FEATURES["base"]),
        ):
            result = oos_pooled_predictions(
                rows, names, archs, alphas, extra_column=long_column
            )
            evaluation = evaluate_predictions(
                rows, result["predictions"], archs, ipcs
            )
            if evaluation["rho_mean"] is not None:
                records[variant]["rho_mean"].append(evaluation["rho_mean"])
            if evaluation["regret_mean"] is not None:
                records[variant]["regret_mean"].append(evaluation["regret_mean"])
            primary = endpoint_rho(evaluation, archs, PRIMARY_IPC)
            if primary is not None:
                records[variant]["primary_rho"].append(primary)
    summary: dict[str, Any] = {"reps": reps, "variants": {}}
    for variant, values in records.items():
        summary["variants"][variant] = {
            "rho_mean": _distribution_summary(values["rho_mean"], "p95"),
            "regret_mean": _distribution_summary(values["regret_mean"], "p05"),
            "primary_rho": _distribution_summary(values["primary_rho"], "p95"),
        }
    return summary


def _distribution_summary(values: list[float], tail: str) -> dict[str, Any]:
    if not values:
        return {"mean": None, tail: None, "n": 0}
    percentile = 95 if tail == "p95" else 5
    return {
        "mean": float(np.mean(values)),
        tail: float(np.percentile(values, percentile)),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Lineage-cluster bootstrap
# ---------------------------------------------------------------------------


def resample_lineage_clusters(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One bootstrap replicate: resample release_lineage clusters with replacement.

    Every draw copies the whole lineage; copies get a ``#draw`` suffix on both
    entry_id and release_lineage while method_family stays unchanged, so the
    LOFO folds keep every copy of an artifact on the same side of a fold.
    """
    lineages = sorted({row["release_lineage"] for row in rows})
    by_lineage = {
        lineage: [row for row in rows if row["release_lineage"] == lineage]
        for lineage in lineages
    }
    sampled = rng.choice(lineages, size=len(lineages), replace=True)
    counts = {lineage: int((sampled == lineage).sum()) for lineage in lineages}
    pseudo: list[dict[str, Any]] = []
    for draw, lineage in enumerate(sampled):
        for row in by_lineage[lineage]:
            pseudo.append(
                dict(
                    row,
                    entry_id=f"{row['entry_id']}#{draw}",
                    release_lineage=f"{row['release_lineage']}#{draw}",
                )
            )
    return pseudo, counts


def lineage_bootstrap(
    rows: list[dict[str, Any]],
    archs: list[str],
    ipcs: list[int],
    alphas,
    rng: np.random.Generator,
    reps: int,
    log_every: int = 100,
) -> dict[str, Any]:
    """Release_lineage-cluster bootstrap with a full pipeline refit per replicate.

    Every replicate re-runs: per-cell train-only z-score, inner-CV alpha
    selection, ridge fit, OOS prediction and endpoint aggregation, for both
    arms. Reports 2.5/97.5 percentile CIs for each arm's primary/secondary
    rho and for the delta (base_plus_topo - base), with frac_positive.
    """
    per_arm = {
        arm: {"primary": [], "secondary": []} for arm in ARMS
    }
    delta = {"primary": [], "secondary": []}
    skipped = 0
    for rep in range(reps):
        pseudo, _counts = resample_lineage_clusters(rows, rng)
        endpoints: dict[str, dict[str, float | None]] = {}
        for arm in ARMS:
            result = oos_pooled_predictions(
                pseudo, ARM_FEATURES[arm], archs, alphas
            )
            evaluation = evaluate_predictions(
                pseudo, result["predictions"], archs, ipcs
            )
            endpoints[arm] = {
                "primary": endpoint_rho(evaluation, archs, PRIMARY_IPC),
                "secondary": endpoint_rho(evaluation, archs, SECONDARY_IPC),
            }
        if any(endpoints[arm]["primary"] is None for arm in ARMS) or any(
            endpoints[arm]["secondary"] is None for arm in ARMS
        ):
            skipped += 1
            continue
        for arm in ARMS:
            per_arm[arm]["primary"].append(endpoints[arm]["primary"])
            per_arm[arm]["secondary"].append(endpoints[arm]["secondary"])
        for key in ("primary", "secondary"):
            delta[key].append(
                endpoints["base_plus_topo"][key] - endpoints["base"][key]
            )
        if log_every and (rep + 1) % log_every == 0:
            print(f"[bootstrap] {rep + 1}/{reps} replicates", flush=True)

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
        "cluster_unit": "release_lineage",
        "reps_requested": reps,
        "reps_used": len(delta["primary"]),
        "reps_skipped_undefined": skipped,
        "arms": {
            arm: {"primary": _ci(per_arm[arm]["primary"]),
                  "secondary": _ci(per_arm[arm]["secondary"])}
            for arm in ARMS
        },
        "delta_base_plus_topo_minus_base": {
            key: {**_ci(values), "frac_positive": float(np.mean([v > 0 for v in values])) if values else None}
            for key, values in delta.items()
        },
    }


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------


def per_family_descriptive(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, float]],
    archs: list[str],
    ipcs: list[int],
) -> dict[str, Any]:
    """Per-family pooled OOS Spearman (descriptive only, no inference).

    Utility is z-scored within each full-cohort (arch, ipc) cell so archs are
    on a common scale, then pooled over the family's entries x archs per ipc.
    """
    z_utility: dict[tuple[str, str], float] = {}
    for arch in archs:
        for ipc in ipcs:
            cell = [row for row in rows if row["ipc"] == ipc]
            values = np.array([row["utility"][arch]["mean"] for row in cell])
            std = max(float(values.std()), 1e-8)
            mean = float(values.mean())
            for row in cell:
                z_utility[(row["entry_id"], arch)] = (
                    row["utility"][arch]["mean"] - mean
                ) / std
    out: dict[str, Any] = {}
    for family in sorted({row["method_family"] for row in rows}):
        family_rows = [row for row in rows if row["method_family"] == family]
        per_ipc: dict[str, Any] = {}
        for ipc in ipcs:
            xs, ys = [], []
            for row in family_rows:
                if row["ipc"] != ipc:
                    continue
                for arch in archs:
                    pred = predictions.get(arch, {}).get(row["entry_id"])
                    if pred is None:
                        continue
                    xs.append(pred)
                    ys.append(z_utility[(row["entry_id"], arch)])
            if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
                per_ipc[str(ipc)] = {
                    "rho": _spearman(np.array(xs), np.array(ys)),
                    "n": len(xs),
                }
            else:
                per_ipc[str(ipc)] = {"rho": None, "n": len(xs)}
        out[family] = {
            "per_ipc": per_ipc,
            "n_entries": len(family_rows),
            "note": "descriptive only; not an inferential estimate",
        }
    return out


def compute_break_even(
    rows: list[dict[str, Any]],
    selections: dict[str, dict[str, dict[str, float]]],
    archs: list[str],
    ipcs: list[int],
    seeds: list[int],
) -> dict[str, Any]:
    """Proxy extraction wall time vs student training wall time.

    ``selections`` maps arm -> arch -> entry_id -> OOS prediction and is used
    to cost the pre-registered screening strategy: rank by proxy, train only
    the top-1 artifact of each (arch, ipc) cell at the frozen 3-seed budget.
    """
    proxy_total = float(sum(row["proxy_runtime_sec"] for row in rows))
    cell_wall: dict[tuple[str, str, int], float] = {}
    coverage = 0
    expected = len(rows) * len(archs) * len(seeds)
    for row in rows:
        for arch in archs:
            for seed in seeds:
                key = f"{arch}|{seed}"
                if key in row["train_wall_sec"]:
                    cell_wall[(row["entry_id"], arch, seed)] = row["train_wall_sec"][key]
                    coverage += 1
    train_total = float(sum(cell_wall.values()))
    per_arch = {
        arch: {
            "total_sec": float(
                sum(v for (e, a, s), v in cell_wall.items() if a == arch)
            ),
            "mean_cell_sec": float(
                np.mean([v for (e, a, s), v in cell_wall.items() if a == arch])
            )
            if any(a == arch for (e, a, s) in cell_wall)
            else None,
        }
        for arch in archs
    }
    strategies: dict[str, Any] = {}
    for arm, predictions in selections.items():
        selected_cost = 0.0
        selected: dict[str, str] = {}
        for arch in archs:
            for ipc in ipcs:
                cell = [row for row in rows if row["ipc"] == ipc]
                scored = [
                    (predictions.get(arch, {}).get(row["entry_id"]), row)
                    for row in cell
                ]
                scored = [(p, row) for p, row in scored if p is not None]
                if not scored:
                    continue
                top = max(scored, key=lambda item: item[0])[1]
                selected[f"{arch}|{ipc}"] = top["entry_id"]
                for seed in seeds:
                    selected_cost += cell_wall.get((top["entry_id"], arch, seed), 0.0)
        selective_total = proxy_total + selected_cost
        strategies[arm] = {
            "selected_top1": selected,
            "proxy_total_sec": proxy_total,
            "selected_train_sec": float(selected_cost),
            "selective_total_sec": float(selective_total),
            "full_matrix_train_sec": train_total,
            "saving_ratio_vs_full_matrix": (
                float(1.0 - selective_total / train_total) if train_total > 0 else None
            ),
        }
    ratio = proxy_total / train_total if train_total > 0 else None
    return {
        "proxy_total_sec": proxy_total,
        "train_total_sec": train_total,
        "n_train_cells_measured": coverage,
        "n_train_cells_expected": expected,
        "per_arch": per_arch,
        "proxy_to_full_matrix_ratio": ratio,
        "break_even_reading": (
            "proxy extraction costs "
            f"{ratio:.4f} of one full 4-arch x 3-seed matrix pass; a screening "
            "strategy that trains only the proxy top-1 per (arch, ipc) cell "
            "pays off whenever its utility loss versus the full-matrix best is "
            "worth less than the avoided training share."
            if ratio is not None
            else "training wall times unavailable"
        ),
        "strategies": strategies,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _prediction_records(
    rows: list[dict[str, Any]],
    archs: list[str],
    folds: list[dict[str, Any]],
    predictions: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Entry x fold OOS prediction records for local endpoint recomputation.

    The long-table ordering is the deterministic entry-major order of
    ``_long_table`` (one row per (entry, arch)); every test index of a fold
    maps back to its (row, arch) pair through the same enumeration.
    """
    long_rows = [(row, arch) for row in rows for arch in archs]
    records: list[dict[str, Any]] = []
    for fold in folds:
        for index in fold["test_index"]:
            row, arch = long_rows[index]
            records.append(
                {
                    "entry_id": row["entry_id"],
                    "arch": arch,
                    "ipc": row["ipc"],
                    "method_family": row["method_family"],
                    "release_lineage": row["release_lineage"],
                    "held_family": fold["held_family"],
                    "held_arch": fold["held_arch"],
                    "alpha": fold["alpha"],
                    "prediction": predictions[arch][row["entry_id"]],
                    "utility_mean": row["utility"][arch]["mean"],
                    "utility_std": row["utility"][arch]["std"],
                }
            )
    return records


def validate_report(report: dict[str, Any], stage: str = "auditbench_e1_report") -> None:
    """Schema check for a machine-readable report of this analysis stage."""
    missing = [key for key in E1_REPORT_REQUIRED_KEYS if key not in report]
    if missing:
        raise ValueError(f"report missing keys: {missing}")
    if report.get("stage") != stage:
        raise ValueError(f"report stage must be {stage!r}")
    for arm in ARMS:
        if arm not in report["arms"]:
            raise ValueError(f"report arms missing arm {arm!r}")
        for field in ("primary_rho_ipc50", "secondary_rho_ipc10"):
            if field not in report["arms"][arm]:
                raise ValueError(f"report arm {arm!r} missing {field!r}")
    if report["random_null"].get("reps") is None:
        raise ValueError("report random_null missing reps")
    if report["bootstrap"].get("reps_requested") is None:
        raise ValueError("report bootstrap missing reps_requested")


def validate_e1_report(report: dict[str, Any]) -> None:
    """Schema check for the machine-readable E1 report (frozen stage name)."""
    validate_report(report, stage="auditbench_e1_report")


def _rows_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "entry_id": row["entry_id"],
            "release_lineage": row["release_lineage"],
            "method_family": row["method_family"],
            "ipc": row["ipc"],
            "scores": row["scores"],
            "utility": row["utility"],
            "proxy_runtime_sec": row["proxy_runtime_sec"],
        }
        for row in rows
    ]


def run_e2_analysis(
    preregistration_path: str | Path,
    formal_dir: str | Path,
    out_path: str | Path | None = None,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    random_reps: int = RANDOM_REPS,
    seed: int = E2_SEED,
    progress_name: str = "formal_progress.json",
    stage: str = "auditbench_e1_report",
    n_datasets: int = 1,
    implementation_map: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the full pre-registered analysis and write the machine-readable report.

    Defaults reproduce the frozen E1 run exactly; ``stage``/``n_datasets``/
    ``implementation_map`` parameterize the same harness for the E2
    (CIFAR-100) pre-registration without touching any E1 frozen value.
    """
    from datetime import datetime, timezone

    preregistration_path = Path(preregistration_path)
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    prereg_sha = hashlib.sha256(
        preregistration_path.read_bytes()
    ).hexdigest()
    formal_dir = Path(formal_dir)
    progress_path = formal_dir / progress_name
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else None
    )
    rows = build_rows(preregistration, formal_dir, progress)
    archs = list(preregistration["student_architectures"]["set"])
    seeds = list(preregistration["training_budget"]["seeds"])
    ipcs = sorted({row["ipc"] for row in rows})
    completeness = check_complete_case(rows, archs, seeds)
    if not completeness["complete"]:
        raise RuntimeError(f"E1 formal matrix incomplete: {completeness['problems']}")

    arms_out: dict[str, Any] = {}
    predictions_out: dict[str, Any] = {}
    alphas_out: dict[str, Any] = {}
    prediction_records: dict[str, Any] = {}
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
            arch: dict(sorted(preds.items()))
            for arch, preds in predictions.items()
        }
        prediction_records[arm] = _prediction_records(
            rows, archs, result["details"]["folds"], predictions
        )

    null_rng = np.random.default_rng(seed)
    null = random_null(rows, archs, ipcs, ALPHA_GRID, null_rng, random_reps)

    boot_rng = np.random.default_rng(seed + 1)
    bootstrap = lineage_bootstrap(
        rows, archs, ipcs, ALPHA_GRID, boot_rng, bootstrap_reps
    )

    break_even = compute_break_even(rows, predictions_out, archs, ipcs, seeds)
    per_family = {
        arm: per_family_descriptive(rows, predictions_out[arm], archs, ipcs)
        for arm in ARMS
    }

    template = preregistration["statistics"]["screening_statement_template"]
    screening_statement = template.format(
        n=len(rows),
        f=len({row["method_family"] for row in rows}),
        l=len({row["release_lineage"] for row in rows}),
        d=n_datasets,
    )

    wall_time = {
        "proxy_total_sec": break_even["proxy_total_sec"],
        "train_total_sec": break_even["train_total_sec"],
        "n_train_cells_measured": break_even["n_train_cells_measured"],
        "per_arch": break_even["per_arch"],
        "started_at": (progress or {}).get("started_at"),
        "updated_at": (progress or {}).get("updated_at"),
    }

    report: dict[str, Any] = {
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "measured",
        "preregistration_sha256": prereg_sha,
        "cohort": {
            "n_entries": len(rows),
            "entry_ids": [row["entry_id"] for row in rows],
            "families": sorted({row["method_family"] for row in rows}),
            "lineages": sorted({row["release_lineage"] for row in rows}),
            "architectures": archs,
            "seeds": seeds,
            "ipcs": ipcs,
            "n_datasets": n_datasets,
            "completeness": completeness,
        },
        "rows": _rows_summary(rows),
        "oos_predictions": predictions_out,
        "oos_prediction_records": prediction_records,
        "arms": arms_out,
        "alpha_selection": {
            "grid": list(ALPHA_GRID),
            "chosen_per_fold": alphas_out,
        },
        "random_null": null,
        "bootstrap": bootstrap,
        "regret": {arm: arms_out[arm]["regret"] for arm in ARMS},
        "compute_break_even": break_even,
        "per_family_descriptive": per_family,
        "wall_time": wall_time,
        "screening_statement": screening_statement,
        "implementation_preregistration_map": (
            IMPLEMENTATION_MAP if implementation_map is None else implementation_map
        ),
        "statistical_uncertainty": {
            "design": (
                "lineage-cluster bootstrap (1000 reps, full refit per replicate) "
                "for grouped CIs; gaussian random-column null (20 reps) as "
                "negative control; LOFO has only "
                f"{len({row['method_family'] for row in rows})} family-level "
                "clusters, so all endpoints are screening-level"
            ),
            "zscore_skipped_cells_total": zscore_skipped_total,
            "bootstrap_reps_used": bootstrap["reps_used"],
            "bootstrap_reps_skipped_undefined": bootstrap[
                "reps_skipped_undefined"
            ],
            "random_null_reps": random_reps,
            "utility_std_descriptive": {
                row["entry_id"]: {
                    arch: row["utility"][arch]["std"] for arch in archs
                }
                for row in rows
            },
        },
        "rng_seed": {"random_null": seed, "bootstrap": seed + 1},
    }
    validate_report(report, stage=stage)
    if out_path is not None:
        out_path = Path(out_path)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        save_json(report, tmp)
        tmp.replace(out_path)
    return report


IMPLEMENTATION_MAP = [
    {
        "preregistration": "rows from the 19-entry primary cohort with family/lineage labels",
        "implementation": "build_rows(): labels read from e1_preregistration.json cohort, never from file names; utility = per-(arch) {mean, std, n} over seeds {0,1,2} from pipeline.evaluate summaries",
    },
    {
        "preregistration": "utility normalization: per-(arch, ipc)-cell z-score fitted on train folds only; cells with <2 train rows keep raw values, occurrences recorded",
        "implementation": "zscore_targets_train_only(): parameters from train rows only, applied to every row of the cell; skipped cells counted (zscore_skipped_cells); differs from cardc._design zscore (whole-cohort per arch) — per-cell is implemented here",
    },
    {
        "preregistration": "outer design: LOFO (6 families) x LOAO (4 archs) pooled OOS (cardc._oos_pooled_v2 scheme)",
        "implementation": "oos_pooled_predictions(): same fold structure as _oos_pooled_v2 but implemented locally; cluster/label semantics differ from cardc v2, which has no release_lineage concept",
    },
    {
        "preregistration": "alpha re-selection inside train folds over the frozen grid (0.01, 0.1, 1, 10, 100)",
        "implementation": "select_alpha(): inner leave-one-family-out CV scored by mean within-(arch, ipc)-cell Spearman, same semantics as cardc._select_alpha",
    },
    {
        "preregistration": "primary endpoint: IPC50 pooled family x arch OOS Spearman per arch cell vs raw utility mean, averaged over 4 archs, per arm",
        "implementation": "endpoint_rho(evaluation, archs, 50); differs from cardc v2 aggregation (rho_mean over all arch x ipc cells) — the frozen IPC50-per-arch-then-mean aggregation is used",
    },
    {
        "preregistration": "top-1 regret = best_utility - utility(argmax proxy); random_expected_regret = best - mean within the (arch, ipc) cell",
        "implementation": "evaluate_predictions() + regret_summary(); both arms and the random null are reported side by side",
    },
    {
        "preregistration": "random null: gaussian random score column through the same OOS pipeline (20 reps)",
        "implementation": "random_null(): variant 'pure' uses the random column as the only feature (negative control); variant 'base_plus_random' appends it to the base arm; rho_mean mean/p95 and regret_mean mean/p05 reported",
    },
    {
        "preregistration": "release_lineage-cluster bootstrap, 1000 reps, full refit per replicate; same artifact never in train and test of one resample",
        "implementation": "lineage_bootstrap()+resample_lineage_clusters(): lineage-granular resampling with #draw suffixes, family labels unchanged so folds keep copies together; cell z-score, alpha selection, fit, prediction and endpoint aggregation re-run per replicate",
    },
    {
        "preregistration": "compute break-even: per-artifact proxy wall time vs per-artifact x arch x seed training wall time",
        "implementation": "compute_break_even(): runtime_sec from scores.json (embedding + scores + short_run_5ep) vs per-cell wall_sec from formal_progress.json; top-1 selective-training strategy costed per arm",
    },
    {
        "preregistration": "per-family OOS Spearman, descriptive only (6 clusters)",
        "implementation": "per_family_descriptive(): within-(arch, ipc)-cell z-scored utility pooled per family per ipc; labelled descriptive only",
    },
    {
        "preregistration": "screening statement template with n=19, f=6, l=7",
        "implementation": "statistics.screening_statement_template filled verbatim from the cohort counts",
    },
]


E2_IMPLEMENTATION_MAP = [
    {
        "preregistration": "rows from the 15-entry primary cohort with family/lineage labels",
        "implementation": "build_rows(): labels read from e2_preregistration.json cohort, never from file names; utility = per-(arch) {mean, std, n} over seeds {0,1,2} from pipeline.evaluate summaries",
    },
    {
        "preregistration": "utility normalization: per-(arch, ipc)-cell z-score fitted on train folds only; cells with <2 train rows keep raw values, occurrences recorded",
        "implementation": "zscore_targets_train_only(): parameters from train rows only, applied to every row of the cell; skipped cells counted (zscore_skipped_cells); identical caliber to the E1 formal analysis",
    },
    {
        "preregistration": "outer design: LOFO (6 families) x LOAO (4 archs) pooled OOS (cardc._oos_pooled_v2 scheme)",
        "implementation": "oos_pooled_predictions(): same fold structure as the E1 formal analysis; cluster/label semantics unchanged",
    },
    {
        "preregistration": "alpha re-selection inside train folds over the frozen grid (0.01, 0.1, 1, 10, 100)",
        "implementation": "select_alpha(): inner leave-one-family-out CV scored by mean within-(arch, ipc)-cell Spearman, identical to E1",
    },
    {
        "preregistration": "primary endpoint: IPC50 pooled family x arch OOS Spearman per arch cell vs raw utility mean, averaged over 4 archs, per arm",
        "implementation": "endpoint_rho(evaluation, archs, 50); the frozen IPC50-per-arch-then-mean aggregation; the E2 IPC50 subset is 6 entries / 6 lineages (idc and vico have no CIFAR-100 IPC50 artifact)",
    },
    {
        "preregistration": "top-1 regret = best_utility - utility(argmax proxy); random_expected_regret = best - mean within the (arch, ipc) cell",
        "implementation": "evaluate_predictions() + regret_summary(); both arms and the random null are reported side by side",
    },
    {
        "preregistration": "random null: gaussian random score column through the same OOS pipeline (20 reps)",
        "implementation": "random_null(): variant 'pure' uses the random column as the only feature (negative control); variant 'base_plus_random' appends it to the base arm; rho_mean mean/p95 and regret_mean mean/p05 reported",
    },
    {
        "preregistration": "release_lineage-cluster bootstrap, 1000 reps, full refit per replicate; same artifact never in train and test of one resample",
        "implementation": "lineage_bootstrap()+resample_lineage_clusters(): lineage-granular resampling with #draw suffixes, family labels unchanged so folds keep copies together; cell z-score, alpha selection, fit, prediction and endpoint aggregation re-run per replicate",
    },
    {
        "preregistration": "compute break-even: per-artifact proxy wall time vs per-artifact x arch x seed training wall time",
        "implementation": "compute_break_even(): runtime_sec from scores.json (embedding + scores + short_run_5ep) vs per-cell wall_sec from formal_progress.json; top-1 selective-training strategy costed per arm",
    },
    {
        "preregistration": "per-family OOS Spearman, descriptive only (6 clusters)",
        "implementation": "per_family_descriptive(): within-(arch, ipc)-cell z-scored utility pooled per family per ipc; labelled descriptive only",
    },
    {
        "preregistration": "screening statement template with n=15, f=6, l=7, d=1",
        "implementation": "statistics.screening_statement_template filled verbatim from the cohort counts (four-parameter version)",
    },
]
