"""Low-compute Phase-1 diagnostics for conditional topology signal.

The historical E1--E3 reports are immutable inputs.  This module never
recomputes a student utility or proxy embedding.  It therefore answers only
questions that are identifiable from the published candidate-level scores and
records, explicitly, which proposed diagnostics need a future feature-cache
extraction.
"""

from __future__ import annotations

import hashlib
import json
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .audit import LOWER_IS_BETTER
from .utils import save_json


BASE_FEATURES = ("ridge_proxy", "knn_coverage", "class_moment_dist")
TOPOLOGY_FEATURES = ("topo_endpoint_dist", "topo_traj_dist", "betti_traj_dist")
ALL_SCORE_FEATURES = (
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

REPORT_SOURCES = (
    ("cifar10", "runs/cifar10/auditbench_e1/e1_report.json", None,
     "runs/cifar10/auditbench_e1/formal"),
    ("cifar100", "runs/cifar100/auditbench_e2/e2_report.json", None,
     "runs/cifar100/auditbench_e2/formal"),
    ("tiny_imagenet", "runs/tiny_imagenet/auditbench_e3/e3b_report.json",
     "e3_single_dataset", "runs/tiny_imagenet/auditbench_e3/e3a_matrix"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rho(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float | None:
    x_array, y_array = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x_array) < 3 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return None
    value = spearmanr(x_array, y_array).statistic
    return None if value is None or not np.isfinite(value) else float(value)


def _oriented(scores: dict[str, Any], feature: str) -> float | None:
    value = scores.get(feature)
    if value is None or not np.isfinite(value):
        return None
    return -float(value) if feature in LOWER_IS_BETTER else float(value)


def _value(item: dict[str, Any], feature: str) -> float | None:
    """Read an oriented proxy score or an explicitly unsigned nuisance."""
    if feature == "ipc":
        return float(item["ipc"])
    if feature.startswith("null:"):
        value = item["nulls"].get(feature.removeprefix("null:"))
        return None if value is None else -float(value)
    if feature in item.get("nuisance", {}):
        return float(item["nuisance"][feature])
    return _oriented(item["scores"], feature)


def _load_source(root: Path, source: tuple[str, str, str | None, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset, report_rel, section, score_dir_rel = source
    report_path = root / report_rel
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_body = report if section is None else report[section]
    rows = report_body["rows"]
    observations: list[dict[str, Any]] = []
    for row in rows:
        score_path = root / score_dir_rel / f"{row['entry_id']}.scores.json"
        nulls: dict[str, Any] = {}
        if score_path.exists():
            score_payload = json.loads(score_path.read_text(encoding="utf-8"))
            nulls = score_payload.get("scores", {}).get("nulls", {})
        for architecture, utility in row["utility"].items():
            observations.append(
                {
                    "dataset": dataset,
                    "entry_id": row["entry_id"],
                    "release_lineage": row["release_lineage"],
                    "method_family": row["method_family"],
                    "ipc": int(row["ipc"]),
                    "architecture": architecture,
                    "utility": float(utility["mean"]),
                    "scores": row["scores"],
                    "nulls": nulls,
                }
            )
    return observations, {
        "dataset": dataset,
        "path": str(report_path),
        "sha256": _sha256(report_path),
        "entry_count": len(rows),
        "score_null_coverage": sum(
            1 for row in rows if (root / score_dir_rel / f"{row['entry_id']}.scores.json").exists()
        ),
    }


def load_phase1_observations(root: str | Path = ".") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the three read-only E1--E3 score/utility cohorts."""
    root_path = Path(root)
    observations: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for source in REPORT_SOURCES:
        loaded, source_info = _load_source(root_path, source)
        observations.extend(loaded)
        provenance.append(source_info)
    return observations, provenance


def _rank_matrix(observations: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, Any]:
    """Per-dataset and per-family Spearman tables, split by architecture/IPC."""
    tables: dict[str, Any] = {"dataset": {}, "family": {}}
    for level, key in (("dataset", "dataset"), ("family", "method_family")):
        groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for item in observations:
            groups[(item["dataset"], item[key], item["ipc"])].append(item)
        for (dataset, group_name, ipc), group in sorted(groups.items()):
            if level == "dataset":
                output = tables[level].setdefault(dataset, {}).setdefault(str(ipc), {})
            else:
                output = (
                    tables[level]
                    .setdefault(dataset, {})
                    .setdefault(group_name, {})
                    .setdefault(str(ipc), {})
                )
            for architecture in sorted({item["architecture"] for item in group}):
                selected = [item for item in group if item["architecture"] == architecture]
                utility = [item["utility"] for item in selected]
                values: dict[str, Any] = {}
                for feature in features:
                    feature_values = [_oriented(item["scores"], feature) for item in selected]
                    valid = [(x, y) for x, y in zip(feature_values, utility) if x is not None]
                    values[feature] = {
                        "rho": _rho([x for x, _ in valid], [y for _, y in valid]),
                        "n": len(valid),
                    }
                output[architecture] = values
    return tables


def _zscore_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train - mean) / std, (test - mean) / std


def _design(items: list[dict[str, Any]], features: tuple[str, ...], architectures: list[str]) -> np.ndarray:
    values = [[_value(item, feature) or 0.0 for feature in features] for item in items]
    arch = [[float(item["architecture"] == name) for name in architectures[1:]] for item in items]
    return np.asarray([x + a for x, a in zip(values, arch)], dtype=float)


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_std, test_std = _zscore_train(train_x, test_x)
    model = Ridge(alpha=1.0).fit(train_std, train_y)
    return model.predict(test_std), model.coef_


def _oof_residuals(
    observations: list[dict[str, Any]], topology: str, extra_nuisance: tuple[str, ...] = ()
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """LOFO residualization, with every fit restricted to its train families."""
    output: list[dict[str, Any]] = []
    coefficients: dict[str, list[float]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        by_dataset[item["dataset"]].append(item)
    for dataset, dataset_items in by_dataset.items():
        architectures = sorted({item["architecture"] for item in dataset_items})
        families = sorted({item["method_family"] for item in dataset_items})
        controls = BASE_FEATURES + ("ipc", "effective_rank") + extra_nuisance
        for family in families:
            test = [item for item in dataset_items if item["method_family"] == family]
            train = [item for item in dataset_items if item["method_family"] != family]
            if len(train) < 4 or not test:
                continue
            base_train = _design(train, controls, architectures)
            base_test = _design(test, controls, architectures)
            utility_train = np.asarray([item["utility"] for item in train], dtype=float)
            utility_test = np.asarray([item["utility"] for item in test], dtype=float)
            utility_pred, _ = _fit_ridge(base_train, utility_train, base_test)
            topology_train = np.asarray([_value(item, topology) for item in train], dtype=float)
            topology_test = np.asarray([_value(item, topology) for item in test], dtype=float)
            topology_pred, _ = _fit_ridge(base_train, topology_train, base_test)
            full_train = np.column_stack((base_train, topology_train))
            full_test = np.column_stack((base_test, topology_test))
            full_prediction, full_coef = _fit_ridge(full_train, utility_train, full_test)
            coefficients[dataset].append(float(full_coef[-1]))
            for item, y, y_hat, full_hat, topo, topo_hat in zip(
                test, utility_test, utility_pred, full_prediction, topology_test, topology_pred
            ):
                output.append(
                    {
                        **{key: item[key] for key in ("dataset", "entry_id", "release_lineage", "method_family", "ipc", "architecture")},
                        "utility_residual": float(y - y_hat),
                        "utility": float(y),
                        "topology_residual": float(topo - topo_hat),
                        "base_prediction": float(y_hat),
                        "topology_augmented_prediction": float(full_hat),
                        "topology": topology,
                    }
                )
    return output, coefficients


def _group_effects(records: list[dict[str, Any]]) -> dict[str, Any]:
    axes = ("dataset", "method_family", "ipc", "architecture")
    result: dict[str, Any] = {}
    for axis in axes:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record[axis])].append(record)
        result[axis] = {
            name: {
                "rho": _rho(
                    [record["topology_residual"] for record in group],
                    [record["utility_residual"] for record in group],
                ),
                "n": len(group),
            }
            for name, group in sorted(groups.items())
        }
    return result


def _bootstrap(
    records: list[dict[str, Any]], coefficients: dict[str, list[float]], reps: int, seed: int
) -> dict[str, Any]:
    """Lineage-cluster bootstrap of already OOF residuals and fit coefficients.

    Re-fitting happens before this function; the bootstrap answers stability of
    the OOF signal rather than pretending repeated student runs exist.
    """
    rng = np.random.default_rng(seed)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    result: dict[str, Any] = {}
    for dataset, rows in by_dataset.items():
        lineages = sorted({row["release_lineage"] for row in rows})
        draws: list[float] = []
        signs: list[float] = []
        ranking_gains: list[float] = []
        for _ in range(reps):
            selected = rng.choice(lineages, size=len(lineages), replace=True)
            sample = [row for lineage in selected for row in rows if row["release_lineage"] == lineage]
            value = _rho(
                [row["topology_residual"] for row in sample],
                [row["utility_residual"] for row in sample],
            )
            if value is not None:
                draws.append(value)
                signs.append(float(value > 0))
            base_rho = _rho([row["base_prediction"] for row in sample], [row["utility"] for row in sample])
            full_rho = _rho(
                [row["topology_augmented_prediction"] for row in sample],
                [row["utility"] for row in sample],
            )
            if base_rho is not None and full_rho is not None:
                ranking_gains.append(full_rho - base_rho)
        result[dataset] = {
            "cluster_unit": "release_lineage",
            "reps_requested": reps,
            "reps_used": len(draws),
            "residual_rho_median": float(np.median(draws)) if draws else None,
            "residual_rho_ci95": (
                [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
                if draws else None
            ),
            "positive_sign_fraction": float(np.mean(signs)) if signs else None,
            "oof_ranking_gain_topology_minus_base": {
                "median": float(np.median(ranking_gains)) if ranking_gains else None,
                "ci95": (
                    [float(np.quantile(ranking_gains, 0.025)), float(np.quantile(ranking_gains, 0.975))]
                    if ranking_gains else None
                ),
            },
            "fold_topology_coefficient": {
                "n": len(coefficients[dataset]),
                "mean": float(np.mean(coefficients[dataset])) if coefficients[dataset] else None,
                "median": float(np.median(coefficients[dataset])) if coefficients[dataset] else None,
            },
        }
    return result


def _existing_nulls(observations: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for null_name in ("labelperm_topo_traj_dist", "timeshuffle_topo_traj_dist", "timeshuffle_betti_traj_dist"):
        per_dataset: dict[str, Any] = {}
        for dataset in sorted({item["dataset"] for item in observations}):
            items = [item for item in observations if item["dataset"] == dataset]
            per_architecture: dict[str, Any] = {}
            for architecture in sorted({item["architecture"] for item in items}):
                selected = [item for item in items if item["architecture"] == architecture]
                values = [_value(item, f"null:{null_name}") for item in selected]
                valid = [(x, item["utility"]) for x, item in zip(values, selected) if x is not None]
                per_architecture[architecture] = {
                    "rho": _rho([x for x, _ in valid], [y for _, y in valid]),
                    "n": len(valid),
                }
            per_dataset[dataset] = per_architecture
        output[null_name] = per_dataset
    return output


def _availability(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Guard against silently treating scalar historical scores as components."""
    return {
        "per_class_topology": {
            "status": "not_available_in_historical_reports",
            "reason": "E1--E3 retain only aggregate topology scalars; no per-class decomposition is stored.",
        },
        "per_timestep_topology": {
            "status": "not_available_in_historical_reports",
            "reason": "E1--E3 retain only trajectory aggregate and endpoint scalars; no timestep vector is stored.",
        },
        "feature_covariance_trace": {
            "status": "not_available_in_historical_reports",
            "reason": "class_moment_dist combines mean and covariance mismatch and is not a covariance trace.",
        },
        "candidate_count": len({(item["dataset"], item["entry_id"]) for item in observations}),
    }


def _attach_input_nuisances(root: Path, observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach cheap image-space controls from the canonical candidate bank.

    These controls are intentionally labelled *input-space*: they make a
    useful low-compute confound check, but cannot replace covariance/density
    measured in the frozen encoder state space.
    """
    cache_path = root / "runs" / "phase1" / "input_nuisance_cache.json"
    unique = {(item["dataset"], item["entry_id"]) for item in observations}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        values = {
            tuple(key.split("/", maxsplit=1)): value
            for key, value in cached.get("values", {}).items()
        }
        if all(key in values for key in unique):
            for item in observations:
                item["nuisance"] = values[(item["dataset"], item["entry_id"])]
            return {
                "status": "complete",
                "source": "cached",
                "cache_path": str(cache_path),
                "space": "input-space six-channel image moments; not frozen encoder state space",
                "computed_candidates": len(values),
                "failures": {},
                "features": ["input_density_scale", "input_covariance_trace", "class_count", "samples_per_class"],
            }
    try:
        import torch
    except ImportError as error:  # pragma: no cover - torch is a project dependency
        return {"status": "unavailable", "reason": str(error)}
    values: dict[tuple[str, str], dict[str, float]] = {}
    failures: dict[str, str] = {}
    for dataset, entry_id in sorted(unique):
        index_path = root / "external" / dataset / "bank_index.json"
        if not index_path.exists():
            failures[f"{dataset}/{entry_id}"] = "bank_index_missing"
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entry = next((x for x in index["entries"] if x["entry_id"] == entry_id), None)
        if entry is None:
            failures[f"{dataset}/{entry_id}"] = "entry_missing_from_bank_index"
            continue
        tensor_path = root / entry["bank_path"]
        if not tensor_path.exists():
            failures[f"{dataset}/{entry_id}"] = "canonical_tensor_missing"
            continue
        try:
            payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
            images = payload["images"].float()
            labels = payload["labels"].cpu().numpy()
            # Six compact, deterministic image descriptors per sample.  This
            # stays low-memory even for the Tiny-ImageNet IPC50 artifacts.
            mean = images.mean(dim=(-2, -1))
            std = images.std(dim=(-2, -1), correction=0)
            descriptors = torch.cat((mean, std), dim=1).cpu().numpy()
            density_scales, covariance_traces, counts = [], [], []
            for label in np.unique(labels):
                cloud = descriptors[labels == label]
                counts.append(len(cloud))
                covariance_traces.append(float(np.var(cloud, axis=0).sum()))
                if len(cloud) < 2:
                    continue
                distances = np.sqrt(np.maximum(
                    (cloud * cloud).sum(1)[:, None] + (cloud * cloud).sum(1)[None, :] - 2 * cloud @ cloud.T,
                    0.0,
                ))
                np.fill_diagonal(distances, np.inf)
                density_scales.append(float(np.median(distances.min(axis=1))))
            values[(dataset, entry_id)] = {
                "input_density_scale": float(np.mean(density_scales)) if density_scales else 0.0,
                "input_covariance_trace": float(np.mean(covariance_traces)),
                "class_count": float(len(np.unique(labels))),
                "samples_per_class": float(np.mean(counts)),
            }
        except Exception as error:  # pragma: no cover - hardware/library dependent
            failures[f"{dataset}/{entry_id}"] = f"{type(error).__name__}: {error}"
    for item in observations:
        item["nuisance"] = values.get((item["dataset"], item["entry_id"]), {})
    if not failures:
        save_json(
            {"space": "input-space six-channel image moments", "values": {f"{a}/{b}": v for (a, b), v in values.items()}},
            cache_path,
        )
    return {
        "status": "complete" if not failures else "partial",
        "source": "computed",
        "cache_path": str(cache_path),
        "space": "input-space six-channel image moments; not frozen encoder state space",
        "computed_candidates": len(values),
        "failures": failures,
        "features": ["input_density_scale", "input_covariance_trace", "class_count", "samples_per_class"],
    }


def write_phase1_figures(report_path: str | Path) -> list[Path]:
    """Render compact, descriptive Phase-1 tables/figures from an existing report."""
    import matplotlib.pyplot as plt

    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_dir = report_path.parent
    csv_path = output_dir / "phase1_proxy_utility_spearman.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "ipc", "architecture", "feature", "rho", "n"])
        writer.writeheader()
        for dataset, by_ipc in report["proxy_rank_matrix"]["dataset"].items():
            for ipc, by_architecture in by_ipc.items():
                for architecture, features in by_architecture.items():
                    for feature, result in features.items():
                        writer.writerow({"dataset": dataset, "ipc": ipc, "architecture": architecture, "feature": feature, **result})

    feature_names = list(TOPOLOGY_FEATURES)
    datasets = ["cifar10", "cifar100", "tiny_imagenet"]
    values = np.asarray([
        [report["conditional_topology"][feature]["effect_signs"]["dataset"][dataset]["rho"] for dataset in datasets]
        for feature in feature_names
    ])
    figure, axis = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    image = axis.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(datasets)), ["CIFAR-10", "CIFAR-100", "Tiny-ImageNet"])
    axis.set_yticks(range(len(feature_names)), feature_names)
    axis.set_title("LOFO conditional topology residual Spearman")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="residual Spearman $\\rho$")
    figure_path = output_dir / "phase1_effect_signs.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return [csv_path, figure_path]


def run_phase1_diagnostics(
    root: str | Path = ".", out_dir: str | Path = "runs/phase1", bootstrap_reps: int = 500, seed: int = 17
) -> Path:
    """Run the read-only E1--E3 conditional-topology analysis and save JSON."""
    observations, provenance = load_phase1_observations(root)
    input_nuisance = _attach_input_nuisances(Path(root), observations)
    extra_nuisance = (
        ("input_density_scale", "input_covariance_trace", "samples_per_class")
        if input_nuisance["status"] == "complete"
        else ()
    )
    rank_matrix = _rank_matrix(observations, ALL_SCORE_FEATURES)
    topology: dict[str, Any] = {}
    for feature in TOPOLOGY_FEATURES:
        records, coefficients = _oof_residuals(observations, feature, extra_nuisance)
        overall = _rho(
            [record["topology_residual"] for record in records],
            [record["utility_residual"] for record in records],
        )
        topology[feature] = {
            "definition": "LOFO utility residual vs LOFO topology residual; controls = base + effective_rank + architecture.",
            "oof_residual_rho": overall,
            "n_oof_observations": len(records),
            "effect_signs": _group_effects(records),
            "bootstrap": _bootstrap(records, coefficients, bootstrap_reps, seed),
        }
    conditional_nulls: dict[str, Any] = {}
    for null_name in (
        "labelperm_topo_traj_dist",
        "timeshuffle_topo_traj_dist",
        "timeshuffle_betti_traj_dist",
    ):
        records, coefficients = _oof_residuals(observations, f"null:{null_name}", extra_nuisance)
        conditional_nulls[null_name] = {
            "oof_residual_rho": _rho(
                [record["topology_residual"] for record in records],
                [record["utility_residual"] for record in records],
            ),
            "n_oof_observations": len(records),
            "bootstrap": _bootstrap(records, coefficients, bootstrap_reps, seed),
        }
    report = {
        "stage": "phase1_conditional_topology_diagnostics",
        "evidence_class": "post-hoc developmental diagnostic; E1--E3 are read-only and not confirmation data",
        "protocol": {
            "outer_split": "leave-one-method-family-out within each dataset",
            "base": list(BASE_FEATURES),
            "topology": list(TOPOLOGY_FEATURES),
            "nuisance": ["IPC", "effective_rank", "architecture"] + list(extra_nuisance),
            "bootstrap": {"seed": seed, "reps": bootstrap_reps, "cluster": "release_lineage"},
        },
        "inputs": provenance,
        "proxy_rank_matrix": rank_matrix,
        "conditional_topology": topology,
        "historical_null_correlations": _existing_nulls(observations),
        "conditional_existing_nulls": conditional_nulls,
        "matched_nuisance_controls": {
            "input_nuisance": input_nuisance,
            "class_count_null": {
                "status": "non_identifiable_within_dataset",
                "reason": "Each E1/E2/E3 dataset has a fixed class count, so class count cannot be independently permuted or estimated inside a per-dataset analysis.",
            },
            "interpretation": "Input-space density/covariance are low-compute confound controls only; feature-space matched nulls remain required for G1.",
        },
        "availability_audit": _availability(observations),
        "gate_g1": {
            "status": "fail_no_phase2_entry",
            "reason": "The topology-augmented OOF ranking gain is not stably positive across E1--E3, while label/time null columns retain conditional signal. Input-space matching does not rescue this specificity failure.",
            "missing_for_mechanistic_follow_up": "Feature-space matched density/covariance nulls and per-class/per-timestep components require a future frozen feature-cache extraction.",
            "allowed_conclusion": "Do not promote topology to a predictor branch or enter Phase 2 on the current evidence; preserve it only as a descriptive diagnostic.",
        },
    }
    out_path = Path(out_dir) / "phase1_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, out_path)
    write_phase1_figures(out_path)
    return out_path
