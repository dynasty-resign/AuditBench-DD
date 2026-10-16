from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import TensorDataset, build_datasets, dataset_spec
from .diffusion import DiffusionSchedule
from .models import classifier
from .pilot_a import _pairwise, _prim_deaths
from .pipeline import _accuracy, build_backbone
from .topology_v2 import betti0_curve, fit_epsilon_grid
from .trajectory import latent_reconstruction_trajectories
from .utils import resolve_device, save_json, seed_everything

# Pre-registered score directions: these must be NEGATED so higher = better.
LOWER_IS_BETTER = {
    "objective_loss",
    "class_moment_dist",
    "fid_endpoint",
    "topo_endpoint_dist",
    "topo_traj_dist",
    "betti_traj_dist",
}
NON_TOPO_SCORES = (
    "objective_loss",
    "ridge_proxy",
    "class_moment_dist",
    "fid_endpoint",
    "knn_precision",
    "knn_recall",
    "knn_coverage",
    "effective_rank",
    "short_run_5ep",
)
TOPO_SCORES = ("topo_endpoint_dist", "topo_traj_dist", "betti_traj_dist")


def parse_run_name(name: str) -> tuple[str, int, int]:
    """Parse ``{method}_ipc{N}`` or ``{method}_ipc{N}_s{seed}`` run directory names."""
    match = re.fullmatch(r"(.+?)_ipc(\d+)(?:_s(\d+))?", name)
    if not match:
        raise ValueError(f"cannot parse run name: {name}")
    method, ipc, seed = match.group(1), int(match.group(2)), match.group(3)
    return method, ipc, int(seed) if seed is not None else 0


def _iter_candidates(runs_dir: Path) -> list[tuple[Path, Path]]:
    out = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        synthetics = sorted(run_dir.glob("synthetic_*.pt"))
        if synthetics:
            out.append((run_dir, synthetics[0]))
    return out


@torch.no_grad()
def extract_synthetic_embeddings(
    cfg: ExperimentConfig, images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Predicted-clean trajectory embeddings for arbitrary images, Phase 1 protocol."""
    model = build_backbone(cfg, device).eval()
    schedule = DiffusionSchedule(
        cfg.trajectory.diffusion_steps,
        cfg.trajectory.beta_start,
        cfg.trajectory.beta_end,
        device,
    )
    chunks = []
    batch_size = cfg.backbone.batch_size
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size].to(device)
        batch_labels = labels[start : start + batch_size].to(device)
        denoiser_labels = (
            model.unconditional_labels(len(batch_labels), device)
            if cfg.trajectory.conditioning == "unconditional"
            else batch_labels
        )
        _, predicted_clean, _ = latent_reconstruction_trajectories(
            model,
            batch_images,
            denoiser_labels,
            cfg.trajectory.timesteps,
            schedule,
            cfg.trajectory.inference_steps,
            source=cfg.trajectory.source,
        )
        batch, times = predicted_clean.shape[:2]
        embeddings = model.embed(
            predicted_clean.reshape(batch * times, *predicted_clean.shape[2:])
        ).reshape(batch, times, -1)
        chunks.append(embeddings.cpu())
    return torch.cat(chunks)


def _fid(real: np.ndarray, syn: np.ndarray) -> float:
    from scipy import linalg

    mu_r, mu_s = real.mean(0), syn.mean(0)
    cov_r = np.cov(real, rowvar=False)
    cov_s = np.cov(syn, rowvar=False) if len(syn) > 1 else np.zeros_like(cov_r)
    eps = 1e-6 * np.trace(cov_r) / max(cov_r.shape[0], 1)
    cov_r = cov_r + eps * np.eye(cov_r.shape[0])
    cov_s = cov_s + eps * np.eye(cov_s.shape[0])
    try:
        covmean, _ = linalg.sqrtm(cov_r @ cov_s, disp=False)
    except TypeError:
        covmean = linalg.sqrtm(cov_r @ cov_s)
    covmean = np.real_if_close(covmean, tol=1000).real
    return float((mu_r - mu_s) @ (mu_r - mu_s) + np.trace(cov_r + cov_s - 2.0 * covmean))


def _class_moment_distance(
    syn: np.ndarray, syn_y: np.ndarray, real: np.ndarray, real_y: np.ndarray
) -> float:
    dims = syn.shape[1]
    total = 0.0
    classes = np.unique(real_y)
    for cls in classes:
        s = syn[syn_y == cls]
        r = real[real_y == cls]
        if len(s) == 0 or len(r) == 0:
            continue
        mean_term = float(((s.mean(0) - r.mean(0)) ** 2).sum())
        cov_r = np.cov(r, rowvar=False)
        cov_s = np.cov(s, rowvar=False) if len(s) > 1 else np.zeros((dims, dims))
        cov_term = float(((cov_s - cov_r) ** 2).sum() ** 0.5) / dims
        total += mean_term + cov_term
    return total / max(len(classes), 1)


def _manifold_pr(real: np.ndarray, syn: np.ndarray, k: int = 3) -> tuple[float, float, float]:
    """Improved precision/recall (Kynkaanniemi-style) plus real-manifold coverage."""
    k_real = max(1, min(k, len(real) - 1))
    radii_real = NearestNeighbors(n_neighbors=k_real + 1).fit(real).kneighbors(real)[0][:, -1]
    if len(syn) > 1:
        k_syn = max(1, min(k, len(syn) - 1))
        radii_syn = NearestNeighbors(n_neighbors=k_syn + 1).fit(syn).kneighbors(syn)[0][:, -1]
    else:
        radii_syn = np.zeros(len(syn))
    precision_hits = []
    recall_hits = []
    coverage_hits = []
    for start in range(0, len(syn), 256):
        chunk = syn[start : start + 256]
        aa = (chunk**2).sum(axis=1)[:, None]
        bb = (real**2).sum(axis=1)[None, :]
        dist = np.sqrt(np.clip(aa + bb - 2.0 * chunk @ real.T, 0.0, None))
        precision_hits.append((dist <= radii_real[None, :]).any(axis=1))
        recall_hits.append((dist <= radii_syn[start : start + 256, None]).any(axis=0))
        coverage_hits.append((dist <= radii_real[None, :]).any(axis=0))
    precision = float(np.concatenate(precision_hits).mean())
    recall = float(np.stack(recall_hits).any(axis=0).mean())
    coverage = float(np.stack(coverage_hits).any(axis=0).mean())
    return precision, recall, coverage


def _effective_rank(embeddings: np.ndarray) -> float:
    centered = embeddings - embeddings.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = singular.sum()
    if total <= 0:
        return 0.0
    p = singular / total
    return float(np.exp(-(p * np.log(np.maximum(p, 1e-12))).sum()))


def _per_class_topology_distance(
    syn_states: np.ndarray,
    syn_y: np.ndarray,
    real_states: np.ndarray,
    real_y: np.ndarray,
    metric: str,
    resamples: int,
    seed: int,
    time_indices: list[int] | None = None,
    betti_bins: int | None = None,
) -> float | None:
    """Equal-cardinality H0 distance between synthetic and real class clouds.

    Returns None when every class has fewer than two synthetic samples (IPC=1).
    With ``betti_bins`` the per-pair deaths are compared as Betti-0 curves on a
    grid fitted to the real subsample; otherwise as sorted-death L2 (exact 1-D
    Wasserstein at equal cardinality).
    """
    rng = np.random.default_rng(seed)
    times = time_indices if time_indices is not None else list(range(syn_states.shape[1]))
    distances: list[float] = []
    for cls in np.unique(real_y):
        syn_c = syn_states[syn_y == cls]
        pool_c = real_states[real_y == cls]
        if len(syn_c) < 2 or len(pool_c) < 2:
            continue
        # Equal-cardinality comparison at k = min(pool, syn). For pool >= syn
        # (all E1/E2 cohorts) k == len(syn_c) and the rng stream is unchanged,
        # so frozen definitions are reproduced bit-exactly; pool < syn (first
        # possible at 200-class E3 reference subsampling) extends the rule to
        # the largest equal cardinality available.
        k = min(len(pool_c), len(syn_c))
        for _ in range(resamples):
            sub = pool_c[rng.choice(len(pool_c), size=k, replace=False)]
            syn_sub = syn_c if k == len(syn_c) else syn_c[rng.choice(len(syn_c), size=k, replace=False)]
            for time_index in times:
                deaths_syn = _prim_deaths(_pairwise(syn_sub[:, time_index], metric))
                deaths_real = _prim_deaths(_pairwise(sub[:, time_index], metric))
                if betti_bins is not None:
                    grid = fit_epsilon_grid(deaths_real[None, :], betti_bins)
                    curve_syn = betti0_curve(deaths_syn[None, :], grid)[0]
                    curve_real = betti0_curve(deaths_real[None, :], grid)[0]
                    distances.append(float(np.sqrt(((curve_syn - curve_real) ** 2).mean())))
                else:
                    distances.append(float(np.sqrt(((deaths_syn - deaths_real) ** 2).mean())))
    if not distances:
        return None
    return float(np.mean(distances))


def _short_run_utility(
    cfg: ExperimentConfig,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = 5,
    seed: int = 0,
) -> float:
    """Cheap utility proxy: short ConvNet training on the candidate set."""
    seed_everything(seed)
    classes, image_size = dataset_spec(cfg.dataset.name)
    _, test_set = build_datasets(cfg.dataset)
    model = classifier("convnet", classes, image_size).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.evaluation.lr,
        momentum=0.9,
        weight_decay=cfg.evaluation.weight_decay,
    )
    synthetic = TensorDataset(images, labels, augment=True)
    loader = DataLoader(
        synthetic,
        batch_size=min(cfg.evaluation.batch_size, len(synthetic)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=cfg.dataset.num_workers,
    )
    for _ in range(epochs):
        model.train()
        for batch_images, batch_labels in loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            loss = F.cross_entropy(model(batch_images), batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    return _accuracy(model, test_loader, device)


def _ridge_proxy(
    syn: np.ndarray, syn_y: np.ndarray, real: np.ndarray, real_y: np.ndarray
) -> float:
    probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0, solver="svd"))
    probe.fit(syn, syn_y)
    return float(probe.score(real, real_y))


def _candidate_scores(
    cfg: ExperimentConfig,
    states: np.ndarray,
    labels: np.ndarray,
    real_states: np.ndarray,
    real_labels: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    metric = "cosine" if cfg.topology.metric == "cosine" else "euclidean"
    times = states.shape[1]
    endpoint_syn = states[:, -1]
    endpoint_real = real_states[:, -1]
    scores: dict[str, Any] = {}
    precision, recall, coverage = _manifold_pr(endpoint_real, endpoint_syn)
    scores["fid_endpoint"] = _fid(endpoint_real, endpoint_syn)
    scores["knn_precision"] = precision
    scores["knn_recall"] = recall
    scores["knn_coverage"] = coverage
    scores["effective_rank"] = _effective_rank(endpoint_syn)
    scores["topo_traj_dist"] = _per_class_topology_distance(
        states, labels, real_states, real_labels, metric, resamples, seed
    )
    scores["topo_endpoint_dist"] = _per_class_topology_distance(
        states, labels, real_states, real_labels, metric, resamples, seed,
        time_indices=[times - 1],
    )
    scores["betti_traj_dist"] = _per_class_topology_distance(
        states, labels, real_states, real_labels, metric, resamples, seed,
        betti_bins=cfg.feasibility.topology_curve_bins,
    )
    scores["class_moment_dist"] = _class_moment_distance(
        endpoint_syn, labels, endpoint_real, real_labels
    )

    nulls: dict[str, Any] = {}
    rng = np.random.default_rng(seed + 1)
    permuted = rng.permutation(labels)
    nulls["labelperm_class_moment_dist"] = _class_moment_distance(
        endpoint_syn, permuted, endpoint_real, real_labels
    )
    nulls["labelperm_topo_traj_dist"] = _per_class_topology_distance(
        states, permuted, real_states, real_labels, metric, resamples, seed
    )
    shuffled = np.stack([row[rng.permutation(times)] for row in states])
    nulls["timeshuffle_topo_traj_dist"] = _per_class_topology_distance(
        shuffled, labels, real_states, real_labels, metric, resamples, seed
    )
    nulls["timeshuffle_betti_traj_dist"] = _per_class_topology_distance(
        shuffled, labels, real_states, real_labels, metric, resamples, seed,
        betti_bins=cfg.feasibility.topology_curve_bins,
    )
    keep = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(labels == cls),
                size=max(1, int((labels == cls).sum() // 2)),
                replace=False,
            )
            for cls in np.unique(labels)
        ]
    )
    nulls["cardinality_half_fid_endpoint"] = _fid(endpoint_real, endpoint_syn[keep])
    nulls["cardinality_half_effective_rank"] = _effective_rank(endpoint_syn[keep])
    scores["nulls"] = nulls
    return scores


def run_audit_extract(
    cfg: ExperimentConfig,
    runs_dir: str | Path,
    cache_dir: str | Path,
    artifact: str | Path,
) -> Path:
    """Pilot B step 1: pre-registered scalar scores for every candidate set."""
    seed_everything(cfg.seed)
    runs_dir = Path(runs_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    times = len(cfg.trajectory.timesteps)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    flat = payload["representations"]["feature_trajectory"].float()
    real_states = flat.reshape(len(flat), times, -1).numpy()
    real_labels = payload["labels"].numpy()
    resamples = 8

    manifest = []
    for run_dir, synthetic_path in _iter_candidates(runs_dir):
        method, ipc, synthesis_seed = parse_run_name(run_dir.name)
        out_path = cache_dir / f"{run_dir.name}.scores.json"
        if out_path.exists():
            manifest.append(str(out_path))
            continue
        started = time.time()
        payload_syn = torch.load(synthetic_path, map_location="cpu", weights_only=False)
        images, labels = payload_syn["images"], payload_syn["labels"]
        emb_path = cache_dir / f"{run_dir.name}.emb.pt"
        if emb_path.exists():
            states_t = torch.load(emb_path, map_location="cpu", weights_only=False)
        else:
            states_t = extract_synthetic_embeddings(cfg, images, labels, device).float()
            torch.save(states_t, emb_path)
        states = states_t.numpy()
        scores = _candidate_scores(
            cfg, states, labels.numpy(), real_states, real_labels, resamples, cfg.seed
        )
        history_path = synthetic_path.with_suffix(".history.json")
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                scores["objective_loss"] = float(history[-1]["loss"])
        scores["ridge_proxy"] = _ridge_proxy(
            states[:, -1], labels.numpy(), real_states[:, -1], real_labels
        )
        scores["short_run_5ep"] = _short_run_utility(cfg, images, labels, device)
        scores["runtime_sec"] = round(time.time() - started, 1)
        record = {
            "run": run_dir.name,
            "method": method,
            "ipc": ipc,
            "synthesis_seed": synthesis_seed,
            "n_synthetic": len(labels),
            "scores": scores,
        }
        save_json(record, out_path)
        manifest.append(str(out_path))
    return cache_dir


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    result = spearmanr(a, b)
    rho = getattr(result, "statistic", None)
    if rho is None:
        rho = result[0]
    return float(rho)


def _oriented(values: dict[str, float], name: str) -> float:
    value = values[name]
    return -value if name in LOWER_IS_BETTER else value


def load_rows(runs_dir: str | Path, cache_dir: str | Path) -> list[dict[str, Any]]:
    """Join cached scores with evaluation utilities for every complete candidate."""
    runs_dir = Path(runs_dir)
    cache_dir = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    for run_dir, synthetic_path in _iter_candidates(runs_dir):
        cache_path = cache_dir / f"{run_dir.name}.scores.json"
        eval_path = synthetic_path.with_suffix(".evaluation.json")
        if not cache_path.exists() or not eval_path.exists():
            continue
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        utility = {entry["architecture"]: entry["mean"] for entry in evaluation["summary"]}
        rows.append(
            {
                "run": record["run"],
                "method": record["method"],
                "ipc": record["ipc"],
                "synthesis_seed": record.get("synthesis_seed", 0),
                "utility": utility,
                "scores": {k: v for k, v in record["scores"].items() if k != "nulls"},
                "nulls": record["scores"].get("nulls", {}),
            }
        )
    return rows


def run_audit_analyze(
    cfg: ExperimentConfig,
    runs_dir: str | Path,
    cache_dir: str | Path,
    architectures: list[str] | None = None,
) -> Path:
    """Pilot B step 2: predictive-validity analysis over the candidate bank."""
    cache_dir = Path(cache_dir)
    rows = load_rows(runs_dir, cache_dir)
    if not rows:
        raise RuntimeError(f"no candidates with both scores and evaluation in {runs_dir}")
    if architectures is None:
        architectures = sorted({a for row in rows for a in row["utility"]})
    score_names = [
        name
        for name in list(NON_TOPO_SCORES) + list(TOPO_SCORES)
        if any(row["scores"].get(name) is not None for row in rows)
    ]
    ipcs = sorted({row["ipc"] for row in rows})

    spearman_table: dict[str, Any] = {}
    for arch in architectures:
        per_ipc: dict[str, Any] = {}
        for ipc in ipcs:
            group = [r for r in rows if r["ipc"] == ipc and arch in r["utility"]]
            utility = np.array([r["utility"][arch] for r in group])
            for name in score_names:
                values = [
                    _oriented(r["scores"], name) if r["scores"].get(name) is not None else None
                    for r in group
                ]
                mask = np.array([v is not None for v in values])
                if mask.sum() < 3:
                    continue
                rho = _spearman(
                    np.array([v for v, m in zip(values, mask) if m], dtype=float),
                    utility[mask],
                )
                if rho is not None:
                    per_ipc.setdefault(str(ipc), {})[name] = rho
        spearman_table[arch] = per_ipc

    report: dict[str, Any] = {
        "protocol": {
            "pilot": "B",
            "candidates": len(rows),
            "architectures": architectures,
            "score_names": score_names,
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "note": (
                "Utility = mean hard-label accuracy over eval seeds. Spearman computed "
                "within IPC groups with >=3 scored candidates."
            ),
        },
        "spearman_within_ipc": spearman_table,
        "lomo": _lomo_analysis(rows, architectures, ipcs, score_names),
        "top1_regret": _regret_analysis(rows, architectures, ipcs, score_names),
        "null_checks": _null_analysis(rows),
        "cost": _cost_analysis(rows),
    }
    out = cache_dir / "pilot_b_report.json"
    save_json(report, out)
    return out


def _lomo_analysis(
    rows: list[dict[str, Any]], architectures: list[str], ipcs: list[int], score_names: list[str]
) -> dict[str, Any]:
    """Leave-one-method-out ridge ensembles: non-topology vs +topology."""
    methods = sorted({row["method"] for row in rows})
    non_topo = [n for n in NON_TOPO_SCORES if n in score_names]
    result: dict[str, Any] = {}
    for arch in architectures:
        usable = [r for r in rows if arch in r["utility"]]

        def design(names: list[str], usable=usable, arch=arch):
            kept, features, targets = [], [], []
            for row in usable:
                if all(row["scores"].get(n) is not None for n in names):
                    kept.append(row)
                    features.append([_oriented(row["scores"], n) for n in names])
                    targets.append(row["utility"][arch])
            return kept, np.array(features, dtype=float), np.array(targets, dtype=float)

        def out_of_fold(names: list[str]) -> dict[str, float]:
            kept, features, targets = design(names)
            if len(kept) < 4:
                return {}
            predictions = np.zeros(len(kept))
            for held_out in methods:
                train_idx = [i for i, r in enumerate(kept) if r["method"] != held_out]
                test_idx = [i for i, r in enumerate(kept) if r["method"] == held_out]
                if not train_idx or not test_idx:
                    continue
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(features[train_idx], targets[train_idx])
                predictions[test_idx] = model.predict(features[test_idx])
            rhos = {}
            kept_ipcs = np.array([r["ipc"] for r in kept])
            for ipc in ipcs:
                mask = kept_ipcs == ipc
                if mask.sum() >= 3:
                    rho = _spearman(predictions[mask], targets[mask])
                    if rho is not None:
                        rhos[str(ipc)] = rho
            return rhos

        base = out_of_fold(non_topo)
        full = out_of_fold(non_topo + [n for n in TOPO_SCORES if n in score_names])
        delta = {ipc: full[ipc] - base[ipc] for ipc in base if ipc in full}
        result[arch] = {"non_topo": base, "plus_topo": full, "delta_rho": delta}
    return result


def _regret_analysis(
    rows: list[dict[str, Any]], architectures: list[str], ipcs: list[int], score_names: list[str]
) -> dict[str, Any]:
    """Top-1 selection regret per score; lower is better."""
    result: dict[str, Any] = {}
    for arch in architectures:
        per_ipc: dict[str, Any] = {}
        for ipc in ipcs:
            group = [r for r in rows if r["ipc"] == ipc and arch in r["utility"]]
            if len(group) < 3:
                continue
            utility = np.array([r["utility"][arch] for r in group])
            best = float(utility.max())
            entry: dict[str, Any] = {"random_expected_regret": best - float(utility.mean())}
            for name in score_names:
                values = [
                    _oriented(r["scores"], name) if r["scores"].get(name) is not None else None
                    for r in group
                ]
                if any(v is None for v in values) or np.std(values) == 0:
                    continue
                pick = int(np.argmax(values))
                entry[name] = best - float(utility[pick])
            per_ipc[str(ipc)] = entry
        result[arch] = per_ipc
    return result


def _null_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    pairs = [
        ("labelperm_topo_traj_dist", "topo_traj_dist"),
        ("timeshuffle_topo_traj_dist", "topo_traj_dist"),
        ("timeshuffle_betti_traj_dist", "betti_traj_dist"),
        ("labelperm_class_moment_dist", "class_moment_dist"),
        ("cardinality_half_fid_endpoint", "fid_endpoint"),
    ]
    for null_name, base_name in pairs:
        deltas = []
        for row in rows:
            null_value = row["nulls"].get(null_name)
            base_value = row["scores"].get(base_name)
            if null_value is not None and base_value is not None:
                deltas.append(float(null_value - base_value))
        if deltas:
            checks[null_name] = {
                "mean_delta": float(np.mean(deltas)),
                "fraction_degraded": float(np.mean([d > 0 for d in deltas])),
                "n": len(deltas),
                "expectation": "positive delta (distance worsens) for *_dist scores",
            }
    return checks


def _cost_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [
        row["scores"].get("runtime_sec")
        for row in rows
        if row["scores"].get("runtime_sec") is not None
    ]
    return {
        "score_runtime_sec_per_candidate": {
            "mean": float(np.mean(runtimes)) if runtimes else None,
            "n": len(runtimes),
        },
        "note": "Compare against full cross-arch evaluation cost recorded in PHASE1.md.",
    }
