"""AuditBench-DD E1 pipeline helpers.

E1 scores official external candidate-bank artifacts with training-free
proxies and (in the formal round) hard-label student utility. This module
contains only the E1 glue:

- bank loading by ``entry_id`` from ``external/cifar10/bank_index.json``;
- the frozen evaluation-entry normalization (per-channel z-score with
  ``data.DATASET_STATS``), applied uniformly to raw_unbounded synthetic
  tensors and to zscore-inverted selection sets;
- real-reference trajectory embeddings (official CIFAR-10 train subset);
- proxy scoring that reuses ``audit._candidate_scores`` verbatim;
- single-cell student training that reuses ``pipeline.evaluate`` unchanged
  (single arch / single seed via a dataclass-replaced config copy).

Nothing here modifies the existing ``pipeline.evaluate`` behavior or its
artifact schema; the E1 entry point writes a normalized copy of the bank
artifact whose ``meta.config.dataset`` contract matches ``evaluate``.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audit import (
    LOWER_IS_BETTER,
    _candidate_scores,
    _ridge_proxy,
    _short_run_utility,
    extract_synthetic_embeddings,
)
from .config import ExperimentConfig
from .data import DATASET_STATS, build_datasets
from .pipeline import evaluate
from .utils import resolve_device, save_json, seed_everything

# Scores produced by audit._candidate_scores for every candidate.
CANDIDATE_SCORE_KEYS = (
    "fid_endpoint",
    "knn_precision",
    "knn_recall",
    "knn_coverage",
    "effective_rank",
    "topo_traj_dist",
    "topo_endpoint_dist",
    "betti_traj_dist",
    "class_moment_dist",
    "nulls",
)
NULL_KEYS = (
    "labelperm_class_moment_dist",
    "labelperm_topo_traj_dist",
    "timeshuffle_topo_traj_dist",
    "timeshuffle_betti_traj_dist",
    "cardinality_half_fid_endpoint",
    "cardinality_half_effective_rank",
)
# Full E1 record schema: candidate scores + probe/short-run scores + runtime.
E1_SCORE_RECORD_KEYS = CANDIDATE_SCORE_KEYS[:-1] + (
    "ridge_proxy",
    "short_run_5ep",
    "runtime_sec",
    "nulls",
)
EVALUATION_SCHEMA = {"runs": ("architecture", "seed", "top1"), "summary": ("architecture", "mean", "std", "n")}

SMOKE_ENTRY_IDS = ("vico.dc.ipc10", "dcbench.random.ipc10")
SMOKE_ARCH = "convnet"
SMOKE_SEED = 0
REFERENCE_N_IMAGES = 10_000
PROXY_RESAMPLES = 8


def load_bank_index(bank_root: str | Path) -> dict[str, Any]:
    """Load the E0 bank index."""
    index_path = Path(bank_root) / "bank_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"bank index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def bank_entry_meta(bank_root: str | Path, entry_id: str) -> dict[str, Any]:
    """Return the bank-index record for ``entry_id`` (raises if unknown)."""
    index = load_bank_index(bank_root)
    for entry in index["entries"]:
        if entry["entry_id"] == entry_id:
            return entry
    raise KeyError(f"entry_id not in bank index: {entry_id}")


def load_bank_artifact(bank_root: str | Path, entry_id: str) -> dict[str, Any]:
    """Load one canonical bank artifact (images/labels/meta) by entry_id.

    The artifact is returned exactly as stored: raw_unbounded synthetics stay
    unbounded, zscore-inverted selection sets stay in [0,1]. Normalization to
    the evaluation input distribution happens only in ``zscore_entry``.
    """
    record = bank_entry_meta(bank_root, entry_id)
    candidates = [Path(record["bank_path"].replace("\\", "/"))]
    if not candidates[0].is_absolute():
        candidates.append(Path(bank_root).parent.parent / candidates[0])
    candidates.append(Path(bank_root) / "bank" / f"{entry_id}.pt")
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"bank artifact missing for {entry_id}: {record['bank_path']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("images", "labels", "meta"):
        if key not in payload:
            raise ValueError(f"bank artifact {entry_id} lacks key {key!r}")
    return payload


def zscore_entry(images: torch.Tensor, dataset: str = "cifar10") -> torch.Tensor:
    """Frozen E1 evaluation-entry normalization: per-channel z-score.

    Applied uniformly to every bank artifact before student training or
    embedding extraction, so raw_unbounded synthetics and [0,1] selection
    sets enter the pipeline through the same transform.
    """
    mean, std = DATASET_STATS[dataset]
    mean_t = images.new_tensor(mean)[None, :, None, None]
    std_t = images.new_tensor(std)[None, :, None, None]
    return (images.float() - mean_t) / std_t


def eval_ready_payload(
    images_z: torch.Tensor,
    labels: torch.Tensor,
    meta: dict[str, Any],
    dataset_name: str,
    image_size: int,
) -> dict[str, Any]:
    """Wrap normalized tensors in a ``pipeline.evaluate``-compatible payload.

    ``evaluate`` requires ``meta.config.dataset.{name,image_size}`` to match
    the evaluation config; the bank meta is preserved and augmented, never
    stripped of provenance.
    """
    out_meta = copy.deepcopy(meta)
    config = dict(out_meta.get("config", {}))
    config["dataset"] = {"name": dataset_name, "image_size": image_size}
    out_meta["config"] = config
    return {"images": images_z.float(), "labels": labels.long(), "meta": out_meta}


def write_eval_artifact(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


@torch.no_grad()
def real_reference_embeddings(
    cfg: ExperimentConfig,
    device: torch.device,
    n_images: int = REFERENCE_N_IMAGES,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted-clean trajectory embeddings of the official train subset.

    Uses the same Phase-1 extraction as candidates
    (``audit.extract_synthetic_embeddings``); the reference set is the first
    ``n_images`` of a seed-fixed permutation of the z-scored clean train set.
    """
    train_set, _ = build_datasets(cfg.dataset, train_augment=False)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_set), generator=generator)[:n_images].tolist()
    images = torch.stack([train_set[i][0] for i in order])
    labels = torch.tensor([int(train_set[i][1]) for i in order])
    states = extract_synthetic_embeddings(cfg, images, labels, device).float()
    return states.numpy(), labels.numpy()


def score_candidate(
    cfg: ExperimentConfig,
    images_z: torch.Tensor,
    labels: torch.Tensor,
    real_states: np.ndarray,
    real_labels: np.ndarray,
    device: torch.device,
    resamples: int = PROXY_RESAMPLES,
    seed: int = 0,
    short_run_epochs: int = 5,
) -> dict[str, Any]:
    """Compute the frozen E1 proxy suite for one candidate.

    Blind by construction: receives only normalized tensors + labels (+ the
    real reference embeddings); no method names or provenance enter here.
    """
    started = time.time()
    states = extract_synthetic_embeddings(cfg, images_z, labels, device).float().numpy()
    labels_np = labels.numpy()
    scores = _candidate_scores(cfg, states, labels_np, real_states, real_labels, resamples, seed)
    scores["ridge_proxy"] = _ridge_proxy(
        states[:, -1], labels_np, real_states[:, -1], real_labels
    )
    scores["short_run_5ep"] = _short_run_utility(
        cfg, images_z, labels, device, epochs=short_run_epochs, seed=seed
    )
    scores["runtime_sec"] = round(time.time() - started, 1)
    validate_score_record(scores)
    return scores


def validate_score_record(scores: dict[str, Any]) -> None:
    """Schema check: keys match the audit._candidate_scores-derived suite."""
    missing = [k for k in E1_SCORE_RECORD_KEYS if k not in scores]
    if missing:
        raise ValueError(f"score record missing keys: {missing}")
    nulls = scores.get("nulls")
    if not isinstance(nulls, dict):
        raise TypeError("score record 'nulls' must be a dict")
    missing_nulls = [k for k in NULL_KEYS if k not in nulls]
    if missing_nulls:
        raise ValueError(f"score record nulls missing keys: {missing_nulls}")
    for name in E1_SCORE_RECORD_KEYS:
        if name in ("nulls", "runtime_sec"):
            continue
        value = scores[name]
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"score {name!r} must be numeric or None, got {type(value)}")
    unknown_lower = set(scores) & set(LOWER_IS_BETTER) - set(E1_SCORE_RECORD_KEYS)
    if unknown_lower:
        raise ValueError(f"unexpected lower-is-better scores: {sorted(unknown_lower)}")


def validate_evaluation_schema(path: str | Path) -> dict[str, Any]:
    """Schema check: file produced by pipeline.evaluate keeps the frozen shape."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in EVALUATION_SCHEMA:
        if key not in payload:
            raise ValueError(f"evaluation artifact missing key {key!r}")
    for run in payload["runs"]:
        for field in EVALUATION_SCHEMA["runs"]:
            if field not in run:
                raise ValueError(f"evaluation run missing field {field!r}")
    for row in payload["summary"]:
        for field in EVALUATION_SCHEMA["summary"]:
            if field not in row:
                raise ValueError(f"evaluation summary missing field {field!r}")
    return payload


def single_cell_evaluate(
    cfg: ExperimentConfig,
    synthetic_path: str | Path,
    arch: str,
    seed: int,
) -> Path:
    """One (arch, seed) student training via the unchanged pipeline.evaluate."""
    eval_cfg = replace(cfg.evaluation, architectures=[arch], seeds=[seed])
    cfg_copy = replace(cfg, evaluation=eval_cfg)
    return evaluate(cfg_copy, synthetic_path)


def run_smoke(
    cfg: ExperimentConfig,
    bank_root: str | Path,
    out_dir: str | Path,
    entry_ids: tuple[str, ...] = SMOKE_ENTRY_IDS,
    arch: str = SMOKE_ARCH,
    seed: int = SMOKE_SEED,
    reference_cache: str | Path | None = None,
    stage: str = "auditbench_e1_smoke",
) -> Path:
    """E1 end-to-end smoke: embeddings -> proxies -> one student training cell.

    Smoke outputs validate pipeline runnability and artifact schemas only;
    they must not feed any proxy-utility inference.
    """
    seed_everything(seed)
    device = resolve_device(cfg.device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(reference_cache) if reference_cache else out_dir / "real_reference_emb.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        real_states, real_labels = cached["states"], cached["labels"]
    else:
        real_states, real_labels = real_reference_embeddings(cfg, device)
        np.savez_compressed(cache_path, states=real_states, labels=real_labels)

    records: dict[str, Any] = {}
    for entry_id in entry_ids:
        artifact = load_bank_artifact(bank_root, entry_id)
        images_z = zscore_entry(artifact["images"], cfg.dataset.name)
        scores = score_candidate(
            cfg,
            images_z,
            artifact["labels"],
            real_states,
            real_labels,
            device,
            seed=seed,
        )
        scores_path = out_dir / f"{entry_id}.scores.json"
        save_json(
            {
                "entry_id": entry_id,
                "ipc": int(artifact["meta"]["ipc"]),
                "smoke_only": True,
                "scores": scores,
            },
            scores_path,
        )
        payload = eval_ready_payload(
            images_z, artifact["labels"], artifact["meta"], cfg.dataset.name, cfg.dataset.image_size
        )
        eval_artifact = write_eval_artifact(payload, out_dir / f"{entry_id}.eval_input.pt")
        evaluation_path = single_cell_evaluate(cfg, eval_artifact, arch, seed)
        evaluation = validate_evaluation_schema(evaluation_path)
        records[entry_id] = {
            "scores_path": str(scores_path),
            "scores_schema_ok": True,
            "eval_artifact": str(eval_artifact),
            "evaluation_path": str(evaluation_path),
            "evaluation_schema_ok": True,
            "smoke_top1_recorded_schema_only": evaluation["runs"][0]["top1"],
            "note": "smoke output; NOT usable for proxy-utility inference",
        }
    manifest = {
        "stage": stage,
        "entry_ids": list(entry_ids),
        "arch": arch,
        "seed": seed,
        "reference_cache": str(cache_path),
        "reference_n_images": int(real_states.shape[0]),
        "records": records,
    }
    manifest_path = out_dir / "smoke_manifest.json"
    save_json(manifest, manifest_path)
    return manifest_path
