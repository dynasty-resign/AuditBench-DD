"""iter-0009 redesign round: gradient-alignment and literature-driven proxies.

Training-dynamics-aware proxy metrics for distilled-set utility screening,
implemented as an add-only extension. Nothing in ``audit.py`` or any frozen
artifact is modified by this module.

Metric families (frozen in
``runs/evidence_packs/iter0009_redesign/proxy_definition_freeze.json``):

* gradient alignment (per-layer cosine similarity of synthetic- vs
  real-data gradients at K random ConvNet initializations; DC/MTT-style
  training-dynamics signal),
* multi-step gradient alignment (alignment measured after 1/5/10 SGD steps
  on the synthetic set),
* NAS-style zero-cost metrics (grad norm, grad variance, activation
  diversity, SynFlow-style score),
* spectral rank on frozen-encoder embeddings (ECCV 2024 inspiration),
* relational structure on frozen-encoder embeddings (CVPR 2024
  inspiration),
* EL2N-style importance-weighted gradient alignment (Neural Networks 2024
  inspiration).

All outputs are screening-level evidence; no inferential statistics here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .audit import _class_moment_distance, _manifold_pr, _ridge_proxy
from .data import DATASET_STATS, dataset_spec
from .models import classifier
from .pipeline import build_backbone
from .utils import seed_everything

GRAD_LAYER_TYPES = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.GroupNorm)
ACTIVATION_TYPES = (nn.ReLU, nn.SiLU)

GRAD_ALIGN_BATCH = 256
GRAD_ALIGN_K = 5
MULTI_STEP_STEPS = (1, 5, 10)
MULTI_STEP_K = 3
MULTI_STEP_LR = 0.01
ZERO_COST_BATCH = 256
IMPORTANCE_K = 5
IMPORTANCE_TEMPERATURE = 1.0
SPECTRAL_REAL_SUBSAMPLE = 5000
RELATIONAL_COMMON_N = 2000
RELATIONAL_REAL_CAP = 5000
EMBED_BATCH = 512


def _deterministic_subset(
    images: torch.Tensor, labels: torch.Tensor, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """First ``min(n, N)`` elements of a seed-fixed permutation."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(images), generator=generator)[: min(n, len(images))]
    return images[order], labels[order]


def _layer_named_parameters(model: nn.Module) -> list[tuple[str, list[nn.Parameter]]]:
    """Direct parameters of each gradient-tracked layer module."""
    layers: list[tuple[str, list[nn.Parameter]]] = []
    for name, module in model.named_modules():
        if not isinstance(module, GRAD_LAYER_TYPES):
            continue
        params = [p for _, p in module.named_parameters(recurse=False)]
        if params:
            layers.append((name, params))
    return layers


def _collect_layer_grads(
    layers: list[tuple[str, list[nn.Parameter]]]
) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, params in layers:
        pieces = [p.grad.detach().float().flatten() for p in params if p.grad is not None]
        if pieces:
            grads[name] = torch.cat(pieces)
    return grads


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm() + 1e-8
    return float((a @ b) / denom)


def _alignment_on_model(
    model: nn.Module,
    syn_batch: torch.Tensor,
    syn_labels: torch.Tensor,
    real_batch: torch.Tensor,
    real_labels: torch.Tensor,
    device: torch.device,
    syn_weights: torch.Tensor | None = None,
    real_weights: torch.Tensor | None = None,
) -> tuple[dict[str, float], float, float]:
    """Per-layer cosine between syn-driven and real-driven CE gradients.

    Returns (per_layer_cosine, layer_mean, parameter-count-weighted mean).
    When per-sample weights are given, the weighted CE gradient
    ``sum_i w_i * d loss_i`` is computed in a single backward pass.
    """
    layers = _layer_named_parameters(model)

    def backward_grads(
        images: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        logits = model(images.to(device))
        losses = F.cross_entropy(logits, labels.to(device), reduction="none")
        loss = losses.mean() if weights is None else (weights.to(device) * losses).sum()
        loss.backward()
        return _collect_layer_grads(layers)

    with torch.enable_grad():
        syn_grads = backward_grads(syn_batch, syn_labels, syn_weights)
        real_grads = backward_grads(real_batch, real_labels, real_weights)
    model.zero_grad(set_to_none=True)

    per_layer: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0
    for name in syn_grads:
        if name not in real_grads:
            continue
        cosine = _cosine(syn_grads[name], real_grads[name])
        per_layer[name] = cosine
        n_params = syn_grads[name].numel()
        weighted_sum += cosine * n_params
        weight_total += n_params
    layer_mean = float(np.mean(list(per_layer.values()))) if per_layer else float("nan")
    layer_weighted = weighted_sum / weight_total if weight_total else float("nan")
    return per_layer, layer_mean, layer_weighted


def gradient_alignment_score(
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    model_fn: Callable[[], nn.Module],
    K: int = GRAD_ALIGN_K,
    device: torch.device | None = None,
    batch_size: int = GRAD_ALIGN_BATCH,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-layer gradient cosine similarity, averaged over K random inits.

    The syn/real batches are drawn once (seed-fixed) and shared across
    inits, so the K replicate variance isolates initialization randomness.
    """
    device = device or torch.device("cpu")
    syn_batch, syn_y = _deterministic_subset(syn_images, syn_labels, batch_size, seed)
    real_batch, real_y = _deterministic_subset(real_images, real_labels, batch_size, seed)
    per_init_means: list[float] = []
    per_init_weighted: list[float] = []
    per_layer_accum: dict[str, list[float]] = {}
    for k in range(K):
        seed_everything(seed + k)
        model = model_fn().to(device).train()
        per_layer, layer_mean, layer_weighted = _alignment_on_model(
            model, syn_batch, syn_y, real_batch, real_y, device
        )
        per_init_means.append(layer_mean)
        per_init_weighted.append(layer_weighted)
        for name, value in per_layer.items():
            per_layer_accum.setdefault(name, []).append(value)
    return {
        "grad_align_mean": float(np.mean(per_init_means)),
        "grad_align_weighted": float(np.mean(per_init_weighted)),
        "grad_align_std": float(np.std(per_init_means)),
        "per_layer_means": {
            name: float(np.mean(values)) for name, values in per_layer_accum.items()
        },
        "per_init_scores": per_init_means,
    }


def multi_step_gradient_alignment(
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    model_fn: Callable[[], nn.Module],
    steps: tuple[int, ...] = MULTI_STEP_STEPS,
    lr: float = MULTI_STEP_LR,
    K: int = MULTI_STEP_K,
    device: torch.device | None = None,
    batch_size: int = GRAD_ALIGN_BATCH,
    seed: int = 0,
) -> dict[str, Any]:
    """Gradient alignment after 0/1/5/10 full-batch SGD steps on the syn set.

    For each step count and each of K replicates: fresh init, train
    ``step_count`` SGD steps on the fixed syn batch, then measure per-layer
    syn/real gradient cosine on the trained model. ``step_0`` reuses
    :func:`gradient_alignment_score` with the same K.
    """
    device = device or torch.device("cpu")
    syn_batch, syn_y = _deterministic_subset(syn_images, syn_labels, batch_size, seed)
    real_batch, real_y = _deterministic_subset(real_images, real_labels, batch_size, seed)
    results: dict[str, Any] = {}
    base = gradient_alignment_score(
        syn_images, syn_labels, real_images, real_labels, model_fn,
        K=K, device=device, batch_size=batch_size, seed=seed,
    )
    results["step_0"] = {
        "grad_align_mean": base["grad_align_mean"],
        "grad_align_weighted": base["grad_align_weighted"],
    }
    for step_count in steps:
        step_means: list[float] = []
        step_weighted: list[float] = []
        for k in range(K):
            seed_everything(seed + 1000 * step_count + k)
            model = model_fn().to(device).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            for _ in range(step_count):
                with torch.enable_grad():
                    loss = F.cross_entropy(model(syn_batch.to(device)), syn_y.to(device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            _, layer_mean, layer_weighted = _alignment_on_model(
                model, syn_batch, syn_y, real_batch, real_y, device
            )
            step_means.append(layer_mean)
            step_weighted.append(layer_weighted)
        results[f"step_{step_count}"] = {
            "grad_align_mean": float(np.mean(step_means)),
            "grad_align_weighted": float(np.mean(step_weighted)),
            "per_init_scores": step_means,
        }
    return results


def zero_cost_metrics(
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    model_fn: Callable[[], nn.Module],
    device: torch.device | None = None,
    batch_size: int = ZERO_COST_BATCH,
    seed: int = 0,
) -> dict[str, float]:
    """NAS-style zero-cost metrics on one seed-fixed init and syn batch."""
    device = device or torch.device("cpu")
    syn_batch, syn_y = _deterministic_subset(syn_images, syn_labels, batch_size, seed)
    seed_everything(seed)
    model = model_fn().to(device).train()

    activation_hits: list[float] = []
    hooks = []

    def make_hook():
        def hook(module, inputs, output):
            activation_hits.append(float((output.detach() > 0).float().mean()))

        return hook

    for module in model.modules():
        if isinstance(module, ACTIVATION_TYPES):
            hooks.append(module.register_forward_hook(make_hook()))
    with torch.enable_grad():
        loss = F.cross_entropy(model(syn_batch.to(device)), syn_y.to(device))
        loss.backward()
    for hook in hooks:
        hook.remove()

    sq_sum = 0.0
    variances: list[float] = []
    synflow = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        sq_sum += float((grad**2).sum())
        variances.append(float(grad.var(unbiased=False)))
        synflow += float((param.detach().float() * grad).abs().sum())
    model.zero_grad(set_to_none=True)
    return {
        "grad_norm": float(np.sqrt(sq_sum)),
        "grad_variance": float(np.mean(variances)) if variances else float("nan"),
        "activation_diversity": (
            float(np.mean(activation_hits)) if activation_hits else float("nan")
        ),
        "synflow_score": float(synflow),
    }


@torch.no_grad()
def encoder_embeddings(
    encoder: nn.Module,
    images: torch.Tensor,
    device: torch.device | None = None,
    batch_size: int = EMBED_BATCH,
) -> torch.Tensor:
    """Frozen LatentDiffusion features: ``embed(encode(x))`` (no trajectory)."""
    device = device or torch.device("cpu")
    encoder = encoder.to(device).eval()
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        chunks.append(encoder.embed(encoder.encode(batch)).float().cpu())
    return torch.cat(chunks)


def _effective_rank_from_embeddings(embeddings: torch.Tensor) -> float:
    centered = embeddings - embeddings.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = float(singular.sum())
    if total <= 0:
        return 0.0
    p = singular / total
    return float(torch.exp(-(p * torch.log(p + 1e-8)).sum()))


def _energy_rank_90(embeddings: torch.Tensor) -> int:
    centered = embeddings - embeddings.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    total = float(singular.sum())
    if total <= 0:
        return 0
    cumulative = torch.cumsum(singular, dim=0)
    k = int(torch.searchsorted(cumulative, torch.tensor(0.9 * total)).item()) + 1
    return min(k, embeddings.shape[1])


def spectral_rank_score(
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    encoder: nn.Module,
    device: torch.device | None = None,
    real_subsample: int = SPECTRAL_REAL_SUBSAMPLE,
    seed: int = 0,
) -> dict[str, float]:
    """Effective-rank ratios (syn/real) on frozen-encoder embeddings.

    Inspired by ECCV 2024 Neural Spectral Decomposition for DD
    (inspiration_log.md iter-0009 finding 3).
    """
    device = device or torch.device("cpu")
    real_sub, real_sub_y = _deterministic_subset(real_images, real_labels, real_subsample, seed)
    syn_emb = encoder_embeddings(encoder, syn_images, device)
    real_emb = encoder_embeddings(encoder, real_sub, device)

    rank_syn = _effective_rank_from_embeddings(syn_emb)
    rank_real = _effective_rank_from_embeddings(real_emb)
    e90_syn = _energy_rank_90(syn_emb)
    e90_real = _energy_rank_90(real_emb)

    per_class_ratios: list[float] = []
    for cls in torch.unique(real_sub_y).tolist():
        syn_c = syn_emb[syn_labels == cls]
        real_c = real_emb[real_sub_y == cls]
        if len(syn_c) < 2 or len(real_c) < 2:
            continue
        rank_c_real = _effective_rank_from_embeddings(real_c)
        if rank_c_real <= 0:
            continue
        per_class_ratios.append(_effective_rank_from_embeddings(syn_c) / rank_c_real)
    return {
        "spectral_rank_global": rank_syn / rank_real if rank_real > 0 else float("nan"),
        "spectral_rank_per_class_mean": (
            float(np.mean(per_class_ratios)) if per_class_ratios else float("nan")
        ),
        "spectral_rank_90pct_global": e90_syn / e90_real if e90_real > 0 else float("nan"),
    }


def relational_structure_score(
    syn_images: torch.Tensor,
    real_images: torch.Tensor,
    encoder: nn.Module,
    device: torch.device | None = None,
    common_n: int = RELATIONAL_COMMON_N,
    real_cap: int = RELATIONAL_REAL_CAP,
    seed: int = 0,
) -> dict[str, float]:
    """Pairwise-similarity structure alignment between syn and real clouds.

    Both clouds are subsampled to a common ``n = min(N_syn, M_real_capped,
    common_n)`` so the upper-triangular similarity vectors are paired.
    Inspired by CVPR 2024 inter-sample relations for DD
    (inspiration_log.md iter-0009 finding 4).
    """
    from scipy.stats import spearmanr

    device = device or torch.device("cpu")
    n_syn = min(len(syn_images), common_n)
    n_real = min(len(real_images), real_cap, common_n)
    n = min(n_syn, n_real)
    generator = torch.Generator().manual_seed(seed)
    syn_idx = torch.randperm(len(syn_images), generator=generator)[:n]
    real_idx = torch.randperm(len(real_images), generator=generator)[:n]
    syn_emb = encoder_embeddings(encoder, syn_images[syn_idx], device)
    real_emb = encoder_embeddings(encoder, real_images[real_idx], device)
    syn_emb = F.normalize(syn_emb, dim=1)
    real_emb = F.normalize(real_emb, dim=1)
    sim_syn = syn_emb @ syn_emb.T
    sim_real = real_emb @ real_emb.T
    triu = torch.triu_indices(n, n, offset=1)
    vec_syn = sim_syn[triu[0], triu[1]].numpy()
    vec_real = sim_real[triu[0], triu[1]].numpy()
    rho = float(spearmanr(vec_syn, vec_real).statistic)
    numerator = float((sim_syn * sim_real).sum())
    denominator = float((sim_syn * sim_syn).sum().sqrt() * (sim_real * sim_real).sum().sqrt())
    rv = numerator / denominator if denominator > 0 else float("nan")
    return {"relational_spearman": rho, "relational_rv_coefficient": rv}


def importance_weighted_grad_align(
    syn_images: torch.Tensor,
    syn_labels: torch.Tensor,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    model_fn: Callable[[], nn.Module],
    K: int = IMPORTANCE_K,
    temperature: float = IMPORTANCE_TEMPERATURE,
    device: torch.device | None = None,
    batch_size: int = GRAD_ALIGN_BATCH,
    seed: int = 0,
) -> dict[str, Any]:
    """EL2N-style per-sample importance-weighted gradient alignment.

    Per-sample weights are ``softmax(CE_loss_i / temperature)`` at the same
    random init; the weighted gradient ``sum_i w_i d loss_i`` is obtained in
    one backward pass. Inspired by Neural Networks 2024 importance-aware DD
    (inspiration_log.md iter-0009 finding 5).
    """
    device = device or torch.device("cpu")
    syn_batch, syn_y = _deterministic_subset(syn_images, syn_labels, batch_size, seed)
    real_batch, real_y = _deterministic_subset(real_images, real_labels, batch_size, seed)
    per_init_means: list[float] = []
    per_init_weighted: list[float] = []
    for k in range(K):
        seed_everything(seed + k)
        model = model_fn().to(device).train()
        with torch.no_grad():
            syn_losses = F.cross_entropy(
                model(syn_batch.to(device)), syn_y.to(device), reduction="none"
            )
            real_losses = F.cross_entropy(
                model(real_batch.to(device)), real_y.to(device), reduction="none"
            )
        syn_weights = torch.softmax(syn_losses / temperature, dim=0).cpu()
        real_weights = torch.softmax(real_losses / temperature, dim=0).cpu()
        _, layer_mean, layer_weighted = _alignment_on_model(
            model,
            syn_batch,
            syn_y,
            real_batch,
            real_y,
            device,
            syn_weights=syn_weights,
            real_weights=real_weights,
        )
        per_init_means.append(layer_mean)
        per_init_weighted.append(layer_weighted)
    return {
        "grad_align_mean_weighted_by_importance": float(np.mean(per_init_means)),
        "grad_align_weighted_by_importance": float(np.mean(per_init_weighted)),
        "grad_align_std_weighted_by_importance": float(np.std(per_init_means)),
        "per_init_scores": per_init_means,
    }


def compute_candidate_proxies(
    images_z: torch.Tensor,
    labels: torch.Tensor,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    *,
    num_classes: int,
    image_size: int,
    device: torch.device,
    encoder: nn.Module | None = None,
    real_embeddings: torch.Tensor | None = None,
    real_embeddings_labels: torch.Tensor | None = None,
    seed: int = 0,
    include_legacy_static: bool = False,
    label_permutation: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute the full iter-0009 proxy suite for one z-scored candidate.

    ``label_permutation`` (a permutation of ``range(N)``) switches the
    candidate to its label-permuted null variant; relational metrics are
    label-free and are then reported as NaN placeholders (they do not read
    labels, so the null value equals the observed value by construction).
    """
    labels_used = labels[label_permutation] if label_permutation is not None else labels
    null_mode = label_permutation is not None
    model_fn = lambda: classifier("convnet", num_classes, image_size)  # noqa: E731
    out: dict[str, float] = {}

    align = gradient_alignment_score(
        images_z, labels_used, real_images, real_labels, model_fn, device=device, seed=seed
    )
    out["grad_align_mean"] = align["grad_align_mean"]
    out["grad_align_weighted"] = align["grad_align_weighted"]
    out["grad_align_std"] = align["grad_align_std"]

    multi = multi_step_gradient_alignment(
        images_z, labels_used, real_images, real_labels, model_fn, device=device, seed=seed
    )
    for step_key, values in multi.items():
        out[f"grad_align_{step_key}"] = values["grad_align_mean"]

    out.update(zero_cost_metrics(images_z, labels_used, model_fn, device=device, seed=seed))

    weighted = importance_weighted_grad_align(
        images_z, labels_used, real_images, real_labels, model_fn, device=device, seed=seed
    )
    for key in (
        "grad_align_mean_weighted_by_importance",
        "grad_align_weighted_by_importance",
        "grad_align_std_weighted_by_importance",
    ):
        out[key] = weighted[key]

    if encoder is not None:
        spectral = spectral_rank_score(
            images_z, labels_used, real_images, real_labels, encoder, device=device, seed=seed
        )
        out.update(spectral)
        if null_mode:
            out["relational_spearman"] = float("nan")
            out["relational_rv_coefficient"] = float("nan")
        else:
            out.update(
                relational_structure_score(images_z, real_images, encoder, device=device, seed=seed)
            )

    if include_legacy_static and encoder is not None and real_embeddings is not None:
        syn_emb = encoder_embeddings(encoder, images_z, device).numpy()
        real_np = real_embeddings.numpy()
        real_y_np = real_embeddings_labels.numpy()
        labels_np = labels_used.numpy()
        out["ridge_proxy_recomputed"] = _ridge_proxy(syn_emb, labels_np, real_np, real_y_np)
        out["class_moment_dist_recomputed"] = _class_moment_distance(
            syn_emb, labels_np, real_np, real_y_np
        )
        _, _, coverage = _manifold_pr(real_np, syn_emb)
        out["knn_coverage_recomputed"] = coverage
    return out


def compute_all_proxies(
    candidate_dir: str | Path,
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    cfg,
    device: torch.device,
    seed: int = 0,
) -> dict[str, float]:
    """Spec wrapper: load the first ``*.pt`` payload in ``candidate_dir``.

    The payload must provide ``images``/``labels`` (bank convention); images
    are z-scored with the frozen per-dataset statistics before scoring.
    """
    candidate_dir = Path(candidate_dir)
    payloads = sorted(candidate_dir.glob("*.pt"))
    if not payloads:
        raise FileNotFoundError(f"no .pt payload in {candidate_dir}")
    payload = torch.load(payloads[0], map_location="cpu", weights_only=False)
    images = payload["images"].float()
    labels = payload["labels"].long()
    mean, std = DATASET_STATS[cfg.dataset.name]
    mean_t = images.new_tensor(mean)[None, :, None, None]
    std_t = images.new_tensor(std)[None, :, None, None]
    images_z = (images - mean_t) / std_t
    num_classes, image_size = dataset_spec(cfg.dataset.name)
    encoder = build_backbone(cfg, device)
    return compute_candidate_proxies(
        images_z,
        labels,
        real_images,
        real_labels,
        num_classes=num_classes,
        image_size=image_size,
        device=device,
        encoder=encoder,
        seed=seed,
        include_legacy_static=False,
    )
