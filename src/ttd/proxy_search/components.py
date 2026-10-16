"""iter-0003 proxy-search component library.

New zero-training proxy components frozen in
runs/evidence_packs/iter0003_search/search_preregistration.json (sha256-pinned
before any computation). Two families:

* class-conditional stratified matching (CCDM-inspired, arXiv 2606.19817):
  class-conditional mean/diagonal-variance matching on frozen-encoder
  embeddings plus a class-composition JSD penalty;
* structural composition statistics (Structural-Assessment-inspired, arXiv
  2606.21705): JSD / HHI / coverage over k-means token histograms of
  frozen-encoder embeddings.

Plus the synflow per-class variant of the iter0009 synflow saliency.

All outputs are screening-level descriptive statistics. Nothing here trains a
student model, touches E4/E5, or modifies any pre-existing frozen artifact.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ..models import classifier
from ..utils import seed_everything

EPS = 1e-9
TOKEN_CLUSTERS = 64
TOKEN_RANDOM_STATE = 0
TOKEN_N_INIT = 10
STD_FLOOR = 1e-8


def jsd_base2(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two probability vectors."""
    p = np.asarray(p, dtype=np.float64) + EPS
    q = np.asarray(q, dtype=np.float64) + EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def synflow_per_class(
    images_z: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    image_size: int,
    seed: int = 0,
    device: torch.device | None = None,
) -> float:
    """Macro-averaged per-class synflow saliency on one seed-fixed convnet init.

    For each class c present in ``labels``: L_c = mean cross-entropy over
    class-c samples only; S_c = sum over parameters of |w * dL_c/dw| (same
    functional form as src/ttd/gradient_proxy.py zero_cost_metrics). Returns
    the macro-average of S_c over present classes. All candidate samples are
    used (no batch subsampling).
    """
    device = device or torch.device("cpu")
    seed_everything(seed)
    model = classifier("convnet", num_classes, image_size).to(device).train()
    labels = labels.long()
    present = torch.unique(labels).tolist()
    per_class: list[float] = []
    for cls in present:
        mask = labels == cls
        images_c = images_z[mask].to(device)
        labels_c = labels[mask].to(device)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(images_c), labels_c)
            loss.backward()
        saliency = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            saliency += float((param.detach().float() * param.grad.detach().float()).abs().sum())
        per_class.append(saliency)
    model.zero_grad(set_to_none=True)
    return float(np.mean(per_class)) if per_class else float("nan")


def reference_zscore(
    embeddings: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score ``embeddings`` with per-dimension reference mean/std (floored)."""
    mean = reference.mean(0, keepdim=True)
    std = reference.std(0, keepdim=True, unbiased=False).clamp_min(STD_FLOOR)
    return (embeddings - mean) / std, mean, std


def class_conditional_moments_raw(
    syn_emb_z: torch.Tensor,
    syn_labels: torch.Tensor,
    real_emb_z: torch.Tensor,
    real_labels: torch.Tensor,
) -> float:
    """Negative class-conditional mean + diagonal-variance matching distance.

    raw = -( mean_c ||mu_syn_c - mu_real_c||_2/sqrt(d)
            + mean_c ||v_syn_c - v_real_c||_2/sqrt(d) )
    over classes present in the candidate labels; diagonal variances use
    unbiased=False; d = embedding dimension.
    """
    dim = syn_emb_z.shape[1]
    scale = float(np.sqrt(dim))
    syn_labels = syn_labels.long()
    real_labels = real_labels.long()
    mean_terms: list[float] = []
    var_terms: list[float] = []
    for cls in torch.unique(syn_labels).tolist():
        syn_c = syn_emb_z[syn_labels == cls]
        real_c = real_emb_z[real_labels == cls]
        if len(syn_c) == 0 or len(real_c) == 0:
            continue
        mean_terms.append(float((syn_c.mean(0) - real_c.mean(0)).norm()) / scale)
        syn_var = syn_c.var(0, unbiased=False) if len(syn_c) > 1 else torch.zeros(dim)
        real_var = real_c.var(0, unbiased=False)
        var_terms.append(float((syn_var - real_var).norm()) / scale)
    if not mean_terms:
        return float("nan")
    return -(float(np.mean(mean_terms)) + float(np.mean(var_terms)))


def class_histogram(labels: torch.Tensor, num_classes: int) -> np.ndarray:
    counts = torch.bincount(labels.long(), minlength=num_classes).numpy().astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def composition_jsd(
    syn_labels: torch.Tensor, real_labels: torch.Tensor, num_classes: int
) -> float:
    """JSD (base 2) between candidate and real-pool class histograms."""
    return jsd_base2(class_histogram(syn_labels, num_classes), class_histogram(real_labels, num_classes))


def fit_token_model(real_emb_z: torch.Tensor, n_clusters: int = TOKEN_CLUSTERS):
    """Frozen k-means tokenizer: KMeans(n_clusters=64, random_state=0, n_init=10)."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, random_state=TOKEN_RANDOM_STATE, n_init=TOKEN_N_INIT)
    model.fit(real_emb_z.numpy().astype(np.float64))
    return model


def assign_tokens(model, emb_z: torch.Tensor) -> np.ndarray:
    return model.predict(emb_z.numpy().astype(np.float64))


def _token_hist(tokens: np.ndarray, n_tokens: int) -> np.ndarray:
    counts = np.bincount(tokens, minlength=n_tokens).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def struct_token_jsd(
    syn_tokens: np.ndarray,
    syn_labels: torch.Tensor,
    real_tokens: np.ndarray,
    real_labels: torch.Tensor,
    n_tokens: int = TOKEN_CLUSTERS,
) -> float:
    """- mean_c JSD_base2(H_syn_c, H_real_c) over classes in candidate labels."""
    syn_labels_np = syn_labels.long().numpy()
    real_labels_np = real_labels.long().numpy()
    values: list[float] = []
    for cls in np.unique(syn_labels_np):
        syn_c = syn_tokens[syn_labels_np == cls]
        real_c = real_tokens[real_labels_np == cls]
        if len(syn_c) == 0 or len(real_c) == 0:
            continue
        values.append(jsd_base2(_token_hist(syn_c, n_tokens), _token_hist(real_c, n_tokens)))
    if not values:
        return float("nan")
    return -float(np.mean(values))


def struct_token_hhi(syn_tokens: np.ndarray, n_tokens: int = TOKEN_CLUSTERS) -> float:
    """Negative Herfindahl-Hirschman index of the candidate token histogram."""
    hist = _token_hist(syn_tokens, n_tokens)
    return -float(np.sum(hist**2))


def struct_token_cov(
    syn_tokens: np.ndarray,
    syn_labels: torch.Tensor,
    real_tokens: np.ndarray,
    real_labels: torch.Tensor,
) -> float:
    """Mean per-class coverage: |tokens(c,syn) ∩ tokens(c,real)| / |tokens(c,real)|."""
    syn_labels_np = syn_labels.long().numpy()
    real_labels_np = real_labels.long().numpy()
    values: list[float] = []
    for cls in np.unique(syn_labels_np):
        syn_set = set(syn_tokens[syn_labels_np == cls].tolist())
        real_set = set(real_tokens[real_labels_np == cls].tolist())
        if not real_set:
            continue
        values.append(len(syn_set & real_set) / len(real_set))
    if not values:
        return float("nan")
    return float(np.mean(values))


def rank_average(component_vectors: list[np.ndarray]) -> np.ndarray:
    """Unweighted average of per-component ranks (scipy rankdata, average ties)."""
    from scipy.stats import rankdata

    if not component_vectors:
        raise ValueError("rank_average needs at least one component")
    ranks = [rankdata(np.asarray(vec, dtype=np.float64), method="average") for vec in component_vectors]
    return np.mean(ranks, axis=0)