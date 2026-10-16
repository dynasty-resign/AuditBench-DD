"""iter-0002 redesign #2 (DRR, mechanism-targeted): conditioned-reference,
magnitude-preserving gradient-alignment proxies.

Add-only extension. Nothing in ``gradient_proxy.py`` or any frozen artifact
is modified by this module. All outputs are screening-level evidence; no
inferential statistics.

Failure-mechanism hypotheses addressed (from the iter-0001-claude DRR
diagnosis of redesign #1, failure_class=design):

* M1 (reference mismatch): the reference model is now *conditioned* -- a
  ConvNet briefly trained on the real training pool (first 5000 of the
  seed-0 randperm, identical to iter-0009) -- instead of random
  initializations. A random-init reference ablation (control C-c) directly
  tests M1 with the same formulas.
* M2 (magnitude discarded by cosine): alignment is measured with the raw
  dot product ``<g_real, g_syn>`` (magnitude preserved) and with a
  first-order realized single-step utility ``delta_l_real``. A cosine
  diagnostic on the same vectors separates direction vs magnitude effects.
* M3 (dataset-dependent confounding): handled at the analysis layer via
  per-dataset reporting and mechanism-level controls (no extrapolation).

Mechanism-level controls (same pipeline, same seeds as the observed arm):

* C-a: label-permuted synthetic labels (per-candidate seed-fixed
  permutation, identical convention to iter-0009 ``_entry_seed``).
* C-b: shuffled-real-gradient -- the real labels used for the CE loss and
  for class grouping are permuted once per dataset (seed-fixed), destroying
  the syn<->real label pairing while preserving the real image
  distribution. In batch-mean gradient comparisons the label structure is
  the only pairing that exists; this instantiation is registered in the
  freeze file.
* C-c: random-init reference ablation (same formulas, reference replaced by
  a seed-fixed random-initialization ConvNet).

Metric definitions are frozen in
``runs/evidence_packs/iter0002_redesign2/proxy_definition_freeze_v2.json``
before any candidate computation.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .gradient_proxy import _deterministic_subset
from .models import classifier
from .utils import seed_everything

REFERENCE_TRAIN_SEED = 0
RANDOM_INIT_SEED = 1
CONTROL_B_SEED = 1
TRAIN_EPOCHS = 10
TRAIN_LR = 0.01
TRAIN_MOMENTUM = 0.9
TRAIN_BATCH = 256
STEP_LR = 0.01
GRAD_BATCH = 256
PER_CLASS_REAL_CAP = 64


def train_conditioned_reference(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    image_size: int,
    epochs: int = TRAIN_EPOCHS,
    lr: float = TRAIN_LR,
    momentum: float = TRAIN_MOMENTUM,
    batch_size: int = TRAIN_BATCH,
    seed: int = REFERENCE_TRAIN_SEED,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[float]]:
    """Briefly train a ConvNet on the real pool; return (model, loss history).

    Fully deterministic: fixed seed for init and batch order, no
    augmentation, CPU-feasible. Fixed epoch count (no early stopping); the
    epoch count is part of the frozen definition.
    """
    seed_everything(seed)
    device = device or torch.device("cpu")
    model = classifier("convnet", num_classes, image_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    generator = torch.Generator().manual_seed(seed)
    n = len(images)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        total_loss, total_n = 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = images[idx].to(device)
            yb = labels[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)
            total_n += len(idx)
        history.append(total_loss / max(total_n, 1))
    model.eval()
    model.zero_grad(set_to_none=True)
    return model, history


def random_init_reference(
    *, num_classes: int, image_size: int, seed: int = RANDOM_INIT_SEED
) -> nn.Module:
    """Seed-fixed random-initialization ConvNet (control C-c reference)."""
    seed_everything(seed)
    model = classifier("convnet", num_classes, image_size)
    model.eval()
    model.zero_grad(set_to_none=True)
    return model


def _tracked_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def grad_vector(
    model: nn.Module,
    params: list[nn.Parameter],
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Flattened full-parameter gradient of the mean CE loss (no .grad state)."""
    logits = model(images.to(device))
    loss = F.cross_entropy(logits, labels.to(device))
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    pieces = [
        (g if g is not None else torch.zeros_like(p)).detach().float().flatten()
        for g, p in zip(grads, params)
    ]
    return torch.cat(pieces)


def per_class_grad_vectors(
    model: nn.Module,
    params: list[nn.Parameter],
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    *,
    per_class_cap: int | None = None,
    seed: int = 0,
) -> dict[int, torch.Tensor]:
    """Mean CE gradient per class; classes without samples are skipped."""
    out: dict[int, torch.Tensor] = {}
    for cls in range(num_classes):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        if per_class_cap is not None and len(idx) > per_class_cap:
            generator = torch.Generator().manual_seed(seed)
            pick = torch.randperm(len(idx), generator=generator)[:per_class_cap]
            idx = idx[pick]
        out[cls] = grad_vector(model, params, images[idx], labels[idx], device)
    return out


def prepare_real_side(
    model: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    heldout_images: torch.Tensor,
    heldout_labels: torch.Tensor,
    *,
    num_classes: int,
    device: torch.device,
    batch_size: int = GRAD_BATCH,
    per_class_cap: int = PER_CLASS_REAL_CAP,
    seed: int = 0,
    label_permutation: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Candidate-independent real-side gradient vectors at a fixed reference.

    Returns ``g_train`` (256-sample seed-fixed subset of the train pool),
    ``g_heldout`` (256-sample seed-fixed subset of the held-out pool) and
    ``per_class`` (per-class mean gradients on the train pool, capped).
    ``label_permutation`` (control C-b) permutes the real labels used by the
    CE loss and the class grouping.
    """
    params = _tracked_parameters(model)
    if label_permutation is not None:
        train_labels = train_labels[label_permutation]
        heldout_labels = heldout_labels[label_permutation]
    real_batch, real_y = _deterministic_subset(train_images, train_labels, batch_size, seed)
    held_batch, held_y = _deterministic_subset(heldout_images, heldout_labels, batch_size, seed)
    return {
        "g_train": grad_vector(model, params, real_batch, real_y, device),
        "g_heldout": grad_vector(model, params, held_batch, held_y, device),
        "per_class": per_class_grad_vectors(
            model,
            params,
            train_images,
            train_labels,
            num_classes,
            device,
            per_class_cap=per_class_cap,
            seed=seed,
        ),
    }


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a @ b)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm() + 1e-12
    return float((a @ b) / denom)


def metrics_from_syn_grads(
    g_syn: torch.Tensor,
    syn_per_class: dict[int, torch.Tensor],
    real_side: dict[str, Any],
    *,
    step_lr: float = STEP_LR,
) -> dict[str, float]:
    """Metric values from precomputed syn-side and real-side gradients."""
    dot = _dot(g_syn, real_side["g_train"])
    out = {
        "grad_dot_cond": dot,
        "delta_l_real": -step_lr * _dot(g_syn, real_side["g_heldout"]),
        "grad_cos_cond_diag": _cosine(g_syn, real_side["g_train"]),
    }
    shared = sorted(set(syn_per_class) & set(real_side["per_class"]))
    if shared:
        out["grad_dot_cond_within_class"] = float(
            sum(_dot(syn_per_class[c], real_side["per_class"][c]) for c in shared) / len(shared)
        )
        out["num_classes_within"] = float(len(shared))
    else:
        out["grad_dot_cond_within_class"] = float("nan")
        out["num_classes_within"] = 0.0
    return out


def compute_conditioned_metrics(
    images_z: torch.Tensor,
    labels: torch.Tensor,
    reference: nn.Module,
    real_side: dict[str, Any],
    *,
    num_classes: int,
    device: torch.device,
    batch_size: int = GRAD_BATCH,
    seed: int = 0,
    syn_label_permutation: torch.Tensor | None = None,
    step_lr: float = STEP_LR,
) -> dict[str, float]:
    """Observed-arm metrics for one candidate at a fixed reference model.

    ``syn_label_permutation`` (control C-a) permutes the candidate labels
    before batching (identical convention to iter-0009).
    """
    labels_used = labels[syn_label_permutation] if syn_label_permutation is not None else labels
    params = _tracked_parameters(reference)
    syn_batch, syn_y = _deterministic_subset(images_z, labels_used, batch_size, seed)
    g_syn = grad_vector(reference, params, syn_batch, syn_y, device)
    syn_per_class = per_class_grad_vectors(
        reference, params, syn_batch, syn_y, num_classes, device
    )
    return metrics_from_syn_grads(g_syn, syn_per_class, real_side, step_lr=step_lr)
