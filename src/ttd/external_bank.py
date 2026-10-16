"""External condensed-set bank: layout adapters, normalization, QC and dedup.

Every artifact that passes provenance verification is converted into the
canonical internal format::

    {"images": float32 NCHW in [0, 1], "labels": int64 [N], "meta": {...}}

which is exactly what :func:`ttd.pipeline.evaluate` consumes, so E1 can train
students on external artifacts without touching raw vendor layouts again.

Deliberate deviations from the historical Card C tooling (forbidden by the
E0 task): method/IPC/seed are never inferred from directory or file names,
the FINAL checkpoint of multi-checkpoint releases is used instead of the
first glob match, cache keys are content fingerprints plus manifest
``source_id`` (never run names), and the method taxonomy comes from the
manifest/config instead of a hardcoded table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .provenance import SOURCE_TIERS, ProvenanceRecord, iter_spec_parts, verify_blob
from .utils import tensor_fingerprint

CANONICAL_IMAGE_RANGE = (-0.01, 1.01)

# CIFAR-10 per-channel z-score constants. Verified 2026-09-05 against the
# official CIFAR-10 train set: inverting the DC-BENCH "random" baseline blobs
# with these constants reproduces uint8-quantized [0,1] images (max residual
# 1.5e-5) that match 620/620 real training images exactly, with agreeing
# labels (see scripts/verify_zscore_inversion.py and the iter-0002 report).
CIFAR10_ZSCORE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_ZSCORE_STD = (0.2023, 0.1994, 0.2010)

# Per-dataset z-score inversion constants for z-score-stored releases
# (selection baselines). Each dataset's constants are pinned only after the
# same empirical verification as CIFAR-10 (uint8-quantization residual plus
# exact image/label match against the official train set).
#
# CIFAR-100 note (measured 2026-09-06, iter-0005): the commonly quoted
# torchvision CIFAR-100 stats ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565,
# 0.2761)) do NOT exactly invert the DC-BENCH CIFAR-100 random blobs
# (max quantization residual 0.097/255, 0/100 exact matches at 1e-4). The
# constants below were fitted per channel from 100 matched (blob, real)
# image pairs and verify with residual 1.2e-5 (255-scale) and 100/100 exact
# image+label matches at 1e-4; they reproduce the official train images
# exactly after uint8 quantization for all 6100 selection images. See
# scripts/verify_zscore_inversion_cifar100.py and the iter-0005 report.
ZSCORE_INVERSION_CONSTANTS = {
    "cifar10": (CIFAR10_ZSCORE_MEAN, CIFAR10_ZSCORE_STD),
    "cifar100": ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
    # Tiny-ImageNet (measured 2026-09-06, iter-0009): the DC-BENCH tiny random
    # selection blobs are z-scored with the ImageNet normalization constants
    # (identified from the blobs' own per-channel statistics, then verified by
    # the E0/E2-grade check: uint8 quantization residual 1.5e-5 (< 1e-3),
    # 2000/2000 + 10000/10000 exact image matches with full label agreement
    # against the official train split; see
    # scripts/iter0009/verify_zscore_inversion_tinyimagenet.py and
    # runs/evidence_packs/iter0009/bank_staging/zscore_inversion_verification.json).
    # Note these are NOT the Tiny-ImageNet train-split statistics (those are
    # empirically pinned in data.DATASET_STATS by the iter-0009 T2 pin).
    "tiny_imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

# Strata: primary = hard-label natural-image cohort; everything else is kept
# but isolated so it can never leak into the primary ranking.
STRATUM_PRIMARY = "primary"
STRATUM_EXTREME = "extreme_compression"  # IPC1: registered, not in coverage gates
STRATUM_SOFT_LABEL = "soft_label_dependent"
STRATUM_ZCA = "zca_only"
STRATUM_DECODER = "decoder_dependent"
STRATUM_QUARANTINE = "quarantine"

QUARANTINE_REASONS = (
    "unrecognized_layout",
    "suspect_preprocessing",
    "hash_mismatch",
    "label_type_mismatch",
    "bad_shape",
    "unmapped_blob",
    "duplicate",
)


@dataclass
class NormalizedArtifact:
    """Result of normalizing one raw blob into the canonical format."""

    entry_id: str
    source_id: str
    release_lineage: str
    method: str
    method_family: str
    distribution: str
    dataset: str
    ipc: int
    n_images: int
    per_class_counts: dict[int, int]
    content_fingerprint: str
    raw_layout: str
    checkpoint_used: str
    value_range_detected: list[float]
    label_type_observed: str  # hard | soft
    stratum: str
    quarantine_reasons: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_layout(payload: Any) -> str:
    """Classify the raw serialization layout without trusting file names."""
    if isinstance(payload, dict):
        keys = set(payload)
        if {"images", "labels"} <= keys:
            return "canonical"
        if "data" in keys and isinstance(payload["data"], (list, tuple)):
            data = payload["data"]
            if (
                len(data) == 2
                and torch.is_tensor(data[0])
                and torch.is_tensor(data[1])
                and data[0].ndim == 4
            ):
                return "pair_dict"
            return "checkpoint_list"
    if (
        isinstance(payload, (list, tuple))
        and len(payload) == 2
        and torch.is_tensor(payload[0])
        and torch.is_tensor(payload[1])
    ):
        return "pair_list"
    return "unknown"


def _extract_tensors(payload: Any) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    """Return (images, labels, raw_layout, checkpoint_used)."""
    layout = detect_layout(payload)
    if layout == "canonical":
        return payload["images"], payload["labels"], layout, "single"
    if layout == "checkpoint_list":
        series = [item for item in payload["data"] if isinstance(item, (list, tuple))]
        if not series:
            raise ValueError("checkpoint_list layout with no usable checkpoints")
        # Always the FINAL checkpoint of the release; never the first match.
        images, labels = series[-1][0], series[-1][1]
        return images, labels, layout, f"last_of_{len(series)}"
    if layout == "pair_dict":
        return payload["data"][0], payload["data"][1], layout, "single"
    if layout == "pair_list":
        return payload[0], payload[1], layout, "single"
    raise ValueError(f"unrecognized_layout: {type(payload)!r}")


def _classify_labels(labels: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Return (label_type, hard_labels). Soft labels are never argmaxed."""
    if labels.ndim == 1 and labels.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        return "hard", labels.long()
    if labels.ndim == 1 and labels.is_floating_point():
        rounded = labels.round()
        if torch.equal(labels, rounded):
            return "hard", rounded.long()
        return "soft", torch.zeros(0, dtype=torch.long)
    return "soft", torch.zeros(0, dtype=torch.long)


def _normalize_images(
    images: torch.Tensor,
    declared_preprocessing: str,
    dataset: str = "cifar10",
) -> tuple[torch.Tensor, list[float], list[str]]:
    """Return (images float32 in [0,1], detected range, quarantine reasons)."""
    reasons: list[str] = []
    out = images.detach().cpu()
    if out.dtype == torch.uint8:
        out = out.float() / 255.0
    elif out.is_floating_point():
        out = out.float()
    else:
        out = out.float()
    detected = [float(out.min()), float(out.max())]
    if declared_preprocessing == "zca":
        reasons.append(STRATUM_ZCA)
        return out, detected, reasons
    if declared_preprocessing == "zscore":
        # Released blobs are per-channel z-scored real images; invert with the
        # pinned, verified per-dataset constants (see constants above).
        if dataset not in ZSCORE_INVERSION_CONSTANTS:
            msg = 'suspect_preprocessing: no verified zscore inversion constants for '
            reasons.append(msg + dataset)
            return out, detected, reasons
        c_mean, c_std = ZSCORE_INVERSION_CONSTANTS[dataset]
        mean = torch.tensor(c_mean).view(1, 3, 1, 1)
        std = torch.tensor(c_std).view(1, 3, 1, 1)
        return (out * std + mean).clamp(0.0, 1.0), detected, reasons
    if declared_preprocessing == "raw_unbounded":
        # Distillation releases whose synthetic pixels are optimized as free
        # (unclamped) float tensors; the released tensor IS the artifact, so
        # it is stored verbatim and the detected range is recorded. Clamping
        # to [0,1] would destroy the audited content.
        return out, detected, reasons
    low, high = CANONICAL_IMAGE_RANGE
    out_of_range = detected[0] < low or detected[1] > high
    if out_of_range and declared_preprocessing in ("raw", "unknown"):
        # Negative or >1 values without a declared normalization scheme:
        # semantics unresolved, do not silently rescale.
        reasons.append("suspect_preprocessing")
        return out, detected, reasons
    return out.clamp(0.0, 1.0), detected, reasons


def normalize_blob(
    record: ProvenanceRecord,
    blob_path: str | Path,
    ipc: int,
    num_classes: int = 10,
    image_size: int = 32,
) -> tuple[NormalizedArtifact, dict[str, torch.Tensor] | None]:
    """Normalize one raw blob; returns (artifact metadata, canonical payload).

    The payload is ``None`` when the artifact cannot be represented (unknown
    layout). Quarantined artifacts still return their payload for auditing
    but are excluded from the primary cohort via ``stratum``.
    """
    try:
        raw = torch.load(blob_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        return normalize_payload(record, None, ipc, num_classes, image_size, load_error=exc)
    return normalize_payload(record, raw, ipc, num_classes, image_size)


def assemble_parts(
    part_payloads: list[tuple[str, Any]],
) -> Any:
    """Assemble a multi-part artifact into a pair_list payload.

    ``part_payloads`` is a list of ``(role, loaded_object)`` tuples. Roles
    ``images`` and ``labels`` combine into ``[images, labels]``; a single
    ``blob`` role passes through unchanged. Anything else raises so the
    caller records an ``unrecognized_layout`` quarantine.
    """
    if len(part_payloads) == 1 and part_payloads[0][0] == "blob":
        return part_payloads[0][1]
    by_role = {role: obj for role, obj in part_payloads}
    if "images" in by_role and "labels" in by_role:
        return [by_role["images"], by_role["labels"]]
    raise ValueError(f"multi-part spec lacks images/labels roles: {sorted(by_role)}")


def normalize_payload(
    record: ProvenanceRecord,
    raw: Any,
    ipc: int,
    num_classes: int = 10,
    image_size: int = 32,
    load_error: Exception | None = None,
) -> tuple[NormalizedArtifact, dict[str, torch.Tensor] | None]:
    """Normalize an already-loaded payload into the canonical format."""
    entry_id = f"{record.source_id}.ipc{ipc}"
    quarantine: list[str] = []
    payload: dict[str, torch.Tensor] | None = None
    raw_layout = "unknown"
    checkpoint_used = "n/a"
    detected_range = [float("nan"), float("nan")]
    label_type = "unknown"
    per_class: dict[int, int] = {}
    n_images = 0
    fingerprint = "n/a"

    if load_error is not None:
        quarantine.append(f"unrecognized_layout: {load_error}")
    else:
        try:
            images, labels, raw_layout, checkpoint_used = _extract_tensors(raw)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            quarantine.append(f"unrecognized_layout: {exc}")
        else:
            images, detected_range, img_reasons = _normalize_images(
                images, record.preprocessing_expected, record.dataset
            )
            quarantine.extend(img_reasons)
            label_type, hard_labels = _classify_labels(labels)
            if label_type == "soft":
                quarantine.append(STRATUM_SOFT_LABEL)
            if images.ndim != 4 or images.shape[1:] != (3, image_size, image_size):
                quarantine.append("bad_shape")
            n_images = int(images.shape[0]) if images.ndim == 4 else 0
            if label_type == "hard":
                if len(hard_labels) != n_images:
                    quarantine.append("bad_shape")
                else:
                    counts = torch.bincount(hard_labels.clamp(0, num_classes - 1), minlength=num_classes)
                    per_class = {int(c): int(n) for c, n in enumerate(counts) if int(n) > 0}
                    out_of_range = int(((hard_labels < 0) | (hard_labels >= num_classes)).sum())
                    if out_of_range:
                        quarantine.append("bad_shape: labels out of range")
            if n_images and n_images != ipc * num_classes:
                # Non-fatal: class-balanced releases are expected but not assumed.
                quarantine.append(f"count_mismatch: expected {ipc * num_classes}, got {n_images}")
            if n_images and label_type == "hard":
                fingerprint = tensor_fingerprint(images, hard_labels)
                payload = {"images": images, "labels": hard_labels}

    stratum = STRATUM_PRIMARY
    if STRATUM_SOFT_LABEL in quarantine:
        stratum = STRATUM_SOFT_LABEL
    elif STRATUM_ZCA in quarantine:
        stratum = STRATUM_ZCA
    elif record.preprocessing_expected == "generator":
        stratum = STRATUM_DECODER
    elif ipc == 1:
        stratum = STRATUM_EXTREME
    hard_blockers = [r for r in quarantine if r.split(":")[0] in (
        "unrecognized_layout",
        "suspect_preprocessing",
        "bad_shape",
        "hash_mismatch",
    )]
    if hard_blockers:
        stratum = STRATUM_QUARANTINE

    artifact = NormalizedArtifact(
        entry_id=entry_id,
        source_id=record.source_id,
        release_lineage=record.release_lineage,
        method=record.method,
        method_family=record.method_family,
        distribution=record.distribution,
        dataset=record.dataset,
        ipc=ipc,
        n_images=n_images,
        per_class_counts=per_class,
        content_fingerprint=fingerprint,
        raw_layout=raw_layout,
        checkpoint_used=checkpoint_used,
        value_range_detected=detected_range,
        label_type_observed=label_type,
        stratum=stratum,
        quarantine_reasons=quarantine,
        provenance={
            "source_tier": record.source_tier,
            "tier_label": SOURCE_TIERS.get(record.source_tier, "unknown"),
            "official_urls": list(record.official_urls),
            "license": record.license,
            "notes": record.notes,
        },
    )
    return artifact, payload


def build_bank(
    records: list[ProvenanceRecord],
    raw_dir: str | Path,
    out_dir: str | Path,
    num_classes: int = 10,
    image_size: int = 32,
) -> dict[str, Any]:
    """Verify, normalize and dedup every fetched blob into the bank.

    Returns the bank index (also written to ``out_dir/bank_index.json``).
    Multi-part file specs (separate images/labels blobs) are verified part by
    part and assembled by declared role before normalization.
    """
    from .utils import save_json

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    bank_dir = out_dir / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen_fingerprints: dict[str, str] = {}  # fingerprint -> entry_id kept
    tier_by_entry: dict[str, int] = {}

    for record in sorted(records, key=lambda r: r.source_tier):
        for file_spec in record.files:
            ipc = int(file_spec["ipc"])
            parts = iter_spec_parts(file_spec)
            part_paths = [(part, raw_dir / record.source_id / part["relpath"]) for part in parts]
            entry_stub = {
                "entry_id": f"{record.source_id}.ipc{ipc}",
                "source_id": record.source_id,
                "release_lineage": record.release_lineage,
                "method": record.method,
                "method_family": record.method_family,
                "ipc": ipc,
                "provenance": {"source_tier": record.source_tier},
            }
            missing = [part["relpath"] for part, blob in part_paths if not blob.exists()]
            if missing:
                entries.append(
                    {
                        **entry_stub,
                        "stratum": STRATUM_QUARANTINE,
                        "quarantine_reasons": ["missing_blob"],
                        "missing_relpaths": missing,
                    }
                )
                continue
            checks = [verify_blob(record, part["relpath"], blob) for part, blob in part_paths]
            if not all(check["hash_verified"] for check in checks):
                entries.append(
                    {
                        **entry_stub,
                        "stratum": STRATUM_QUARANTINE,
                        "quarantine_reasons": ["hash_mismatch"],
                        "hash_checks": checks,
                    }
                )
                continue
            if len(part_paths) == 1 and part_paths[0][0]["role"] == "blob":
                artifact, payload = normalize_blob(
                    record, part_paths[0][1], ipc, num_classes, image_size
                )
            else:
                loaded: list[tuple[str, Any]] = []
                load_error: Exception | None = None
                for part, blob in part_paths:
                    try:
                        loaded.append((part["role"], torch.load(blob, map_location="cpu", weights_only=False)))
                    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                        load_error = exc
                        break
                if load_error is None:
                    try:
                        raw_payload = assemble_parts(loaded)
                    except Exception as exc:  # noqa: BLE001
                        load_error = exc
                        raw_payload = None
                artifact, payload = normalize_payload(
                    record,
                    None if load_error is not None else raw_payload,
                    ipc,
                    num_classes,
                    image_size,
                    load_error=load_error,
                )
            row = artifact.to_dict()
            row["hash_checks"] = checks
            if payload is not None:
                fp = artifact.content_fingerprint
                if fp in seen_fingerprints:
                    kept = seen_fingerprints[fp]
                    duplicates.append(
                        {
                            "fingerprint": fp,
                            "kept": kept,
                            "dropped": artifact.entry_id,
                            "reason": "identical content under a different source",
                        }
                    )
                    row["stratum"] = STRATUM_QUARANTINE
                    if "duplicate" not in row["quarantine_reasons"]:
                        row["quarantine_reasons"].append("duplicate")
                else:
                    seen_fingerprints[fp] = artifact.entry_id
                    tier_by_entry[artifact.entry_id] = record.source_tier
                    blob_out = bank_dir / f"{artifact.entry_id}.pt"
                    meta = {
                        "entry_id": artifact.entry_id,
                        "source_id": record.source_id,
                        "release_lineage": record.release_lineage,
                        "method": record.method,
                        "method_family": record.method_family,
                        "distribution": record.distribution,
                        "ipc": ipc,
                        "stratum": artifact.stratum,
                        "provenance": artifact.provenance,
                        "hash_checks": checks,
                        "config": {
                            "dataset": {"name": record.dataset, "image_size": image_size}
                        },
                    }
                    torch.save({**payload, "meta": meta}, blob_out)
                    row["bank_path"] = str(blob_out)
            entries.append(row)

    index = {
        "dataset": records[0].dataset if records else "unknown",
        "num_classes": num_classes,
        "image_size": image_size,
        "entries": entries,
        "duplicates": duplicates,
    }
    save_json(index, out_dir / "bank_index.json")
    return index
