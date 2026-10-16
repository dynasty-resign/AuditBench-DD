"""Structural blob inspection and blob-to-IPC mapping for AuditBench-DD E0.

The E0 discipline forbids inferring method/IPC/seed from file or directory
names. This module therefore assigns downloaded blobs to declared IPCs using
only structural evidence measured from the tensor payload itself:

* number of images (``n_images == ipc * num_classes`` for a class-balanced
  release, with per-class counts recorded as evidence),
* image tensor shape (NCHW, 3 x image_size x image_size),
* label type decided from label values (hard vs soft, soft never argmaxed),
* value range of the image tensor.

A blob whose structure matches none of the source's declared IPCs stays
unmapped and is reported with a reason; it is never forced into a class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .external_bank import detect_layout


@dataclass
class BlobInfo:
    """Structural signature of one downloaded blob (measured, not inferred)."""

    relpath: str
    sha256: str
    kind: str  # combined | checkpoint_list | images_only | labels_only | unrecognized
    n_images: int = 0
    image_shape: list[int] = field(default_factory=list)
    image_dtype: str = ""
    value_range: list[float] = field(default_factory=list)
    label_type: str = "none"  # hard | soft | none
    n_labels: int = 0
    per_class_counts: dict[int, int] = field(default_factory=dict)
    n_checkpoints: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _label_stats(labels: torch.Tensor, num_classes: int) -> tuple[str, int, dict[int, int]]:
    """Return (label_type, n_labels, per_class_counts) from label values."""
    if labels.ndim == 1 and labels.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        hard = labels.long()
    elif labels.ndim == 1 and labels.is_floating_point() and torch.equal(labels, labels.round()):
        hard = labels.round().long()
    else:
        n = int(labels.shape[0]) if labels.ndim >= 1 else 0
        return "soft", n, {}
    counts = torch.bincount(hard.clamp(0, num_classes - 1), minlength=num_classes)
    per_class = {int(c): int(n) for c, n in enumerate(counts) if int(n) > 0}
    return "hard", int(hard.numel()), per_class


def inspect_blob(path: str | Path, sha256: str = "", num_classes: int = 10) -> BlobInfo:
    """Load one blob and return its structural signature.

    Never raises: load failures are recorded as ``kind="unrecognized"`` with
    the exception string so the mapping report keeps the failure scene.
    """
    path = Path(path)
    info = BlobInfo(relpath=path.name, sha256=sha256, kind="unrecognized")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        info.error = f"{type(exc).__name__}: {exc}"
        return info
    try:
        _fill_from_payload(info, payload, num_classes)
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        info.kind = "unrecognized"
        info.error = f"{type(exc).__name__}: {exc}"
    return info


def _fill_from_payload(info: BlobInfo, payload: Any, num_classes: int) -> None:
    layout = detect_layout(payload)
    if layout == "checkpoint_list":
        series = [item for item in payload["data"] if isinstance(item, (list, tuple))]
        if not series:
            info.error = "checkpoint_list with no usable checkpoints"
            return
        info.kind = "checkpoint_list"
        info.n_checkpoints = len(series)
        images, labels = series[-1][0], series[-1][1]
    elif layout == "canonical":
        info.kind = "combined"
        images, labels = payload["images"], payload["labels"]
    elif layout in ("pair_list", "pair_dict"):
        info.kind = "combined"
        images, labels = (payload[0], payload[1]) if layout == "pair_list" else (
            payload["data"][0], payload["data"][1]
        )
    elif torch.is_tensor(payload):
        tensor = payload.detach().cpu()
        if tensor.ndim == 4:
            info.kind = "images_only"
            images, labels = tensor, None
        elif tensor.ndim in (1, 2):
            info.kind = "labels_only"
            images, labels = None, tensor
        else:
            info.error = f"tensor ndim={tensor.ndim} not image/label-like"
            return
    else:
        info.error = f"unrecognized payload type {type(payload)!r}"
        return

    if images is not None:
        images = images.detach().cpu()
        info.n_images = int(images.shape[0]) if images.ndim == 4 else 0
        info.image_shape = [int(d) for d in images.shape]
        info.image_dtype = str(images.dtype)
        if images.numel():
            info.value_range = [float(images.min()), float(images.max())]
    if labels is not None:
        labels = labels.detach().cpu()
        info.label_type, info.n_labels, info.per_class_counts = _label_stats(labels, num_classes)


def match_ipc(
    info: BlobInfo,
    declared_ipcs: list[int],
    num_classes: int,
    image_size: int,
) -> int | None:
    """Return the declared IPC whose structure matches ``info``, else None.

    Matching uses only measured structure: NCHW shape, image size, and a
    total count equal to ``ipc * num_classes``. Soft-label blobs still match
    by count (the label type is carried as evidence; QC isolates them at
    ingest).
    """
    if info.kind not in {"combined", "checkpoint_list", "images_only"}:
        return None
    if len(info.image_shape) != 4 or info.image_shape[1:] != [3, image_size, image_size]:
        return None
    for ipc in declared_ipcs:
        if info.n_images == ipc * num_classes:
            return ipc
    return None


def propose_file_map(
    record: Any,
    blobs: list[BlobInfo],
    ledgers: dict[str, str],
    num_classes: int = 10,
    image_size: int = 32,
) -> dict[str, Any]:
    """Propose a manifest ``files`` map for one source from staged blobs.

    ``record`` is a ProvenanceRecord (declared ipcs), ``blobs`` the measured
    signatures of every staged blob for this source, and ``ledgers`` maps
    relpath -> origin URL recorded at download time. Returns a dict with
    ``files`` (manifest-ready entries, each carrying structural evidence and
    the blob SHA-256) and ``unmapped`` (relpaths with reasons). Ambiguous
    matches (two blobs matching the same IPC) are rejected for both blobs.
    """
    files: list[dict[str, Any]] = []
    unmapped: list[dict[str, str]] = []
    by_relpath = {b.relpath: b for b in blobs}

    combined = [b for b in blobs if b.kind in {"combined", "checkpoint_list"}]
    images_only = [b for b in blobs if b.kind == "images_only"]
    labels_only = [b for b in blobs if b.kind == "labels_only"]

    claims: dict[int, list[dict[str, Any]]] = {ipc: [] for ipc in record.ipcs}

    for blob in combined:
        ipc = match_ipc(blob, record.ipcs, num_classes, image_size)
        if ipc is None:
            unmapped.append({"relpath": blob.relpath, "reason": _miss_reason(blob, record, num_classes, image_size)})
            continue
        claims[ipc].append({"kind": "single", "blobs": [blob]})

    used_labels: set[str] = set()
    for blob in images_only:
        ipc = match_ipc(blob, record.ipcs, num_classes, image_size)
        partners = [
            lab for lab in labels_only
            if lab.relpath not in used_labels and lab.n_labels == blob.n_images
        ]
        if ipc is None:
            unmapped.append({"relpath": blob.relpath, "reason": _miss_reason(blob, record, num_classes, image_size)})
            continue
        if len(partners) != 1:
            unmapped.append({
                "relpath": blob.relpath,
                "reason": f"expected exactly 1 labels blob with n={blob.n_images}, found {len(partners)}",
            })
            continue
        partner = partners[0]
        used_labels.add(partner.relpath)
        claims[ipc].append({"kind": "pair", "blobs": [blob, partner]})

    for blob in labels_only:
        if blob.relpath not in used_labels:
            unmapped.append({"relpath": blob.relpath, "reason": "labels blob without an images partner"})

    for blob in blobs:
        if blob.kind == "unrecognized":
            unmapped.append({"relpath": blob.relpath, "reason": f"unrecognized: {blob.error}"})

    for ipc, claim_list in claims.items():
        if not claim_list:
            continue
        if len(claim_list) > 1:
            for claim in claim_list:
                for blob in claim["blobs"]:
                    unmapped.append({
                        "relpath": blob.relpath,
                        "reason": f"ambiguous: {len(claim_list)} blobs match ipc={ipc}",
                    })
            continue
        claim = claim_list[0]
        evidence = [
            {
                "relpath": b.relpath,
                "sha256": b.sha256,
                "kind": b.kind,
                "n_images": b.n_images,
                "image_shape": b.image_shape,
                "label_type": b.label_type,
                "per_class_counts": b.per_class_counts,
                "value_range": b.value_range,
            }
            for b in claim["blobs"]
        ]
        if claim["kind"] == "single":
            blob = claim["blobs"][0]
            files.append({
                "relpath": blob.relpath,
                "ipc": ipc,
                "url": ledgers.get(blob.relpath, ""),
                "evidence": evidence,
            })
        else:
            files.append({
                "parts": [
                    {"relpath": b.relpath, "role": ("images" if b.kind == "images_only" else "labels"),
                     "url": ledgers.get(b.relpath, "")}
                    for b in claim["blobs"]
                ],
                "ipc": ipc,
                "evidence": evidence,
            })

    files.sort(key=lambda spec: spec["ipc"])
    return {"source_id": record.source_id, "files": files, "unmapped": unmapped,
            "by_relpath": sorted(by_relpath)}


def _miss_reason(blob: BlobInfo, record: Any, num_classes: int, image_size: int) -> str:
    if len(blob.image_shape) != 4 or blob.image_shape[1:] != [3, image_size, image_size]:
        return f"image shape {blob.image_shape} != NCHW 3x{image_size}x{image_size}"
    expected = sorted({ipc * num_classes for ipc in record.ipcs})
    return f"n_images={blob.n_images} matches none of declared ipc*classes {expected}"
