"""AuditBench-DD E3 module skeleton (Tiny-ImageNet; design-round readiness only).

This module is the code anchor of runs/tiny_imagenet/auditbench_e3/e3_preregistration.json.
It provides:

- the frozen cohort identifiers (10 design entries; the TM ipc50 entry expands to 5 expert
  replica measurement entries under the pre-registered tm_replica_rule);
- the 64px evaluation-architecture factory (the 4 frozen architectures via
  models.classifier at image_size=64; ConvNetD4 is provenance-only and is NEVER built here);
- the expected on-disk data paths for the real dataset (path constants only; this module
  never loads the real dataset and never downloads anything);
- a schema/tensor-shape level smoke check (random tensors, CPU-capable, seconds-level).

Discipline (iter-0008): this skeleton is NOT wired into any training execution path. No
utility is measured here; no smoke numeric output may feed any inference
(smoke_*_recorded_schema_only discipline). The z-score constants for tiny_imagenet must be
empirically pinned on the official train split before any formal E3 utility (E0/E2 lesson;
see the pre-registration input_normalization.zscore_pin_plan); the inherited published
constants in data.DATASET_STATS are never trusted silently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .data import DATASET_STATS, dataset_spec
from .models import classifier

DATASET_NAME = "tiny_imagenet"
NUM_CLASSES, IMAGE_SIZE = dataset_spec(DATASET_NAME)

REPO = Path(__file__).resolve().parents[2]
PREREGISTRATION_PATH = (
    REPO / "runs" / "tiny_imagenet" / "auditbench_e3" / "e3_preregistration.json"
)

EVALUATION_ARCHITECTURES = ("convnet", "resnet18", "vgg11", "mobilenet_v2")

PRIMARY_ENTRY_IDS = (
    "dcbench.dc.ipc10",
    "dcbench.dc.ipc50",
    "dcbench.dm.ipc10",
    "dcbench.dm.ipc50",
    "dcbench.dsa.ipc10",
    "dcbench.dsa.ipc50",
    "dcbench.tm_mtt.ipc10",
    "dcbench.tm_mtt.ipc50",
    "dcbench.random.ipc10",
    "dcbench.random.ipc50",
)

TM_IPC50_REPLICA_IDS = tuple(f"dcbench.tm_mtt.ipc50_r{r}" for r in range(5))

# ConvNetD4 is registered ONLY as the distillation source architecture of the DC-BENCH tiny
# artifacts (provenance); it never enters the evaluation architecture family (LOAO control).
CONVNETD4_PROVENANCE_ONLY = True

# z-score empirical pin plan (summary; authoritative text lives in the pre-registration):
# before any E3 formal utility, measure per-channel mean/std on the official Tiny-ImageNet
# train split (post-download-authorization), record n and seed, and update DATASET_STATS if
# the measured values deviate; silent use of the inherited constants is forbidden.
ZSCORE_PIN_REQUIRED = True
REGISTERED_ZSCORE = DATASET_STATS[DATASET_NAME]


def load_preregistration(path: str | Path | None = None) -> dict[str, Any]:
    """Load the frozen E3 pre-registration JSON (read-only)."""
    prereg_path = Path(path) if path is not None else PREREGISTRATION_PATH
    return json.loads(prereg_path.read_text(encoding="utf-8"))


def cohort_entries(prereg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the 10 primary design entries with their A3 grounding metadata."""
    return list(prereg["cohort"]["entries"])


def expand_measurement_entries(prereg: dict[str, Any]) -> list[str]:
    """Expand design entries into measurement entries (TM ipc50 -> 5 replicas, D3)."""
    ids: list[str] = []
    for entry in prereg["cohort"]["entries"]:
        replicas = entry.get("replica_entry_ids")
        ids.extend(replicas if replicas else [entry["entry_id"]])
    return ids


def expected_data_paths(root: str | Path) -> dict[str, Path]:
    """Expected real-dataset layout (constants only; never reads or downloads)."""
    base = Path(root) / "tiny-imagenet-200"
    return {
        "base": base,
        "train": base / "train",
        "val": base / "val",
    }


def build_evaluation_architectures(
    num_classes: int = NUM_CLASSES, image_size: int = IMAGE_SIZE
) -> dict[str, torch.nn.Module]:
    """Build the 4 frozen 64px evaluation architectures (adaptation fixed per arch)."""
    return {
        arch: classifier(arch, num_classes, image_size) for arch in EVALUATION_ARCHITECTURES
    }


def smoke_64px_shape_check(batch_size: int = 2) -> dict[str, Any]:
    """Schema/tensor-shape level readiness check (random tensors; seconds-level; CPU ok).

    Returns output shapes per architecture. This is NOT a utility measurement and the
    record must never feed any inference (smoke_*_recorded_schema_only discipline).
    """
    torch.manual_seed(0)
    x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    record: dict[str, Any] = {
        "smoke_64px_shape_recorded_schema_only": True,
        "input_shape": list(x.shape),
        "architectures": {},
    }
    with torch.no_grad():
        for arch, model in build_evaluation_architectures().items():
            out = model(x)
            record["architectures"][arch] = {"output_shape": list(out.shape)}
    return record
