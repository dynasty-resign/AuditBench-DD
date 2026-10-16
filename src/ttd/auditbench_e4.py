"""AuditBench-DD E4 module (ImageWoof; iter-0007 execution-round authoring).

Code anchor of runs/imagewoof/auditbench_e4/e4_preregistration.json
(sha16 08db9ca9fa63067c, frozen iter-0006; zero-byte-change discipline).

Provides:

- the 64px evaluation-architecture factory (the 4 frozen architectures via
  models.classifier at image_size=64; ConvNetD4-style source architectures are
  provenance-only and are NEVER built here);
- the expected on-disk data paths for the real dataset (path constants only);
- the MECHANICAL cohort-fill rule (two-phase lock: the fill runs before any
  utility observation; the filled list is sha256-registered at fill time;
  post-observation rule changes are protocol deviations);
- a schema/tensor-shape level smoke check (random tensors, CPU-capable).

The z-score constants for imagewoof are empirically pinned on the official
train split at 64px BEFORE any formal E4 utility (E0/E2 lesson; see the
pre-registration input_normalization.zscore_pin_plan); inherited constants are
never trusted silently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .data import DATASET_STATS, dataset_spec
from .models import classifier

DATASET_NAME = "imagewoof"
NUM_CLASSES, IMAGE_SIZE = dataset_spec(DATASET_NAME)

REPO = Path(__file__).resolve().parents[2]
PREREGISTRATION_PATH = (
    REPO / "runs" / "imagewoof" / "auditbench_e4" / "e4_preregistration.json"
)

EVALUATION_ARCHITECTURES = ("convnet", "resnet18", "mobilenet_v2", "vgg11")

# Frozen stratification (e4_preregistration cohort_rules.excluded_strata).
STRATUM_PRIMARY = "primary"
STRATUM_EXTREME = "extreme_compression_ipc1"
STRATUM_QUARANTINE = "soft_label_quarantined"
STRATUM_GAP = "source_gaps"

# Frozen a-priori-registered family label for the DataDAM lineage
# (e4_preregistration cohort_rules.inclusion_rule; cross-dataset note: the
# E1/E2 manifests label the same lineage "attention_family" - registered for
# the E4b analysis round, never silently unified here).
DATADAM_FAMILY = "attention_matching_family"
# Lineage slug assigned at fill time (Q4 adjudication): identical spelling to
# the E1/E2 manifests so POOLED4 lineage clusters stay consistent; the venue
# field carries the hypothesis tag per the iter-0007 task (B3 stays closed).
DATADAM_LINEAGE = "datadam_sajedi2023_iccv"
DATADAM_VENUE_TAG = "hypothesis"

# In-house random baseline derivation rule (frozen before any utility):
# per class, a seed-0 permutation of that class's official train images,
# first `ipc` images; entries at ipc in RANDOM_BASELINE_IPCS.
RANDOM_BASELINE_SEED = 0
RANDOM_BASELINE_IPCS = (10, 50)
RANDOM_LINEAGE = "inhouse_random_baseline_iter0007e4a"

# z-score empirical pin (E0/E2 lesson): measured on the official train split
# at 64px in iter-0007 T3 before any E4 utility; evidence
# runs/evidence_packs/iter0007_e4a/zscore_pin/zscore_pin.json.
ZSCORE_PIN_REQUIRED = True


def load_preregistration(path: str | Path | None = None) -> dict[str, Any]:
    """Load the frozen E4 pre-registration JSON (read-only)."""
    prereg_path = Path(path) if path is not None else PREREGISTRATION_PATH
    return json.loads(prereg_path.read_text(encoding="utf-8"))


def expected_data_paths(root: str | Path) -> dict[str, Path]:
    """Expected real-dataset layout (constants only; never reads or downloads)."""
    base = Path(root) / "imagewoof2"
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

    NOT a utility measurement; the record must never feed any inference.
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


def datadam_entry(
    ipc: int,
    structure: dict[str, Any],
    sha256_pin: str,
) -> dict[str, Any]:
    """Mechanical entry construction for a verified DataDAM artifact.

    ``structure`` is the pre-utility structure-inspection record (schema only).
    Hard-label ipc>=10 artifacts route to the primary stratum; ipc==1 routes to
    the extreme-compression descriptive stratum (frozen excluded_strata);
    soft-label artifacts route to the quarantine stratum. Artifacts whose image
    shape is incompatible with the frozen 64px evaluation resolution route to
    the quarantine stratum (bad_shape; the frozen canonicalization rule in
    ttd.external_bank.normalize_payload, applied mechanically).
    """
    label_type = structure.get("label_type")
    shape = tuple(structure.get("image_shape", [])[1:])
    shape_ok = shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    reasons: list[str] = []
    if label_type != "hard":
        stratum = STRATUM_QUARANTINE
        reasons.append(f"label_type={label_type} (soft-label quarantine)")
    elif not shape_ok:
        stratum = STRATUM_QUARANTINE
        reasons.append(
            f"bad_shape: image shape {structure.get('image_shape')} incompatible with the "
            f"frozen {IMAGE_SIZE}px evaluation resolution (canonicalization bad_shape rule)"
        )
    else:
        stratum = STRATUM_EXTREME if ipc == 1 else STRATUM_PRIMARY
    return {
        "entry_id": f"datadam.datadam.ipc{ipc}",
        "source_id": "datadam.datadam",
        "release_lineage": DATADAM_LINEAGE,
        "venue": f"iccv2023 ({DATADAM_VENUE_TAG} tag retained per iter-0007 task Q4; "
                 "B3 retrieval closure stays in force)",
        "method": "DataDAM",
        "method_family": DATADAM_FAMILY,
        "distribution": "datadam_official",
        "tier": "tier-2 (author official HF release)",
        "ipc": ipc,
        "stratum": stratum,
        "stratum_reasons": reasons,
        "sha256_pin": sha256_pin,
        "fill_basis": ("e4_preregistration cohort_rules: hard-label public-release artifacts; "
                       "ipc1 routes to extreme_compression_ipc1; soft labels to "
                       "soft_label_quarantined; frozen-resolution-incompatible shapes "
                       "quarantine via the canonicalization bad_shape rule"),
    }


def random_entry(ipc: int) -> dict[str, Any]:
    """Mechanical entry construction for the in-house random baseline."""
    return {
        "entry_id": f"inhouse.random.ipc{ipc}",
        "source_id": "inhouse.random",
        "release_lineage": RANDOM_LINEAGE,
        "venue": "in-house (derivation baseline; not a publication release)",
        "method": "random",
        "method_family": "selection_family",
        "distribution": "in_house",
        "tier": "in-house baseline (homologous to E1/E2/E3 random entries)",
        "ipc": ipc,
        "stratum": STRATUM_PRIMARY,
        "sha256_pin": None,
        "fill_basis": ("e4_preregistration candidate enumeration "
                       "in_house_random_baseline (derivable_post_download); "
                       f"derivation rule: per-class seed-{RANDOM_BASELINE_SEED} permutation, "
                       "first ipc images"),
    }


def source_gap(source: str, note: str) -> dict[str, Any]:
    """Register a not-enumerated source (never written as does-not-exist)."""
    return {
        "source": source,
        "status": "not_enumerated",
        "note": note,
        "fill_basis": "e4_preregistration cohort_rules.excluded_strata.source_gaps",
    }


def fill_cohort(
    datadam_ipc10: dict[str, Any] | None,
    datadam_ipc1: dict[str, Any] | None,
    dcbench_note: str,
    mtt_note: str,
) -> dict[str, Any]:
    """MECHANICAL two-phase-lock fill under the frozen cohort rules.

    Inputs are the pre-utility structure-inspection records (or None when the
    artifact was not downloaded). The output queue is sorted by entry_id; no
    free discretion exists in this function.
    """
    entries: list[dict[str, Any]] = []
    extreme: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    gaps = [source_gap("dcbench_imagewoof", dcbench_note),
            source_gap("mtt_official_imagewoof", mtt_note)]

    for ipc, record in ((10, datadam_ipc10), (1, datadam_ipc1)):
        if record is None:
            gaps.append(source_gap(f"datadam_imagewoof_ipc{ipc}",
                                   "not downloaded (authorization-chain gate)"))
            continue
        entry = datadam_entry(ipc, record["structure"], record["sha256_pin"])
        if entry["stratum"] == STRATUM_PRIMARY:
            entries.append(entry)
        elif entry["stratum"] == STRATUM_EXTREME:
            extreme.append(entry)
        else:
            quarantine.append(entry)

    for ipc in RANDOM_BASELINE_IPCS:
        entries.append(random_entry(ipc))

    entries.sort(key=lambda e: e["entry_id"])
    return {
        "primary_entries": entries,
        "extreme_compression_entries": extreme,
        "quarantine_entries": quarantine,
        "source_gaps": gaps,
        "queue_order": [e["entry_id"] for e in entries],
    }


def registered_zscore() -> dict[str, Any]:
    """Return the pinned imagewoof z-score constants (post-pin; read-only)."""
    mean, std = DATASET_STATS[DATASET_NAME]
    return {"mean": list(mean), "std": list(std)}