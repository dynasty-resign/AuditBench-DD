"""AuditBench-DD E2a CIFAR-100 regression tests.

Covers the minimal backward-compatible additions made for the CIFAR-100
bank build: dataset stats/spec, per-dataset z-score inversion constants and
quarantine guard, AuditBenchE0Config cifar100 validation, structure-only
blob mapping at num_classes=100, zscore_entry dataset parameterization,
bank_index schema isomorphism with the CIFAR-10 bank, and the frozen E2
pre-registration cohort consistency.
"""
import inspect
import json
from pathlib import Path

import pytest
import torch

from ttd.auditbench_e1 import run_smoke, zscore_entry
from ttd.blobmap import inspect_blob, propose_file_map
from ttd.config import AuditBenchE0Config
from ttd.data import DATASET_STATS, dataset_spec
from ttd.external_bank import (
    CIFAR10_ZSCORE_MEAN,
    CIFAR10_ZSCORE_STD,
    ZSCORE_INVERSION_CONSTANTS,
    _normalize_images,
)
from ttd.provenance import ProvenanceRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
BANK100 = REPO_ROOT / "external" / "cifar100" / "bank_index.json"
BANK10 = REPO_ROOT / "external" / "cifar10" / "bank_index.json"
PREREG100 = REPO_ROOT / "runs" / "cifar100" / "auditbench_e2" / "e2_preregistration.json"

EXPECTED_C100_MEAN = (0.5071, 0.4867, 0.4408)
EXPECTED_C100_STD = (0.2675, 0.2565, 0.2761)
# Verified 2026-09-06 (iter-0005) by exact uint8 inversion of the DC-BENCH
# CIFAR-100 random blobs against the official train set (6100/6100 matches).
EXPECTED_C100_INV_MEAN = (0.5071, 0.4866, 0.4409)
EXPECTED_C100_INV_STD = (0.2673, 0.2564, 0.2762)


def test_dataset_stats_cifar100_pinned():
    mean, std = DATASET_STATS["cifar100"]
    assert tuple(mean) == EXPECTED_C100_MEAN
    assert tuple(std) == EXPECTED_C100_STD


def test_dataset_spec_cifar100():
    assert dataset_spec("cifar100") == (100, 32)


def test_inversion_constants_registry():
    assert ZSCORE_INVERSION_CONSTANTS["cifar10"] == (CIFAR10_ZSCORE_MEAN, CIFAR10_ZSCORE_STD)
    inv_mean, inv_std = ZSCORE_INVERSION_CONSTANTS["cifar100"]
    assert inv_mean == EXPECTED_C100_INV_MEAN
    assert inv_std == EXPECTED_C100_INV_STD


def test_zscore_inversion_roundtrip_cifar100():
    images = torch.rand(8, 3, 32, 32)
    mean = torch.tensor(EXPECTED_C100_INV_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(EXPECTED_C100_INV_STD).view(1, 3, 1, 1)
    stored = (images - mean) / std
    out, detected, reasons = _normalize_images(stored, "zscore", "cifar100")
    assert reasons == []
    assert torch.allclose(out, images, atol=1e-6)
    assert detected[0] == pytest.approx(float(stored.min()))
    assert detected[1] == pytest.approx(float(stored.max()))


def test_zscore_inversion_unknown_dataset_quarantines():
    images = torch.rand(4, 3, 32, 32)
    _, _, reasons = _normalize_images(images, "zscore", "mnist")
    assert any(r.split(":")[0] == "suspect_preprocessing" for r in reasons)


def test_zscore_entry_cifar100_transform():
    images = torch.rand(4, 3, 32, 32)
    out = zscore_entry(images, "cifar100")
    mean = torch.tensor(EXPECTED_C100_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(EXPECTED_C100_STD).view(1, 3, 1, 1)
    assert torch.allclose(out, (images - mean) / std, atol=1e-6)


def test_e0_config_accepts_cifar100():
    cfg = AuditBenchE0Config(dataset="cifar100", num_classes=100)
    cfg.validate()
    assert cfg.stage == "auditbench_e0"  # default stage name unchanged


def test_e0_config_defaults_unchanged():
    cfg = AuditBenchE0Config()
    cfg.validate()
    assert cfg.dataset == "cifar10"
    assert cfg.num_classes == 10
    assert cfg.gate_thresholds["pass_strong"]["min_primary_entries"] == 12


def test_e0_config_rejects_other_datasets():
    cfg = AuditBenchE0Config(dataset="tiny_imagenet", num_classes=200)
    with pytest.raises(ValueError):
        cfg.validate()


def _record(source_id: str, ipcs: list[int]) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=source_id,
        release_lineage="unit_lineage",
        method="UNIT",
        method_family="gradient_family",
        distribution="unit",
        dataset="cifar100",
        ipcs=ipcs,
        source_tier=3,
        official_urls=["https://example.org"],
    )


def test_inspect_blob_num_classes_100(tmp_path):
    images = torch.randn(100, 3, 32, 32)
    labels = torch.arange(100)
    path = tmp_path / "blob.pt"
    torch.save({"data": [[images, labels]]}, path)
    info = inspect_blob(path, sha256="x", num_classes=100)
    assert info.kind == "checkpoint_list"
    assert info.n_images == 100
    assert info.label_type == "hard"
    assert len(info.per_class_counts) == 100
    assert all(v == 1 for v in info.per_class_counts.values())


def test_propose_file_map_num_classes_100(tmp_path):
    images = torch.randn(100, 3, 32, 32)
    labels = torch.arange(100)
    img_path = tmp_path / "images.pt"
    lab_path = tmp_path / "labels.pt"
    torch.save(images, img_path)
    torch.save(labels, lab_path)
    blobs = [
        inspect_blob(img_path, sha256="a", num_classes=100),
        inspect_blob(lab_path, sha256="b", num_classes=100),
    ]
    for blob, name in zip(blobs, ["images.pt", "labels.pt"]):
        blob.relpath = name
    proposal = propose_file_map(_record("unit.pair", [1]), blobs, {}, num_classes=100)
    assert len(proposal["files"]) == 1
    spec = proposal["files"][0]
    assert spec["ipc"] == 1
    assert {p["role"] for p in spec["parts"]} == {"images", "labels"}
    assert proposal["unmapped"] == []


def test_run_smoke_stage_parameter():
    signature = inspect.signature(run_smoke)
    assert "stage" in signature.parameters
    assert signature.parameters["stage"].default == "auditbench_e1_smoke"


@pytest.mark.skipif(not (BANK100.exists() and BANK10.exists()), reason="bank indexes not built")
def test_bank_index_schema_isomorphic_to_cifar10():
    c100 = json.loads(BANK100.read_text(encoding="utf-8"))
    c10 = json.loads(BANK10.read_text(encoding="utf-8"))
    assert set(c100) == set(c10)
    keys100 = set().union(*(set(e) for e in c100["entries"]))
    keys10 = set().union(*(set(e) for e in c10["entries"]))
    assert keys100 == keys10
    assert c100["dataset"] == "cifar100"
    assert c100["num_classes"] == 100
    primary = [e for e in c100["entries"] if e["stratum"] == "primary"]
    assert all(e["label_type_observed"] == "hard" for e in primary)
    assert all(e["hash_checks"] and all(c["hash_verified"] for c in e["hash_checks"]) for e in primary)


@pytest.mark.skipif(not (PREREG100.exists() and BANK100.exists()), reason="E2 pre-registration not on disk")
def test_e2_preregistration_cohort_matches_bank():
    prereg = json.loads(PREREG100.read_text(encoding="utf-8"))
    bank = json.loads(BANK100.read_text(encoding="utf-8"))
    bank_primary = sorted(e["entry_id"] for e in bank["entries"] if e["stratum"] == "primary")
    assert prereg["cohort"]["entry_ids"] == bank_primary
    assert prereg["e2a_gate"]["decision"] == "PASS"
    assert prereg["training_budget"]["epochs"] == 200
    assert prereg["arms"]["base"] == ["ridge_proxy", "knn_coverage", "class_moment_dist"]
    assert prereg["smoke"]["smoke_only_flag"] is True
    assert len(prereg["smoke"]["artifacts"]) <= 2
