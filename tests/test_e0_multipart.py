"""Tests for multi-part file specs (separate images/labels blobs) in E0."""

import json

import torch
import yaml

from ttd.auditbench import run_auditbench_e0
from ttd.config import AuditBenchE0Config
from ttd.external_bank import assemble_parts, build_bank
from ttd.provenance import ProvenanceRecord, iter_spec_parts, validate_file_spec

TAXONOMY = ["selection_family"]


def _spec_single():
    return {"relpath": "a.pt", "ipc": 10, "url": "file:///x"}


def _spec_parts():
    return {
        "parts": [
            {"relpath": "i.pt", "role": "images", "url": "file:///i"},
            {"relpath": "l.pt", "role": "labels", "url": "file:///l"},
        ],
        "ipc": 10,
    }


def test_iter_spec_parts_normalizes_single_and_multi():
    assert iter_spec_parts(_spec_single()) == [
        {"relpath": "a.pt", "role": "blob", "url": "file:///x"}
    ]
    parts = iter_spec_parts(_spec_parts())
    assert [p["role"] for p in parts] == ["images", "labels"]
    assert [p["relpath"] for p in parts] == ["i.pt", "l.pt"]


def test_validate_file_spec_rules():
    assert validate_file_spec(_spec_single()) == []
    assert validate_file_spec(_spec_parts()) == []
    assert validate_file_spec({"ipc": 10})
    assert validate_file_spec({"ipc": 10, "parts": []})
    assert validate_file_spec({"ipc": 10, "parts": [{"relpath": "x", "role": "weird"}]})


def test_assemble_parts_pair_and_passthrough():
    images = torch.rand(10, 3, 32, 32)
    labels = torch.arange(10)
    payload = assemble_parts([("images", images), ("labels", labels)])
    assert isinstance(payload, list) and torch.equal(payload[0], images)
    single = assemble_parts([("blob", {"images": images, "labels": labels})])
    assert isinstance(single, dict)


def test_build_bank_with_multipart_spec(tmp_path):
    raw = tmp_path / "raw" / "src.test"
    raw.mkdir(parents=True)
    torch.save(torch.rand(100, 3, 32, 32), raw / "i.pt")
    torch.save(torch.arange(100) % 10, raw / "l.pt")
    record = ProvenanceRecord(
        source_id="src.test",
        release_lineage="lin_test",
        method="TEST",
        method_family="selection_family",
        distribution="test",
        dataset="cifar10",
        ipcs=[10],
        source_tier=2,
        official_urls=["https://example.org/repo"],
        files=[
            {
                "parts": [
                    {"relpath": "i.pt", "role": "images"},
                    {"relpath": "l.pt", "role": "labels"},
                ],
                "ipc": 10,
            }
        ],
        status="pending_fetch",
    )
    index = build_bank([record], tmp_path / "raw", tmp_path / "out")
    entry = index["entries"][0]
    assert entry["stratum"] == "primary"
    assert entry["n_images"] == 100
    assert len(entry["hash_checks"]) == 2
    assert all(c["hash_pinned_by"] == "first_fetch" for c in entry["hash_checks"])


def test_build_bank_multipart_missing_part(tmp_path):
    raw = tmp_path / "raw" / "src.test"
    raw.mkdir(parents=True)
    torch.save(torch.rand(100, 3, 32, 32), raw / "i.pt")
    record = ProvenanceRecord(
        source_id="src.test",
        release_lineage="lin_test",
        method="TEST",
        method_family="selection_family",
        distribution="test",
        dataset="cifar10",
        ipcs=[10],
        source_tier=2,
        official_urls=["https://example.org/repo"],
        files=[
            {
                "parts": [
                    {"relpath": "i.pt", "role": "images"},
                    {"relpath": "l.pt", "role": "labels"},
                ],
                "ipc": 10,
            }
        ],
        status="pending_fetch",
    )
    index = build_bank([record], tmp_path / "raw", tmp_path / "out")
    entry = index["entries"][0]
    assert entry["stratum"] == "quarantine"
    assert entry["quarantine_reasons"] == ["missing_blob"]
    assert entry["missing_relpaths"] == ["l.pt"]


def test_e0_pipeline_fetches_multipart_via_file_urls(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    torch.save(torch.rand(100, 3, 32, 32), staging / "i.pt")
    torch.save(torch.arange(100) % 10, staging / "l.pt")
    manifest = {
        "version": 1,
        "dataset": "cifar10",
        "taxonomy": TAXONOMY,
        "sources": [
            {
                "source_id": "official.pair",
                "release_lineage": "lin_pair",
                "method": "PAIR",
                "method_family": "selection_family",
                "distribution": "official",
                "dataset": "cifar10",
                "ipcs": [10],
                "source_tier": 2,
                "official_urls": ["https://example.org/repo"],
                "files": [
                    {
                        "parts": [
                            {"relpath": "i.pt", "role": "images", "url": f"file://{staging / 'i.pt'}"},
                            {"relpath": "l.pt", "role": "labels", "url": f"file://{staging / 'l.pt'}"},
                        ],
                        "ipc": 10,
                    }
                ],
                "status": "pending_fetch",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    cfg = AuditBenchE0Config(
        manifest_path=str(manifest_path),
        raw_dir=str(tmp_path / "raw"),
        out_dir=str(tmp_path / "bank"),
        report_path=str(tmp_path / "e0_report.json"),
        e1_protocol_path=str(tmp_path / "e1.json"),
        resolved_manifest_path=str(tmp_path / "resolved.yaml"),
        fetch_enabled=True,
        fetch_retries=1,
    )
    report = json.loads(run_auditbench_e0(cfg).read_text(encoding="utf-8"))
    outcomes = report["fetch"]["outcomes"]
    assert sorted(o["status"] for o in outcomes) == ["fetched", "fetched"]
    assert report["coverage"]["primary_entries"] == 1
