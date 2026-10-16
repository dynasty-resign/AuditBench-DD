import torch

from ttd.provenance import ProvenanceRecord, sha256_file, verify_blob

TAXONOMY = ["gradient_family", "selection_family"]


def _record(**overrides):
    base = {
        "source_id": "dist.method",
        "release_lineage": "m_conf2024",
        "method": "M",
        "method_family": "gradient_family",
        "distribution": "dist",
        "dataset": "cifar10",
        "ipcs": [10, 50],
        "source_tier": 2,
        "official_urls": ["https://example.org/repo"],
    }
    base.update(overrides)
    return ProvenanceRecord(**base)


def test_valid_record_passes():
    assert _record().validate(TAXONOMY) == []


def test_unknown_family_rejected():
    problems = _record(method_family="mystery_family").validate(TAXONOMY)
    assert any("taxonomy" in p for p in problems)


def test_mirror_requires_pinned_hash():
    problems = _record(source_tier=5).validate(TAXONOMY)
    assert any("tier 5" in p for p in problems)
    ok = _record(source_tier=5, pinned_sha256={"blob.pt": "ab" * 32})
    assert ok.validate(TAXONOMY) == []


def test_reproduced_requires_label():
    problems = _record(source_tier=4, notes="regenerated data").validate(TAXONOMY)
    assert any("REPRODUCED" in p for p in problems)
    ok = _record(source_tier=4, notes="REPRODUCED at commit abc123")
    assert ok.validate(TAXONOMY) == []


def test_pending_url_verification_allows_empty_urls():
    record = _record(official_urls=[], status="pending_url_verification")
    assert record.validate(TAXONOMY) == []
    problems = _record(official_urls=[], status="pending_fetch").validate(TAXONOMY)
    assert any("official URL" in p for p in problems)


def test_unknown_manifest_fields_rejected():
    payload = _record().to_dict()
    payload["surprise"] = 1
    try:
        ProvenanceRecord.from_dict(payload)
    except ValueError as exc:
        assert "surprise" in str(exc)
    else:
        raise AssertionError("unknown field accepted")


def test_verify_blob_first_fetch_and_pinned(tmp_path):
    blob = tmp_path / "blob.pt"
    torch.save({"x": torch.arange(4)}, blob)
    record = _record()
    first = verify_blob(record, "blob.pt", blob)
    assert first["hash_verified"] and first["hash_pinned_by"] == "first_fetch"

    pinned = _record(pinned_sha256={"blob.pt": first["observed_sha256"]})
    again = verify_blob(pinned, "blob.pt", blob)
    assert again["hash_verified"] and again["hash_pinned_by"] == "manifest"

    wrong = _record(pinned_sha256={"blob.pt": "00" * 32})
    assert not verify_blob(wrong, "blob.pt", blob)["hash_verified"]

    mirror = _record(source_tier=5)
    assert not verify_blob(mirror, "blob.pt", blob)["hash_verified"]


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    blob = tmp_path / "x.bin"
    blob.write_bytes(b"auditbench")
    assert sha256_file(blob) == hashlib.sha256(b"auditbench").hexdigest()