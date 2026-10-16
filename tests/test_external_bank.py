import torch

from ttd.external_bank import build_bank, detect_layout, normalize_blob
from ttd.provenance import ProvenanceRecord


def _record(source_id="dist.m", preprocessing="raw", tier=2, **overrides):
    base = {
        "source_id": source_id,
        "release_lineage": f"lin_{source_id}",
        "method": "M",
        "method_family": "gradient_family",
        "distribution": "dist",
        "dataset": "cifar10",
        "ipcs": [1, 10],
        "source_tier": tier,
        "official_urls": ["https://example.org/repo"],
        "preprocessing_expected": preprocessing,
    }
    base.update(overrides)
    return ProvenanceRecord(**base)


def _images(n=20, seed=0):
    return torch.rand(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def _labels(n=20):
    return torch.arange(n, dtype=torch.long) % 10


def test_detect_layouts():
    assert detect_layout({"images": _images(), "labels": _labels()}) == "canonical"
    assert detect_layout({"data": [[_images(), _labels()]]}) == "checkpoint_list"
    assert detect_layout([_images(), _labels()]) == "pair_list"
    assert detect_layout({"weird": 1}) == "unknown"


def test_canonical_normalization_primary(tmp_path):
    blob = tmp_path / "a.pt"
    torch.save({"images": _images(), "labels": _labels()}, blob)
    artifact, payload = normalize_blob(_record(), blob, ipc=10)
    assert artifact.stratum == "primary"
    assert artifact.raw_layout == "canonical"
    assert artifact.label_type_observed == "hard"
    assert payload is not None
    assert payload["images"].dtype == torch.float32
    assert 0.0 <= float(payload["images"].min())
    assert float(payload["images"].max()) <= 1.0
    assert "count_mismatch" not in artifact.quarantine_reasons


def test_checkpoint_list_uses_last_not_first(tmp_path):
    first = torch.zeros(20, 3, 32, 32)
    last = torch.ones(20, 3, 32, 32)
    blob = tmp_path / "ckpt.pt"
    torch.save({"data": [[first, _labels()], [last, _labels()]]}, blob)
    artifact, payload = normalize_blob(_record(), blob, ipc=10)
    assert artifact.raw_layout == "checkpoint_list"
    assert artifact.checkpoint_used == "last_of_2"
    assert float(payload["images"].mean()) == 1.0


def test_pair_list_layout(tmp_path):
    blob = tmp_path / "pair.pt"
    torch.save([_images(), _labels()], blob)
    artifact, _payload = normalize_blob(_record(), blob, ipc=10)
    assert artifact.raw_layout == "pair_list"
    assert artifact.stratum == "primary"


def test_uint8_images_rescaled(tmp_path):
    blob = tmp_path / "u8.pt"
    torch.save({"images": (_images() * 255).to(torch.uint8), "labels": _labels()}, blob)
    artifact, payload = normalize_blob(_record(), blob, ipc=10)
    assert artifact.stratum == "primary"
    assert float(payload["images"].max()) <= 1.0


def test_soft_labels_quarantined_never_argmaxed(tmp_path):
    blob = tmp_path / "soft.pt"
    soft = torch.zeros(20, 10)
    soft[torch.arange(20), _labels()] = 0.7
    torch.save({"images": _images(), "labels": soft}, blob)
    artifact, payload = normalize_blob(_record(), blob, ipc=10)
    assert artifact.label_type_observed == "soft"
    assert artifact.stratum == "soft_label_dependent"
    assert payload is None


def test_suspect_preprocessing_quarantined(tmp_path):
    blob = tmp_path / "neg.pt"
    torch.save({"images": _images() * 2 - 1, "labels": _labels()}, blob)
    artifact, _payload = normalize_blob(_record(), blob, ipc=10)
    assert "suspect_preprocessing" in artifact.quarantine_reasons
    assert artifact.stratum == "quarantine"


def test_zca_declared_goes_to_zca_stratum(tmp_path):
    blob = tmp_path / "zca.pt"
    torch.save({"images": _images(), "labels": _labels()}, blob)
    artifact, _ = normalize_blob(_record(preprocessing="zca"), blob, ipc=10)
    assert artifact.stratum == "zca_only"


def test_ipc1_is_extreme_stratum(tmp_path):
    blob = tmp_path / "ipc1.pt"
    torch.save({"images": _images(10), "labels": torch.arange(10)}, blob)
    artifact, _ = normalize_blob(_record(), blob, ipc=1)
    assert artifact.stratum == "extreme_compression"


def test_bad_shape_quarantined(tmp_path):
    blob = tmp_path / "bad.pt"
    torch.save({"images": torch.rand(20, 3, 64, 64), "labels": _labels()}, blob)
    artifact, _ = normalize_blob(_record(), blob, ipc=10, image_size=32)
    assert "bad_shape" in artifact.quarantine_reasons
    assert artifact.stratum == "quarantine"


def test_build_bank_dedup_and_index(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "dist.a").mkdir(parents=True)
    (raw_dir / "dist.b").mkdir(parents=True)
    same_images = _images(seed=7)
    torch.save({"images": same_images, "labels": _labels()}, raw_dir / "dist.a" / "x.pt")
    torch.save({"images": same_images.clone(), "labels": _labels()}, raw_dir / "dist.b" / "x.pt")
    records = [
        _record("dist.a", files=[{"relpath": "x.pt", "ipc": 10}]),
        _record("dist.b", files=[{"relpath": "x.pt", "ipc": 10}]),
    ]
    index = build_bank(records, raw_dir, tmp_path / "out")
    primary = [e for e in index["entries"] if e["stratum"] == "primary"]
    assert len(primary) == 1
    assert primary[0]["source_id"] == "dist.a"
    assert len(index["duplicates"]) == 1
    assert index["duplicates"][0]["dropped"] == "dist.b.ipc10"
    banked = list((tmp_path / "out" / "bank").glob("*.pt"))
    assert len(banked) == 1
    saved = torch.load(banked[0], map_location="cpu", weights_only=False)
    assert saved["meta"]["config"]["dataset"]["name"] == "cifar10"
    assert (tmp_path / "out" / "bank_index.json").exists()


def test_build_bank_missing_blob_quarantined(tmp_path):
    records = [_record("dist.missing", files=[{"relpath": "nope.pt", "ipc": 10}])]
    index = build_bank(records, tmp_path / "raw", tmp_path / "out")
    entry = index["entries"][0]
    assert entry["stratum"] == "quarantine"
    assert "missing_blob" in entry["quarantine_reasons"]