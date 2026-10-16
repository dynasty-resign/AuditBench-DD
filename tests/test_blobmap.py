"""Tests for structural blob-to-IPC mapping (ttd.blobmap).

The mapping must be decided by measured tensor structure only, never by
file or directory names; unmatched blobs stay unmapped with a reason.
"""

import torch

from ttd.blobmap import inspect_blob, match_ipc, propose_file_map
from ttd.provenance import ProvenanceRecord


def _record(ipcs=(1, 10, 50)):
    return ProvenanceRecord(
        source_id="src.test",
        release_lineage="lin_test",
        method="TEST",
        method_family="gradient_family",
        distribution="test",
        dataset="cifar10",
        ipcs=list(ipcs),
        source_tier=2,
        official_urls=["https://example.org/repo"],
        status="pending_fetch",
    )


def _save_combined(path, ipc, num_classes=10, image_size=32, seed=0):
    n = ipc * num_classes
    images = torch.rand(n, 3, image_size, image_size, generator=torch.Generator().manual_seed(seed))
    labels = torch.arange(n, dtype=torch.long) % num_classes
    torch.save({"images": images, "labels": labels}, path)


def test_inspect_combined_blob(tmp_path):
    blob = tmp_path / "a.pt"
    _save_combined(blob, 10)
    info = inspect_blob(blob, sha256="x")
    assert info.kind == "combined"
    assert info.n_images == 100
    assert info.image_shape == [100, 3, 32, 32]
    assert info.label_type == "hard"
    assert info.per_class_counts == {c: 10 for c in range(10)}


def test_inspect_checkpoint_list_uses_last(tmp_path):
    blob = tmp_path / "ckpt.pt"
    early = (torch.zeros(10, 3, 32, 32), torch.arange(10))
    final = (torch.rand(100, 3, 32, 32), torch.arange(100) % 10)
    torch.save({"data": [early, final], "accs": [0.1, 0.2]}, blob)
    info = inspect_blob(blob)
    assert info.kind == "checkpoint_list"
    assert info.n_checkpoints == 2
    assert info.n_images == 100


def test_inspect_images_and_labels_only(tmp_path):
    images = tmp_path / "i.pt"
    labels = tmp_path / "l.pt"
    torch.save(torch.rand(500, 3, 32, 32), images)
    torch.save(torch.arange(500) % 10, labels)
    info_i = inspect_blob(images)
    info_l = inspect_blob(labels)
    assert info_i.kind == "images_only" and info_i.n_images == 500
    assert info_l.kind == "labels_only" and info_l.n_labels == 500


def test_inspect_unrecognized_never_raises(tmp_path):
    blob = tmp_path / "junk.pt"
    blob.write_bytes(b"not a torch file")
    info = inspect_blob(blob)
    assert info.kind == "unrecognized"
    assert info.error


def test_match_ipc_uses_structure_only(tmp_path):
    blob = tmp_path / "deliberately_misleading_name_ipc50.pt"
    _save_combined(blob, 10)
    info = inspect_blob(blob)
    assert match_ipc(info, [1, 10, 50], 10, 32) == 10
    assert match_ipc(info, [1, 50], 10, 32) is None


def test_propose_file_map_pairs_and_singles(tmp_path):
    _save_combined(tmp_path / "c10.pt", 10)
    torch.save(torch.rand(500, 3, 32, 32), tmp_path / "i50.pt")
    torch.save(torch.arange(500) % 10, tmp_path / "l50.pt")
    torch.save(torch.rand(7, 3, 32, 32), tmp_path / "odd.pt")
    names = ("c10.pt", "i50.pt", "l50.pt", "odd.pt")
    blobs = [inspect_blob(tmp_path / p) for p in names]
    for b, p in zip(blobs, names):
        b.relpath = p
    ledger = {p: f"https://example.org/{p}" for p in ("c10.pt", "i50.pt", "l50.pt", "odd.pt")}
    proposal = propose_file_map(_record(), blobs, ledger)
    files = proposal["files"]
    by_ipc = {f["ipc"]: f for f in files}
    assert set(by_ipc) == {10, 50}
    assert by_ipc[10]["relpath"] == "c10.pt"
    assert by_ipc[10]["url"] == "https://example.org/c10.pt"
    roles = {p["role"]: p["relpath"] for p in by_ipc[50]["parts"]}
    assert roles == {"images": "i50.pt", "labels": "l50.pt"}
    assert any(u["relpath"] == "odd.pt" for u in proposal["unmapped"])


def test_propose_file_map_rejects_ambiguity(tmp_path):
    _save_combined(tmp_path / "a.pt", 10, seed=1)
    _save_combined(tmp_path / "b.pt", 10, seed=2)
    blobs = [inspect_blob(tmp_path / "a.pt"), inspect_blob(tmp_path / "b.pt")]
    blobs[0].relpath = "a.pt"
    blobs[1].relpath = "b.pt"
    proposal = propose_file_map(_record(), blobs, {})
    assert proposal["files"] == []
    assert len(proposal["unmapped"]) == 2
    assert all("ambiguous" in u["reason"] for u in proposal["unmapped"])


def test_propose_file_map_never_forces_unknown(tmp_path):
    _save_combined(tmp_path / "big.pt", 100)  # 1000 images, no declared ipc
    blobs = [inspect_blob(tmp_path / "big.pt")]
    blobs[0].relpath = "big.pt"
    proposal = propose_file_map(_record(), blobs, {})
    assert proposal["files"] == []
    assert proposal["unmapped"][0]["relpath"] == "big.pt"
