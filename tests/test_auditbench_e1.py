import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ttd import auditbench_e1
from ttd.auditbench_e1 import (
    E1_SCORE_RECORD_KEYS,
    SMOKE_ENTRY_IDS,
    bank_entry_meta,
    eval_ready_payload,
    load_bank_artifact,
    run_smoke,
    score_candidate,
    validate_evaluation_schema,
    validate_score_record,
    write_eval_artifact,
    zscore_entry,
)
from ttd.config import ExperimentConfig
from ttd.data import DATASET_STATS

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = REPO_ROOT / "runs" / "cifar10" / "auditbench_e1" / "e1_preregistration.json"


def _fake_cfg(tmp_path) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment="e1_test",
        seed=0,
        device="cpu",
        output_dir=str(tmp_path / "runs"),
    )
    cfg.dataset.name = "fake"
    cfg.dataset.image_size = 16
    cfg.dataset.fake_train_size = 60
    cfg.dataset.fake_test_size = 40
    cfg.dataset.num_workers = 0
    cfg.trajectory.timesteps = [4, 2, 0]
    cfg.backbone.embed_dim = 8
    cfg.evaluation.architectures = ["convnet"]
    cfg.evaluation.seeds = [0]
    cfg.evaluation.epochs = 1
    cfg.evaluation.batch_size = 20
    cfg.evaluation.lr = 0.01
    cfg.feasibility.topology_curve_bins = 8
    cfg.validate()
    return cfg


def _fake_bank(tmp_path, entry_ids=("unit.alpha", "unit.beta")) -> Path:
    bank_root = tmp_path / "bank_root"
    (bank_root / "bank").mkdir(parents=True)
    entries = []
    for index, entry_id in enumerate(entry_ids):
        generator = torch.Generator().manual_seed(index)
        images = torch.rand(20, 3, 16, 16, generator=generator)
        labels = torch.arange(10).repeat(2)
        meta = {
            "entry_id": entry_id,
            "ipc": 2,
            "release_lineage": f"lineage_{index}",
            "method_family": "unit_family",
            "stratum": "primary",
        }
        artifact_path = bank_root / "bank" / f"{entry_id}.pt"
        torch.save({"images": images, "labels": labels, "meta": meta}, artifact_path)
        entries.append(
            {
                "entry_id": entry_id,
                "source_id": entry_id,
                "release_lineage": f"lineage_{index}",
                "method": "Unit",
                "method_family": "unit_family",
                "dataset": "fake",
                "ipc": 2,
                "n_images": 20,
                "label_type_observed": "hard",
                "stratum": "primary",
                "bank_path": str(artifact_path),
            }
        )
    index = {"dataset": "fake", "num_classes": 10, "image_size": 16, "entries": entries}
    (bank_root / "bank_index.json").write_text(json.dumps(index), encoding="utf-8")
    return bank_root


def test_zscore_entry_matches_dataset_stats():
    images = torch.rand(5, 3, 8, 8)
    out = zscore_entry(images, "cifar10")
    mean, std = DATASET_STATS["cifar10"]
    expected = (images - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[
        None, :, None, None
    ]
    assert torch.allclose(out, expected, atol=1e-6)
    assert out.dtype == torch.float32


def test_zscore_entry_uniform_transform_for_both_input_paths():
    raw_unbounded = torch.randn(4, 3, 8, 8) * 5.0  # synthetic free-float
    inverted_unit = torch.rand(4, 3, 8, 8)  # zscore-inverted selection set
    out_a = zscore_entry(raw_unbounded)
    out_b = zscore_entry(inverted_unit)
    mean, std = DATASET_STATS["cifar10"]
    assert torch.allclose(out_a, (raw_unbounded - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[None, :, None, None])
    assert torch.allclose(out_b, (inverted_unit - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[None, :, None, None])


def test_load_bank_artifact_roundtrip(tmp_path):
    bank_root = _fake_bank(tmp_path)
    artifact = load_bank_artifact(bank_root, "unit.alpha")
    assert artifact["images"].shape == (20, 3, 16, 16)
    assert artifact["labels"].dtype == torch.int64
    assert artifact["meta"]["entry_id"] == "unit.alpha"
    meta = bank_entry_meta(bank_root, "unit.beta")
    assert meta["stratum"] == "primary"


def test_load_bank_artifact_unknown_entry_raises(tmp_path):
    bank_root = _fake_bank(tmp_path)
    with pytest.raises(KeyError):
        load_bank_artifact(bank_root, "unit.missing")
    with pytest.raises(FileNotFoundError):
        load_bank_artifact(tmp_path / "nowhere", "unit.alpha")


def test_eval_ready_payload_satisfies_evaluate_contract():
    meta = {"entry_id": "x", "ipc": 10, "provenance": {"tier": 2}}
    payload = eval_ready_payload(torch.zeros(2, 3, 4, 4), torch.tensor([0, 1]), meta, "cifar10", 32)
    dataset = payload["meta"]["config"]["dataset"]
    assert dataset == {"name": "cifar10", "image_size": 32}
    assert payload["meta"]["provenance"] == {"tier": 2}
    assert payload["labels"].dtype == torch.int64
    assert payload["images"].dtype == torch.float32


def _patched_embeddings(monkeypatch, dim=8, times=3):
    def fake_extract(cfg, images, labels, device):
        generator = torch.Generator().manual_seed(0)
        n = len(images)
        base = torch.randn(n, dim, generator=generator)
        states = torch.stack(
            [base * (i + 1) / times + 0.05 * torch.randn(n, dim, generator=generator) for i in range(times)],
            dim=1,
        )
        return states

    monkeypatch.setattr(auditbench_e1, "extract_synthetic_embeddings", fake_extract)


def test_score_candidate_schema_and_blinding(tmp_path, monkeypatch):
    cfg = _fake_cfg(tmp_path)
    _patched_embeddings(monkeypatch)
    rng = np.random.default_rng(0)
    real_states = rng.normal(size=(60, 3, 8))
    real_labels = np.repeat(np.arange(10), 6)
    images_z = zscore_entry(torch.rand(20, 3, 16, 16), "fake")
    labels = torch.arange(10).repeat(2)
    scores = score_candidate(
        cfg, images_z, labels, real_states, real_labels, torch.device("cpu"), resamples=2, seed=0
    )
    validate_score_record(scores)
    for key in E1_SCORE_RECORD_KEYS:
        assert key in scores
    assert isinstance(scores["runtime_sec"], float)
    assert set(scores["nulls"]) >= {
        "labelperm_class_moment_dist",
        "timeshuffle_topo_traj_dist",
        "cardinality_half_fid_endpoint",
    }


def test_validate_score_record_rejects_missing_key():
    with pytest.raises(ValueError):
        validate_score_record({"fid_endpoint": 1.0})


def test_run_smoke_end_to_end_fake(tmp_path, monkeypatch):
    cfg = _fake_cfg(tmp_path)
    _patched_embeddings(monkeypatch)
    monkeypatch.setattr(
        auditbench_e1,
        "real_reference_embeddings",
        lambda cfg, device, n_images=10000, seed=0: (
            np.random.default_rng(0).normal(size=(60, 3, 8)),
            np.repeat(np.arange(10), 6),
        ),
    )
    bank_root = _fake_bank(tmp_path)
    out_dir = tmp_path / "smoke"
    manifest_path = run_smoke(
        cfg, bank_root, out_dir, entry_ids=("unit.alpha", "unit.beta"), arch="convnet", seed=0
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "auditbench_e1_smoke"
    assert set(manifest["records"]) == {"unit.alpha", "unit.beta"}
    for record in manifest["records"].values():
        assert record["scores_schema_ok"]
        assert record["evaluation_schema_ok"]
        evaluation = validate_evaluation_schema(record["evaluation_path"])
        assert evaluation["runs"][0]["architecture"] == "convnet"
        assert evaluation["runs"][0]["seed"] == 0
        assert set(evaluation["runs"][0]) == {"architecture", "seed", "top1"}
        scores_payload = json.loads(Path(record["scores_path"]).read_text(encoding="utf-8"))
        assert scores_payload["smoke_only"] is True
        validate_score_record(scores_payload["scores"])


def test_smoke_defaults_match_frozen_preregistration():
    assert PREREG_PATH.exists(), "e1_preregistration.json must be frozen before smoke"
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    smoke = prereg["smoke"]
    assert tuple(smoke["artifacts"]) == SMOKE_ENTRY_IDS
    assert smoke["student_architectures"] == [auditbench_e1.SMOKE_ARCH]
    assert smoke["seeds"] == [auditbench_e1.SMOKE_SEED]
    assert len(smoke["artifacts"]) <= 2
    cohort_ids = set(prereg["cohort"]["entry_ids"])
    assert set(smoke["artifacts"]) <= cohort_ids


def test_write_eval_artifact_atomic(tmp_path):
    payload = {"images": torch.zeros(1, 3, 4, 4), "labels": torch.tensor([0]), "meta": {}}
    out = write_eval_artifact(payload, tmp_path / "a" / "x.pt")
    assert out.exists()
    loaded = torch.load(out, map_location="cpu", weights_only=False)
    assert loaded["labels"].item() == 0