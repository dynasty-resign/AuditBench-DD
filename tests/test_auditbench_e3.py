"""iter-0008 T3 tests: E3 module skeleton, config, pre-registration discipline.

All checks are design-round level: schema/tensor-shape only; zero utility; zero download.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ttd import auditbench_e3
from ttd.config import load_config

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "runs" / "tiny_imagenet" / "auditbench_e3" / "e3_preregistration.json"
A3_LISTING = REPO / "runs" / "evidence_packs" / "iter0007" / "third_dataset_scan" / "evidence" / "drive_dcbench_listing.json"

REQUIRED_SECTIONS = (
    "cohort", "tm_replica_rule", "student_architectures", "training_budget",
    "proxy_suite", "arms", "alpha_grid", "resampling", "primary_endpoint",
    "secondary_endpoints", "statistics", "budget_table", "venue_citation_policy",
    "license_gate", "input_normalization", "embedding_backbone", "smoke", "forbidden",
)


def test_preregistration_has_required_sections():
    preg = auditbench_e3.load_preregistration()
    for section in REQUIRED_SECTIONS:
        assert section in preg, section
    assert preg["cohort"]["n_design_entries"] == 10
    assert preg["cohort"]["n_measured_entries"] == 14
    assert len(preg["forbidden"]) == 6


def test_cohort_grounded_in_a3_listing():
    preg = auditbench_e3.load_preregistration()
    listing = {item["path"]: item["id"] for item in json.loads(A3_LISTING.read_text(encoding="utf-8"))}
    entries = auditbench_e3.cohort_entries(preg)
    assert [e["entry_id"] for e in entries] == list(auditbench_e3.PRIMARY_ENTRY_IDS)
    for entry in entries:
        paths = entry["a3_source"]["drive_paths"]
        assert paths, entry["entry_id"]
        for path, fid in paths.items():
            assert listing.get(path) == fid, (entry["entry_id"], path)


def test_tm_replica_expansion_single_cluster():
    preg = auditbench_e3.load_preregistration()
    ids = auditbench_e3.expand_measurement_entries(preg)
    assert len(ids) == 14
    for rid in auditbench_e3.TM_IPC50_REPLICA_IDS:
        assert rid in ids
    rule = preg["tm_replica_rule"]
    assert rule["replica_entry_ids"] == list(auditbench_e3.TM_IPC50_REPLICA_IDS)
    assert "mtt_cazenavette2022_cvpr" in rule["cluster"]
    assert "post_observation_lock" in rule


def test_dataset_spec_and_arch_factory_64px():
    assert (auditbench_e3.NUM_CLASSES, auditbench_e3.IMAGE_SIZE) == (200, 64)
    models = auditbench_e3.build_evaluation_architectures()
    assert set(models) == set(auditbench_e3.EVALUATION_ARCHITECTURES)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        for arch, model in models.items():
            out = model(x)
            assert list(out.shape) == [2, 200], arch


def test_smoke_shape_record_is_schema_only():
    record = auditbench_e3.smoke_64px_shape_check(batch_size=2)
    assert record["smoke_64px_shape_recorded_schema_only"] is True
    assert record["input_shape"] == [2, 3, 64, 64]
    for arch in auditbench_e3.EVALUATION_ARCHITECTURES:
        assert record["architectures"][arch]["output_shape"] == [2, 200]


def test_config_loads_and_matches_frozen_budget():
    cfg = load_config(REPO / "configs" / "auditbench_e3_tinyimagenet.yaml")
    assert cfg.dataset.name == "tiny_imagenet"
    assert cfg.dataset.image_size == 64
    assert cfg.dataset.download is False
    assert cfg.evaluation.epochs == 200
    assert cfg.evaluation.batch_size == 128
    assert cfg.evaluation.lr == 0.1
    assert cfg.evaluation.weight_decay == 0.0005
    assert cfg.evaluation.seeds == [0, 1, 2]
    assert set(cfg.evaluation.architectures) == set(auditbench_e3.EVALUATION_ARCHITECTURES)
    assert cfg.backbone.latent_channels == 16
    assert cfg.backbone.ae_epochs == 100
    assert cfg.backbone.diffusion_epochs == 200


def _iter0008_text_assets() -> list[Path]:
    return [
        PREREG,
        REPO / "src" / "ttd" / "auditbench_e3.py",
        REPO / "configs" / "auditbench_e3_tinyimagenet.yaml",
        REPO / "runs" / "evidence_packs" / "iter0008" / "license_check" / "license_check.md",
    ]


def test_no_banned_words_in_iter0008_text_assets():
    # Banned-claim-word scan, iter-0007 convention: mandated verbatim zones are exempt
    # (the screening template wording, the quoted forbidden-mirror section, the A3
    # artifact identifiers images_best/labels_best, and the regret definition identifier
    # best_utility). Everything else must be clean.
    for path in _iter0008_text_assets():
        text = path.read_text(encoding="utf-8").lower()
        if path.suffix == ".json":
            preg = json.loads(path.read_text(encoding="utf-8"))
            for key in ("forbidden",):
                preg.pop(key, None)
            preg.get("tm_replica_rule", {}).pop("verbatim", None)
            preg.get("venue_citation_policy", {}).pop("verbatim", None)
            text = json.dumps(preg, ensure_ascii=False).lower()
        text = text.replace("confirmatory multi-dataset follow-up", "")
        # E2-frozen screening wording ("E2 is SCREENING evidence, not confirmatory: ...")
        text = text.replace("screening evidence, not confirmatory", "")
        text = text.replace("best_utility", "")
        text = text.replace("images_best", "").replace("labels_best", "")
        text = text.replace("lr_best", "")  # DATM release artifact identifier (A3 metadata)
        for word in ("confirmatory", "significant", "universal", "best"):
            assert not re.search(rf"[a-z_]*{word}[a-z_]*", text), (path.name, word)
