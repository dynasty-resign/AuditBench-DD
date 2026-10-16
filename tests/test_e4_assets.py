"""iter-0007 E4a execution-round tests (add-only): retrieval-evidence schema,
structure inspection, mechanical fill, z-score pin, append-only addendum,
matrix orchestrator unit-level logic, and measurement-report schema.

Execution-round discipline: no GPU, no network; all checks run on the round's
registered on-disk artifacts or synthetic fixtures.

New explicit constants (R0 adjudication family): EXPECTED_MEAN/STD (the
measured pin), EXPECTED_FILL_SHA16, EXPECTED_QUEUE, EXPECTED_TGZ_SHA256,
EXPECTED_DATADAM_PINS, EXPECTED_N_TRAIN/VAL.
This test file is sha256-registered in runs/evidence_packs/iter0007_e4a/
sha256_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "runs" / "evidence_packs" / "iter0007_e4a"
E4 = REPO / "runs" / "imagewoof" / "auditbench_e4"
sys.path.insert(0, str(REPO / "src"))

EXPECTED_MEAN = [0.486150, 0.455984, 0.393782]
EXPECTED_STD = [0.247363, 0.239938, 0.249480]
EXPECTED_FILL_SHA16 = "ced0673c5831916f"
EXPECTED_QUEUE = ["inhouse.random.ipc10", "inhouse.random.ipc50"]
EXPECTED_TGZ_SHA256 = "de3f58c4ea3e042cf3f8365fbc699288cfe1d8c151059040d181c221bd5a55b8"
EXPECTED_DATADAM_PIN_PREFIXES = {
    "datadam_ipc10": "9c8178fdf913a507",
    "datadam_ipc1": "35ddc3f9c606e77a",
}
EXPECTED_N_TRAIN = 9025
EXPECTED_N_VAL = 3929
EXPECTED_ZSCORE_SHA16 = "664a45b68ed2eb03"


def _j(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- module + dataset registration ----

def test_dataset_spec_and_stats_registered():
    from ttd.data import DATASET_STATS, dataset_spec
    assert dataset_spec("imagewoof") == (10, 64)
    mean, std = DATASET_STATS["imagewoof"]
    assert [round(v, 6) for v in mean] == EXPECTED_MEAN
    assert [round(v, 6) for v in std] == EXPECTED_STD


def test_e4_module_architectures_shape():
    from ttd.auditbench_e4 import smoke_64px_shape_check
    record = smoke_64px_shape_check(batch_size=2)
    assert record["smoke_64px_shape_recorded_schema_only"] is True
    assert sorted(record["architectures"]) == ["convnet", "mobilenet_v2", "resnet18", "vgg11"]
    for arch, row in record["architectures"].items():
        assert row["output_shape"] == [2, 10], arch


def test_config_validates():
    from ttd.config import load_config
    cfg = load_config(str(REPO / "configs" / "auditbench_e4_imagewoof.yaml"), [])
    assert cfg.dataset.name == "imagewoof"
    assert cfg.dataset.image_size == 64
    assert cfg.evaluation.epochs == 200
    assert list(cfg.evaluation.architectures) == ["convnet", "resnet18", "vgg11", "mobilenet_v2"]
    assert list(cfg.evaluation.seeds) == [0, 1, 2]


# ---- mechanical fill unit-level ----

def _struct(shape, label="hard"):
    return {"image_shape": [shape[0], 3, shape[1], shape[1]], "label_type": label,
            "n_classes": 10}


def test_fill_mechanics_quarantine_128px():
    from ttd.auditbench_e4 import datadam_entry
    entry = datadam_entry(10, _struct((100, 128)), "0" * 64)
    assert entry["stratum"] == "soft_label_quarantined"
    assert any("bad_shape" in r for r in entry["stratum_reasons"])


def test_fill_mechanics_soft_label_quarantine():
    from ttd.auditbench_e4 import datadam_entry
    entry = datadam_entry(10, _struct((100, 64), label="soft"), "0" * 64)
    assert entry["stratum"] == "soft_label_quarantined"


def test_fill_mechanics_primary_and_extreme():
    from ttd.auditbench_e4 import datadam_entry
    e10 = datadam_entry(10, _struct((100, 64)), "0" * 64)
    e1 = datadam_entry(1, _struct((10, 64)), "0" * 64)
    assert e10["stratum"] == "primary"
    assert e1["stratum"] == "extreme_compression_ipc1"
    assert e10["release_lineage"] == "datadam_sajedi2023_iccv"
    assert "hypothesis" in e10["venue"]


def test_fill_cohort_deterministic_sorted():
    from ttd.auditbench_e4 import fill_cohort
    rec10 = {"structure": _struct((100, 128)), "sha256_pin": "a" * 64}
    rec1 = {"structure": _struct((10, 128)), "sha256_pin": "b" * 64}
    out1 = fill_cohort(rec10, rec1, "gap-a", "gap-b")
    out2 = fill_cohort(rec10, rec1, "gap-a", "gap-b")
    assert out1 == out2
    assert out1["queue_order"] == sorted(out1["queue_order"])
    assert out1["queue_order"] == ["inhouse.random.ipc10", "inhouse.random.ipc50"]
    assert len(out1["quarantine_entries"]) == 2
    assert len(out1["source_gaps"]) == 2
    assert all(g["status"] == "not_enumerated" for g in out1["source_gaps"])


# ---- on-disk artifact schemas ----

def test_zscore_pin_artifact():
    pin = _j(PACK / "zscore_pin" / "zscore_pin.json")
    assert pin["n_images"] == EXPECTED_N_TRAIN
    assert pin["accumulation_dtype"] == "float64"
    assert pin["mean"] == EXPECTED_MEAN
    assert pin["std"] == EXPECTED_STD
    assert pin["constants_sha16"] == EXPECTED_ZSCORE_SHA16


def test_structure_inspection_artifact():
    rep = _j(PACK / "structure_inspection" / "structure_inspection.json")
    ipc10 = rep["artifacts"]["datadam_ipc10"]
    assert ipc10["image_shape"] == [100, 3, 128, 128]
    assert ipc10["label_type"] == "hard"
    assert ipc10["n_classes"] == 10
    ipc1 = rep["artifacts"]["datadam_ipc1"]
    assert ipc1["image_shape"] == [10, 3, 128, 128]
    assert rep["shard_probe"]["datadam_dir_listing"] == [
        "syn_data_imagewoof_ipc_1.pt", "syn_data_imagewoof_ipc_10.pt"]


def test_cohort_fill_registration_lock():
    reg = _j(PACK / "cohort_fill_registration.json")
    assert reg["fill_sha16"] == EXPECTED_FILL_SHA16
    assert reg["queue_order"] == EXPECTED_QUEUE
    assert reg["fill_sha256"] == _sha256(E4 / "e4_cohort_fill.json")
    assert reg["registered_at_utc"] < "2026-09-08T00:01:00", (
        "fill registration must precede the first utility observation (~00:01:31Z smoke)")


def test_structural_note_schema():
    note = _j(E4 / "e4_preregistration_structural_note_iter0007.json")
    assert note["kind"] == "append_only_structural_correction"
    assert note["correction"]["post_observation_lock"] == "NOT triggered: zero utility values had been observed when this correction was registered"
    assert note["correction"]["measured_entries"] == 2
    assert "128" in note["finding"]["measured_structure"][0]


def test_budget_addendum_append_only():
    add = _j(E4 / "e4_preregistration_budget_addendum.json")
    assert add["preregistration_body_match"] is True
    assert add["preregistration_body_sha16"] == "08db9ca9fa63067c"
    assert add["execution_form_verdict"] == "matrix_full"
    assert add["convergence_check"]["diverged_any"] is False
    assert add["steps_actual"] == {"10": 200, "50": 800}
    preg_sha = hashlib.sha256((E4 / "e4_preregistration.json").read_bytes()).hexdigest()[:16]
    assert preg_sha == "08db9ca9fa63067c"


def test_authorization_chain_artifacts():
    log = _j(PACK / "authorization_chain" / "t1_download_log.json")
    assert log["imagewoof2_tgz"]["sha256"] == EXPECTED_TGZ_SHA256
    assert log["train_images"] == EXPECTED_N_TRAIN
    assert log["val_images"] == EXPECTED_N_VAL
    assert log["train_class_dirs"] == 10
    assert (PACK / "authorization_chain" / "license_compliance.md").exists()
    assert (PACK / "authorization_chain" / "t1_abort_scene.json").exists()
    adj = _j(PACK / "license_chain" / "license_adjudication.json")
    assert adj["items"]["imagewoof_real_set"]["verdict"] == "CLOSED_WITH_CONSTRAINTS"
    assert adj["items"]["dcbench_imagewoof"]["verdict"] == "NOT_ENUMERATED"
    assert adj["items"]["mtt_official_imagewoof"]["verdict"] == "NOT_ENUMERATED"
    fetch = _j(PACK / "authorization_chain" / "t1_datadam_local_fetch.json")
    pins = {it["label"]: it["sha256"] for it in fetch["items"]}
    assert set(pins) == {"datadam_ipc10", "datadam_ipc1"}
    assert all(len(v) == 64 for v in pins.values())
    for label, prefix in EXPECTED_DATADAM_PIN_PREFIXES.items():
        assert pins[label].startswith(prefix), label


def test_retrieval_log_schema():
    log = _j(PACK / "license_chain" / "retrieval_log.json")
    ids = {r["evidence_id"] for r in log["records"]}
    assert {"imagenette_readme_raw", "imagenette_license_raw", "imagewoof2_tgz_head",
            "hf_datadam_card_api", "dcbench_readme_raw"} <= ids
    for r in log["records"]:
        assert "http_status" in r and "requested_at_utc" in r
        if r.get("body_sha256"):
            assert len(r["body_sha256"]) == 64
    sup = _j(PACK / "license_chain" / "retrieval_log_supplement.json")
    sup_ids = {r["evidence_id"] for r in sup["records"]}
    assert {"mtt_readme_raw", "mtt_license_raw", "datadam_github_api"} <= sup_ids


def test_smoke_calibration_artifact():
    smoke = _j(PACK / "smoke_calibration" / "smoke_calibration.json")
    assert smoke["probes"]["heaviest"]["architecture"] == "resnet18"
    assert smoke["probes"]["heaviest"]["entry_id"] == "inhouse.random.ipc50"
    assert smoke["probes"]["lightest"]["architecture"] == "convnet"
    assert smoke["probes"]["lightest"]["entry_id"] == "inhouse.random.ipc10"
    for label in ("heaviest", "lightest"):
        assert smoke["probes"][label]["loss_trajectory_finite"] is True
        assert smoke["probes"][label]["diverged"] is False


# ---- measurement report (post-matrix artifacts) ----

def test_measurement_report_schema():
    rep = _j(E4 / "e4a_measurement_report.json")
    assert rep["evidence_class"].startswith("measured")
    assert "screening" in rep["screening_statement"]
    comp = rep["matrix_completeness"]
    assert comp["cells_total"] == 24
    assert comp["cells_done"] == comp["cells_total"]
    assert comp["complete"] is True
    assert set(rep["proxy_coverage"]) == set(EXPECTED_QUEUE)
    assert all(v["status"] == "complete" for v in rep["proxy_coverage"].values())
    assert rep["screening_fill"]["n"] == 2
    assert rep["screening_fill"]["d"] == 1
    # two-entry queue partitions into single-entry (arch|ipc) cells: first tier
    # is honestly registered not-applicable; the frozen >=2/5 second tier is
    # still applied mechanically and fires degenerately by construction
    # (random-baseline-only measured cohort). Descriptive registration only.
    assert len(rep["near_random"]) == 8
    for nr in rep["near_random"].values():
        assert nr["status"] == "single_entry_first_tier_not_applicable"
        assert nr["n_entries"] == 1
        assert nr["near_random_descriptive_first_tier"] is None
        assert nr["near_random_descriptive_second_tier"] is True
        assert nr["second_tier_threshold"] == 0.4
        assert nr["collapse_clause_triggered"] is True
        assert set(nr["second_tier_g6_annotation"]) == {
            "descriptive", "distinct", "frozen", "LOFO", "LOAO", "endpoint"}
    md = (E4 / "e4a_measurement_report.md").read_text(encoding="utf-8")
    assert "screening" in md
    # inference-output patterns must not appear; the tokens Spearman/bootstrap
    # may only occur inside the mandatory negative boundary declaration and the
    # frozen screening template line
    forbidden_patterns = ["Spearman rho", "rho =", "rho=", "bootstrap CI:",
                          "95% CI", "CI [", "LOFO x LOAO pooled", "p <", "p="]
    for token in forbidden_patterns:
        assert token not in md
    for line in md.splitlines():
        if "Spearman" in line or "bootstrap" in line:
            assert "inference boundary" in line or "screening evidence" in line