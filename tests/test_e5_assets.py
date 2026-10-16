"""iter-0009 E5a tests: freeze ordering, two-phase lock timing, fill mechanicity,
z-score pin existence, append-only addendum discipline, zero-inference guard,
runtime shim purity.

Explicit new constants only; no existing assertion is modified. Remote-produced
artifacts are asserted when present (the round registers them before the
regression gate); the freeze/draft/config discipline tests are unconditional.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ITER9 = REPO / "scripts" / "iter0009"
sys.path.insert(0, str(ITER9))
sys.path.insert(0, str(REPO / "src"))

E5_PREREG = REPO / "runs/imagenette/auditbench_e5/e5_preregistration.json"
E5_DRAFT = REPO / "runs/imagenette/auditbench_e5/e5_preregistration_draft.json"
PACK = REPO / "runs/evidence_packs/iter0009_e5a"
E5_PREREG_SHA16_T1 = "8283d09ec4425f90"
E5_DRAFT_SHA256_ITER0008 = "30a4f06386aab53eb9a24debfb7a92a6eaaf9c96e0cf5f5f173a149d2c491987"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_e5_preregistration_frozen_and_sha16_registered():
    body = json.loads(E5_PREREG.read_text(encoding="utf-8"))
    assert body["status"] == "FROZEN"
    assert body["frozen_by"] == "codex iter-0009 (E5a execution round)"
    assert sha256_file(E5_PREREG)[:16] == E5_PREREG_SHA16_T1
    reg = json.loads((PACK / "closure_e5_preregistration_freeze.json").read_text(encoding="utf-8"))
    assert reg["frozen_sha16"] == E5_PREREG_SHA16_T1
    assert reg["draft_zero_byte_change_match"] is True


def test_draft_zero_byte_change():
    assert sha256_file(E5_DRAFT) == E5_DRAFT_SHA256_ITER0008


def test_freeze_precedes_fill_and_observation():
    freeze = json.loads((PACK / "closure_e5_preregistration_freeze.json").read_text(encoding="utf-8"))
    fillreg_path = PACK / "cohort_fill_registration.json"
    if not fillreg_path.exists():
        pytest.skip("fill not executed yet")
    fillreg = json.loads(fillreg_path.read_text(encoding="utf-8"))
    assert freeze["frozen_at_utc"] <= fillreg["registered_at_utc"]
    smoke_path = PACK / "smoke_calibration/smoke_calibration.json"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        first_obs = min(p["recorded_at_utc"] for p in smoke["probes"].values())
        assert fillreg["registered_at_utc"] <= first_obs


def test_fill_mechanicity_replay():
    replay_path = PACK / "fill_mechanicity_replay.json"
    if not replay_path.exists():
        pytest.skip("mechanicity replay not executed yet")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["cohort_byte_identical"] is True


def test_fill_queue_mechanical():
    fill_path = REPO / "runs/imagenette/auditbench_e5/e5_cohort_fill.json"
    if not fill_path.exists():
        pytest.skip("fill not executed yet")
    fill = json.loads(fill_path.read_text(encoding="utf-8"))
    queue = fill["cohort"]["queue_order"]
    assert queue == sorted(queue)
    assert fill["preregistration_body_match"] is True
    for gap in fill["cohort"]["source_gaps"]:
        assert gap["status"] == "not_enumerated"
        assert "does not exist" not in gap["note"].lower()
        assert "does-not-exist" not in gap["note"].lower()


def test_zscore_pin_schema():
    pin_path = PACK / "zscore_pin/zscore_pin.json"
    if not pin_path.exists():
        pytest.skip("z-score pin not executed yet")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    assert pin["accumulation_dtype"] == "float64"
    assert pin["n_images"] > 0
    assert len(pin["mean"]) == 3 and len(pin["std"]) == 3
    assert all(0.0 < m < 1.0 for m in pin["mean"])
    assert all(0.0 < s < 1.0 for s in pin["std"])
    assert pin["registered_inherited_constants"] is None


def test_addendum_append_only():
    addendum_path = REPO / "runs/imagenette/auditbench_e5/e5_preregistration_budget_addendum.json"
    if not addendum_path.exists():
        pytest.skip("addendum not written yet")
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    assert addendum["preregistration_body_match"] is True
    assert sha256_file(E5_PREREG)[:16] == E5_PREREG_SHA16_T1
    for arch, ev in addendum["coefficients_64px"].items():
        assert ev["b64_per_step_sec"]["evidence_class"] == "measured"
        assert ev["a64_overhead_sec"]["evidence_class"] == "measured"


def test_zero_inference_guard():
    report_path = REPO / "runs/imagenette/auditbench_e5/e5a_measurement_report.json"
    if not report_path.exists():
        pytest.skip("measurement report not written yet")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "inference_boundary" in report
    forbidden_keys = {"lofo", "loao", "bootstrap", "confidence_interval", "pooled_endpoint",
                      "pooled5", "spearman"}
    assert forbidden_keys.isdisjoint(k.lower() for k in report)


def test_config_validates():
    from ttd.config import load_config
    cfg = load_config(REPO / "configs/auditbench_e5_imagenette.yaml")
    assert cfg.dataset.name == "imagenette"
    assert cfg.dataset.image_size == 64
    assert cfg.evaluation.epochs == 200
    assert cfg.evaluation.seeds == [0, 1, 2]
    assert cfg.evaluation.architectures == ["convnet", "resnet18", "vgg11", "mobilenet_v2"]


def test_runtime_shim_registers_imagenette():
    import e5_runtime
    import ttd.data as data
    result = e5_runtime.register((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert result["dataset"] == "imagenette"
    assert data.dataset_spec("imagenette") == (10, 64)
    assert data.DATASET_STATS["imagenette"] == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert data.dataset_spec("cifar10") == (10, 32)
    mean, std = data.DATASET_STATS["cifar10"]
    assert mean[0] == pytest.approx(0.4914)


def test_banned_word_scan_zero_hits():
    scan_path = PACK / "banned_word_scan_report.json"
    if not scan_path.exists():
        pytest.skip("banned-word scan not executed yet")
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    assert scan["verdict"] == "ZERO_HITS"
