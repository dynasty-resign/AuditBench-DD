import json

import torch
import yaml

from ttd.auditbench import decide_gate, freeze_e1_protocol, run_auditbench_e0
from ttd.config import AuditBenchE0Config

TAXONOMY = [
    "selection_family",
    "gradient_family",
    "distribution_family",
    "trajectory_family",
    "kernel_family",
    "parameterization_family",
]


def _fixture_blob(path, seed):
    images = torch.rand(20, 3, 32, 32, generator=torch.Generator().manual_seed(seed))
    labels = torch.arange(20, dtype=torch.long) % 10
    torch.save({"images": images, "labels": labels}, path)


def _write_manifest(tmp_path, lineages, dual_ipc=True):
    staging = tmp_path / "staging"
    staging.mkdir()
    sources = []
    seed = 0
    for index, (lineage, family) in enumerate(lineages):
        files = []
        for ipc in ([10, 50] if dual_ipc else [10]):
            seed += 1
            blob = staging / f"{lineage}_ipc{ipc}.pt"
            _fixture_blob(blob, seed)
            files.append({"relpath": f"ipc{ipc}.pt", "ipc": ipc, "url": f"file://{blob}"})
        sources.append(
            {
                "source_id": f"official.{lineage}",
                "release_lineage": lineage,
                "method": lineage.upper(),
                "method_family": family,
                "distribution": "official",
                "dataset": "cifar10",
                "ipcs": [10, 50] if dual_ipc else [10],
                "source_tier": 2,
                "official_urls": ["https://example.org/repo"],
                "files": files,
                "status": "pending_fetch",
            }
        )
    manifest = {"version": 1, "dataset": "cifar10", "taxonomy": TAXONOMY, "sources": sources}
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def _cfg(tmp_path, manifest):
    return AuditBenchE0Config(
        manifest_path=str(manifest),
        raw_dir=str(tmp_path / "raw"),
        out_dir=str(tmp_path / "bank"),
        report_path=str(tmp_path / "e0_report.json"),
        e1_protocol_path=str(tmp_path / "e1_protocol.json"),
        resolved_manifest_path=str(tmp_path / "resolved.yaml"),
        fetch_enabled=True,
        fetch_retries=1,
    )


def test_e0_pass_strong_with_full_staged_coverage(tmp_path):
    lineages = [(f"lin{i}", family) for i, family in enumerate(TAXONOMY)]
    cfg = _cfg(tmp_path, _write_manifest(tmp_path, lineages))
    report_path = run_auditbench_e0(cfg)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gate"]["gate"] == "PASS_STRONG"
    assert report["coverage"]["n_lineages"] == 6
    assert report["coverage"]["n_families"] == 6
    assert report["coverage"]["primary_entries"] == 12
    assert (tmp_path / "e1_protocol.json").exists()
    assert (tmp_path / "resolved.yaml").exists()


def test_e0_pass_minimum(tmp_path):
    lineages = [(f"lin{i}", family) for i, family in enumerate(TAXONOMY[:4])]
    cfg = _cfg(tmp_path, _write_manifest(tmp_path, lineages))
    report = json.loads(run_auditbench_e0(cfg).read_text(encoding="utf-8"))
    assert report["gate"]["gate"] == "PASS_MINIMUM"
    assert report["coverage"]["primary_entries"] == 8


def test_e0_human_review_when_coverage_thin(tmp_path):
    lineages = [(f"lin{i}", family) for i, family in enumerate(TAXONOMY[:2])]
    cfg = _cfg(tmp_path, _write_manifest(tmp_path, lineages, dual_ipc=False))
    report = json.loads(run_auditbench_e0(cfg).read_text(encoding="utf-8"))
    assert report["gate"]["gate"] == "HUMAN_REVIEW"


def test_e0_failed_when_nothing_fetchable(tmp_path):
    manifest = {
        "version": 1,
        "dataset": "cifar10",
        "taxonomy": TAXONOMY,
        "sources": [
            {
                "source_id": "official.gone",
                "release_lineage": "gone_2024",
                "method": "GONE",
                "method_family": "gradient_family",
                "distribution": "official",
                "dataset": "cifar10",
                "ipcs": [10],
                "source_tier": 2,
                "official_urls": ["http://127.0.0.1:9/unreachable.pt"],
                "files": [{"relpath": "x.pt", "ipc": 10}],
                "status": "pending_fetch",
            }
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    cfg = _cfg(tmp_path, path)
    cfg.fetch_timeout_sec = 1
    report = json.loads(run_auditbench_e0(cfg).read_text(encoding="utf-8"))
    assert report["gate"]["gate"] == "FAILED"
    assert report["fetch"]["outcomes"][0]["status"] == "fetch_failed"
    assert report["fetch"]["outcomes"][0]["error"]


def test_decide_gate_blocks_on_suspect_preprocessing():
    coverage = {
        "primary_entries": 12,
        "n_lineages": 6,
        "n_families": 6,
        "n_dual_ipc_lineages": 6,
        "lineages": {},
    }
    index = {
        "entries": [
            {
                "stratum": "primary",
                "bank_path": "x.pt",
                "quarantine_reasons": ["suspect_preprocessing"],
            }
        ]
    }
    cfg = AuditBenchE0Config()
    decision = decide_gate(coverage, index, [{"status": "fetched"}], cfg.gate_thresholds)
    assert decision["gate"] == "HUMAN_REVIEW"
    assert "suspect_preprocessing" in decision["blocking_quarantine"]


def test_freeze_e1_protocol_rules():
    protocol = freeze_e1_protocol(TAXONOMY, [10, 50], [1])
    assert protocol["primary_cohort"]["label_type"] == "hard"
    assert protocol["grouping"]["leave_one_family_out"] is True
    assert any("never appear in both train and test" in r for r in protocol["resampling"])
    assert any("reported accuracy" in r for r in protocol["blinding"])


def test_manifest_schema_problems_reported(tmp_path):
    manifest = {
        "version": 1,
        "dataset": "cifar10",
        "taxonomy": TAXONOMY,
        "sources": [
            {
                "source_id": "bad.entry",
                "release_lineage": "x",
                "method": "X",
                "method_family": "not_a_family",
                "distribution": "d",
                "dataset": "cifar10",
                "ipcs": [10],
                "source_tier": 9,
                "official_urls": [],
                "status": "pending_url_verification",
            }
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    cfg = _cfg(tmp_path, path)
    cfg.fetch_enabled = False
    report = json.loads(run_auditbench_e0(cfg).read_text(encoding="utf-8"))
    problems = report["manifest"]["schema_problems"]
    assert "bad.entry" in problems
    assert report["gate"]["gate"] == "FAILED"