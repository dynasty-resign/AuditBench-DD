"""AuditBench-DD E0: build, audit and freeze the CIFAR-10 external bank.

Pipeline stages: manifest load -> provenance validation -> fetch -> hash
verification -> normalization/QC (external_bank) -> coverage matrix ->
pre-frozen E1 protocol emission -> E0 gate decision.

This stage never trains student models, never computes proxy/utility
correlations, and never selects artifacts by reported accuracy.
"""

from __future__ import annotations

import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .external_bank import STRATUM_PRIMARY, build_bank
from .provenance import ProvenanceRecord, iter_spec_parts, sha256_file
from .utils import save_json

E0_GATES = ("PASS_STRONG", "PASS_MINIMUM", "HUMAN_REVIEW", "FAILED")

# Quarantine reasons that block any PASS gate even when coverage is met:
# the preprocessing semantics or content integrity of an ingested artifact
# is unresolved and needs human judgment.
BLOCKING_QUARANTINE = ("suspect_preprocessing", "hash_mismatch", "unrecognized_layout")


def load_manifest(path: str | Path) -> tuple[list[ProvenanceRecord], list[str], dict[str, Any]]:
    """Load and validate the external-sources manifest."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    taxonomy = list(raw.get("taxonomy", []))
    if not taxonomy:
        raise ValueError("manifest must declare the pre-frozen method_family taxonomy")
    records = [ProvenanceRecord.from_dict(item) for item in raw.get("sources", [])]
    problems = {}
    seen_ids: set[str] = set()
    for record in records:
        issues = record.validate(taxonomy)
        if record.source_id in seen_ids:
            issues.append(f"duplicate source_id {record.source_id}")
        seen_ids.add(record.source_id)
        if issues:
            problems[record.source_id] = issues
    meta = {k: v for k, v in raw.items() if k not in {"sources", "taxonomy"}}
    return records, taxonomy, {"schema_problems": problems, **meta}


def _fetch_one_url(url: str, tmp: Path, timeout: float) -> str:
    """Fetch ``url`` into ``tmp``; return the transport actually used.

    Google Drive share links need the confirm-token flow that ``gdown``
    implements; plain HTTPS and ``file://``/local paths use urllib/shutil.
    """
    if url.startswith("file://") or "://" not in url:
        src = url.removeprefix("file://")
        shutil.copyfile(src, tmp)
        return "file_copy"
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError:
            gdown = None
        if gdown is not None:
            result = gdown.download(url, str(tmp), quiet=True)
            if result is None or not tmp.exists() or tmp.stat().st_size == 0:
                raise RuntimeError("gdown returned no file")
            return "gdown"
    req = urllib.request.Request(url, headers={"User-Agent": "auditbench-e0/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return "urllib"


def fetch_sources(
    records: list[ProvenanceRecord],
    raw_dir: str | Path,
    timeout: float = 120.0,
    retries: int = 1,
) -> list[dict[str, Any]]:
    """Download blobs for fetchable entries; verify and stage them.

    ``file://`` URLs and plain paths are supported so vetted transport copies
    can be staged manually when direct egress is unavailable. Failures are
    recorded per entry and never abort the whole run.
    """
    raw_dir = Path(raw_dir)
    outcomes: list[dict[str, Any]] = []
    for record in records:
        if record.status == "pending_url_verification":
            outcomes.append(
                {
                    "source_id": record.source_id,
                    "status": "skipped",
                    "reason": "official URL not yet verified by a human",
                }
            )
            continue
        if not record.files:
            outcomes.append(
                {
                    "source_id": record.source_id,
                    "status": "skipped",
                    "reason": "no file map declared; blob-to-IPC mapping must be explicit",
                }
            )
            continue
        target_dir = raw_dir / record.source_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_spec in record.files:
            for part in iter_spec_parts(file_spec):
                outcomes.append(
                    _fetch_part(record, part, target_dir, timeout, retries)
                )
    return outcomes


def _fetch_part(
    record: ProvenanceRecord,
    part: dict[str, Any],
    target_dir: Path,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    relpath = part["relpath"]
    dest = target_dir / relpath
    if dest.exists():
        return {
            "source_id": record.source_id,
            "relpath": relpath,
            "status": "already_staged",
            "sha256": sha256_file(dest),
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    error: str | None = None
    fetched: dict[str, Any] | None = None
    urls = [part["url"]] if part.get("url") else list(record.official_urls)
    for url in urls:
        if fetched is not None:
            break
        for attempt in range(max(retries, 1)):
            tmp = dest.with_name(dest.name + ".part")
            try:
                transport = _fetch_one_url(url, tmp, timeout)
                tmp.replace(dest)
                fetched = {
                    "source_id": record.source_id,
                    "relpath": relpath,
                    "url_used": url,
                    "transport": transport,
                    "status": "fetched",
                    "sha256": sha256_file(dest),
                    "error": None,
                }
                break
            except Exception as exc:  # noqa: BLE001 - recorded verbatim
                error = f"{type(exc).__name__}: {exc}"[:300]
                if tmp.exists():
                    tmp.unlink()
                time.sleep(min(2**attempt, 8))
    if fetched is not None:
        return fetched
    return {
        "source_id": record.source_id,
        "relpath": relpath,
        "url_used": None,
        "status": "fetch_failed",
        "sha256": None,
        "error": error or "no official URL registered",
    }


def coverage_matrix(index: dict[str, Any], primary_ipcs: list[int]) -> dict[str, Any]:
    """Lineage x IPC coverage over the primary (hard-label, natural) cohort."""
    primary = [
        e
        for e in index["entries"]
        if e.get("stratum") == STRATUM_PRIMARY and e.get("bank_path") and e["ipc"] in primary_ipcs
    ]
    lineages: dict[str, dict[str, Any]] = {}
    for entry in primary:
        lin = lineages.setdefault(
            entry["release_lineage"],
            {"method": entry["method"], "method_family": entry["method_family"], "ipcs": set()},
        )
        lin["ipcs"].add(entry["ipc"])
    matrix = {
        lineage: {
            "method": info["method"],
            "method_family": info["method_family"],
            "ipcs": sorted(info["ipcs"]),
        }
        for lineage, info in sorted(lineages.items())
    }
    families = sorted({info["method_family"] for info in lineages.values()})
    dual = [name for name, info in matrix.items() if set(primary_ipcs) <= set(info["ipcs"])]
    return {
        "primary_entries": len(primary),
        "lineages": matrix,
        "n_lineages": len(matrix),
        "families": families,
        "n_families": len(families),
        "dual_ipc_lineages": sorted(dual),
        "n_dual_ipc_lineages": len(dual),
    }


def decide_gate(
    coverage: dict[str, Any],
    index: dict[str, Any],
    fetch_outcomes: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Map measured coverage to one of the four E0 gates (pure function)."""
    reasons: list[str] = []
    fetched_ok = sum(1 for o in fetch_outcomes if o.get("status") in {"fetched", "already_staged"})
    blocking = sorted(
        {
            reason.split(":")[0]
            for entry in index["entries"]
            for reason in entry.get("quarantine_reasons", [])
            if entry.get("bank_path")
        }
        & set(BLOCKING_QUARANTINE)
    )
    n_primary = coverage["primary_entries"]
    if n_primary == 0:
        if fetched_ok == 0:
            reasons.append("no artifact could be fetched or staged in this environment")
        else:
            reasons.append("blobs were fetched but none passed provenance/QC into the primary cohort")
        return {"gate": "FAILED", "reasons": reasons, "blocking_quarantine": blocking}

    strong = thresholds["pass_strong"]
    minimum = thresholds["pass_minimum"]
    strong_ok = (
        coverage["n_lineages"] >= strong["min_lineages"]
        and coverage["n_families"] >= strong["min_families"]
        and coverage["n_dual_ipc_lineages"] >= strong["min_dual_ipc_lineages"]
        and n_primary >= strong["min_primary_entries"]
    )
    minimum_ok = (
        coverage["n_lineages"] >= minimum["min_lineages"]
        and coverage["n_families"] >= minimum["min_families"]
        and coverage["n_dual_ipc_lineages"] >= minimum["min_dual_ipc_lineages"]
        and n_primary >= minimum["min_primary_entries"]
    )
    if strong_ok and not blocking:
        gate = "PASS_STRONG"
        reasons.append("coverage meets the pre-frozen PASS_STRONG thresholds")
    elif minimum_ok and not blocking:
        gate = "PASS_MINIMUM"
        reasons.append("coverage meets only the pre-frozen PASS_MINIMUM thresholds")
    else:
        gate = "HUMAN_REVIEW"
        if blocking:
            reasons.append(f"blocking quarantine reasons present: {blocking}")
        if not minimum_ok:
            reasons.append(
                "coverage below PASS_MINIMUM: "
                f"lineages={coverage['n_lineages']}, families={coverage['n_families']}, "
                f"dual-IPC lineages={coverage['n_dual_ipc_lineages']}, primary={n_primary}"
            )
    return {"gate": gate, "reasons": reasons, "blocking_quarantine": blocking}


def freeze_e1_protocol(
    taxonomy: list[str],
    primary_ipcs: list[int],
    extreme_ipcs: list[int],
    dataset: str = "cifar10",
) -> dict[str, Any]:
    """Pre-frozen blind-test protocol (data, fixed before any utility result)."""
    return {
        "frozen_by": "auditbench-e0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "primary_cohort": {
            "dataset": dataset,
            "ipcs": primary_ipcs,
            "label_type": "hard",
            "image_form": (
                "natural pixels; no ZCA, no decoder dependence. Synthetic "
                "distillation releases are stored as released free-float "
                "tensors (raw_unbounded, value ranges recorded); selection "
                "releases storing z-scored real images are inverted with "
                "pinned verified constants into [0,1]."
            ),
        },
        "strata": {
            "extreme_compression": {"ipcs": extreme_ipcs, "use": "descriptive stratum only"},
            "quarantine": [
                "soft_label_dependent (learned soft labels are never argmaxed into primary)",
                "zca_only",
                "decoder_dependent / generator-dependent",
            ],
        },
        "grouping": {
            "release_lineage": "independent paper release; cluster unit for all resampling",
            "method_family": taxonomy,
            "leave_one_family_out": True,
        },
        "blinding": [
            "proxy extraction receives only normalized tensors plus ipc, never method names",
            "artifact inclusion decided by provenance/QC before any utility is measured",
            "no artifact may be selected or dropped using official reported accuracy",
            "utility is measured only in E1 (student training), never in E0",
        ],
        "resampling": [
            "bootstrap/permutation clusters are release_lineage units",
            "the same artifact must never appear in both train and test of any resample",
        ],
        "utility_metric": "hard-label student test accuracy (E1); top-k regret and grouped CIs",
        "forbidden_in_e1": [
            "training-free claims of cross-architecture generalization from internal pool only",
            "reviving closed claims (topology-loss gain, trajectory utility main line, H0-best)",
        ],
    }


def run_auditbench_e0(cfg) -> Path:
    """Execute the full E0 pipeline and write report + frozen E1 protocol."""
    started = datetime.now(timezone.utc)
    records, taxonomy, manifest_meta = load_manifest(cfg.manifest_path)
    raw_dir = Path(cfg.raw_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fetch_outcomes: list[dict[str, Any]] = []
    if cfg.fetch_enabled:
        fetch_outcomes = fetch_sources(
            records, raw_dir, timeout=cfg.fetch_timeout_sec, retries=cfg.fetch_retries
        )

    index = build_bank(records, raw_dir, out_dir, cfg.num_classes, cfg.image_size)
    coverage = coverage_matrix(index, cfg.primary_ipcs)
    gate = decide_gate(coverage, index, fetch_outcomes, cfg.gate_thresholds)
    e1_protocol = freeze_e1_protocol(taxonomy, cfg.primary_ipcs, cfg.extreme_ipcs, cfg.dataset)

    report = {
        "stage": getattr(cfg, "stage", "auditbench_e0"),
        "dataset": cfg.dataset,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(cfg.manifest_path), **manifest_meta},
        "fetch": {"enabled": cfg.fetch_enabled, "outcomes": fetch_outcomes},
        "bank_index": str(out_dir / "bank_index.json"),
        "n_entries": len(index["entries"]),
        "coverage": coverage,
        "quarantine": [
            {
                "entry_id": e["entry_id"],
                "stratum": e["stratum"],
                "reasons": e.get("quarantine_reasons", []),
            }
            for e in index["entries"]
            if e.get("stratum") != STRATUM_PRIMARY
        ],
        "duplicates": index["duplicates"],
        "gate": gate,
        "gate_thresholds": cfg.gate_thresholds,
        "e1_protocol": e1_protocol,
    }
    report_path = Path(cfg.report_path)
    save_json(report, report_path)
    save_json(e1_protocol, cfg.e1_protocol_path)
    resolved = {
        "taxonomy": taxonomy,
        "sources": [r.to_dict() for r in records],
    }
    resolved_path = Path(cfg.resolved_manifest_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)
    return report_path