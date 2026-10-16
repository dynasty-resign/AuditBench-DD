"""Provenance auditing for external condensed-set artifacts (AuditBench-DD E0).

Only artifacts whose origin can be traced to an official public source may
enter the external candidate bank. The accepted source tiers mirror the
priority order fixed by the E0 task:

1. release/artifact linked directly from the paper authors' project page;
2. artifact provided by the authors' official code repository;
3. DC-BENCH official repository / official data package;
4. data regenerated with the authors' code at a pinned commit and official
   configuration (must be labelled ``REPRODUCED``);
5. community mirror, accepted only as a transport copy whose SHA-256 matches
   a tier 1-3 reference exactly.

Unverifiable third-party shares, repackaged data, screenshots, paper tables
or orphan ``.pt`` files are rejected.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SOURCE_TIERS = {
    1: "author_project_page",
    2: "author_official_repo",
    3: "dcbench_official",
    4: "reproduced",
    5: "community_mirror",
}

# Lifecycle of a manifest entry. ``pending_url_verification`` means the
# official URL itself still has to be confirmed by a human (no fetch allowed);
# ``pending_fetch`` means the URL is registered but nothing was downloaded yet.
ENTRY_STATES = (
    "pending_url_verification",
    "pending_fetch",
    "fetched",
    "hash_verified",
    "ingested",
    "rejected_unverifiable",
    "rejected_hash_mismatch",
    "rejected_fetch_failed",
)

ALLOWED_DATASETS = {"cifar10", "cifar100", "tiny_imagenet"}


FILE_ROLES = ("images", "labels", "blob")


def iter_spec_parts(file_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a manifest ``files`` entry into a list of part dicts.

    A spec either names a single blob via ``relpath`` or a multi-part
    artifact via ``parts`` (each part carrying ``relpath`` and a ``role``
    such as ``images``/``labels``). Every part may carry its own ``url``.
    """
    if "parts" in file_spec:
        parts = []
        for part in file_spec["parts"]:
            parts.append(
                {
                    "relpath": part["relpath"],
                    "role": part.get("role", "blob"),
                    "url": part.get("url", file_spec.get("url", "")),
                }
            )
        return parts
    return [
        {
            "relpath": file_spec["relpath"],
            "role": "blob",
            "url": file_spec.get("url", ""),
        }
    ]


def validate_file_spec(file_spec: dict[str, Any]) -> list[str]:
    """Return schema violations for one manifest ``files`` entry."""
    problems: list[str] = []
    if "ipc" not in file_spec:
        problems.append("file spec missing ipc")
    parts = file_spec.get("parts")
    if parts is None:
        if not file_spec.get("relpath"):
            problems.append("file spec missing relpath")
    else:
        if not parts:
            problems.append("file spec parts must be non-empty")
        roles = [part.get("role", "blob") for part in parts]
        for part in parts:
            if not part.get("relpath"):
                problems.append("file spec part missing relpath")
        for role in roles:
            if role not in FILE_ROLES:
                problems.append(f"unknown part role {role!r}")
    return problems


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ProvenanceRecord:
    """One auditable source entry from the external-sources manifest."""

    source_id: str
    release_lineage: str
    method: str
    method_family: str
    distribution: str
    dataset: str
    ipcs: list[int]
    source_tier: int
    official_urls: list[str] = field(default_factory=list)
    label_type_expected: str = "unknown"  # hard | soft | unknown
    preprocessing_expected: str = "unknown"  # raw | zca | normalized | unknown
    license: str = "to-verify"
    pinned_sha256: dict[str, str] = field(default_factory=dict)  # relpath -> sha256
    files: list[dict[str, Any]] = field(default_factory=list)  # relpath/ipc map
    status: str = "pending_fetch"
    notes: str = ""

    def validate(self, taxonomy: list[str]) -> list[str]:
        """Return a list of schema/policy violations (empty when valid)."""
        problems: list[str] = []
        if not self.source_id:
            problems.append("source_id must be non-empty")
        if self.source_tier not in SOURCE_TIERS:
            problems.append(f"source_tier must be one of {sorted(SOURCE_TIERS)}")
        if self.dataset not in ALLOWED_DATASETS:
            problems.append(f"dataset must be one of {sorted(ALLOWED_DATASETS)}")
        if self.method_family not in taxonomy:
            problems.append(f"method_family {self.method_family!r} not in pre-frozen taxonomy")
        if not self.ipcs or any(ipc < 1 for ipc in self.ipcs):
            problems.append("ipcs must be a non-empty list of positive integers")
        if self.status not in ENTRY_STATES:
            problems.append(f"status {self.status!r} not in {ENTRY_STATES}")
        if self.source_tier == 5 and not self.pinned_sha256:
            problems.append(
                "community mirrors (tier 5) require a pinned SHA-256 from a tier 1-3 reference"
            )
        if self.source_tier == 4 and "REPRODUCED" not in self.notes:
            problems.append("tier 4 entries must carry the REPRODUCED label in notes")
        if self.status != "pending_url_verification" and not self.official_urls:
            problems.append("entries ready for fetch must list at least one official URL")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProvenanceRecord:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown manifest fields for {payload.get('source_id')}: {unknown}")
        return cls(**payload)


def verify_blob(record: ProvenanceRecord, relpath: str, blob_path: str | Path) -> dict[str, Any]:
    """Hash ``blob_path`` and check it against the record's pinned hash.

    Trust-on-first-fetch applies only to tiers 1-4: when no hash is pinned the
    observed digest is reported with ``hash_pinned_by == "first_fetch"`` so a
    human can pin it in the manifest. Tier 5 mirrors must always match a
    pinned reference hash.
    """
    observed = sha256_file(blob_path)
    pinned = record.pinned_sha256.get(relpath)
    result = {
        "source_id": record.source_id,
        "relpath": relpath,
        "observed_sha256": observed,
        "pinned_sha256": pinned,
        "hash_verified": False,
        "hash_pinned_by": None,
    }
    if pinned is not None:
        result["hash_verified"] = observed == pinned
        result["hash_pinned_by"] = "manifest"
    elif record.source_tier in (1, 2, 3, 4):
        result["hash_verified"] = True
        result["hash_pinned_by"] = "first_fetch"
    return result