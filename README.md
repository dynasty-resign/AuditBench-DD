# AuditBench-DD

Training-free proxy benchmark for distilled-set utility under architecture and method shift.

研究现状、拓扑失败诊断、CDG-Proxy 方法重构与确认实验路线见 [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md)。

## Project scope

This repository contains the reproducible research code, experiment configurations, datasets and external-data provenance, final experiment artifacts, paper materials, and the SSH entry points used for remote GPU execution.

The current codebase covers AuditBench stages E0–E5, pooled analysis, external-bank provenance, gradient-alignment proxies, and proxy-search components.

## Layout

| Path | Purpose |
|---|---|
| `src/ttd/` | Research implementation and CLI package |
| `configs/` | Versioned experiment configurations |
| `tests/` | Unit and regression tests for the research code |
| `scripts/` | Experiment helpers and retained SSH/GPU utilities |
| `data/` | Local benchmark datasets |
| `external/` | External distilled-set artifacts |
| `manifests/` | External-data source and provenance manifests |
| `runs/` | Retained experiment results and evidence artifacts |
| `paper/` | Research manuscript, figures, claims, and literature records |
| `参考论文/` | Reference papers and reading notes |

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Phase-0 smoke and research commands

```bash
python -m ttd.cli validate --config configs/smoke.yaml
python -m ttd.cli smoke-audit --config configs/smoke.yaml
python -m ttd.cli phase1-diagnose --config configs/smoke.yaml --out-dir runs/phase1

# External-artifact provenance gate
python -m ttd.cli auditbench-e0 --config configs/auditbench_e0_cifar10.yaml

# Train / extract / evaluate an explicitly configured experiment
python -m ttd.cli train-backbone --config configs/auditbench_e1_cifar10.yaml
python -m ttd.cli extract-targets --config configs/auditbench_e1_cifar10.yaml
python -m ttd.cli distill --config configs/auditbench_e1_cifar10.yaml
```

`smoke-audit` is CPU-only and network-free; it verifies proxy extraction and audit analysis without creating scientific evidence. `phase1-diagnose` is a read-only, low-compute analysis of the existing E1--E3 reports; its decision and limits are documented in [docs/PHASE1_DIAGNOSTICS.md](docs/PHASE1_DIAGNOSTICS.md). Experiment outputs are written below `runs/`. The Phase-0 recovery audit is in [docs/PHASE0_RECOVERY.md](docs/PHASE0_RECOVERY.md).

## Tests and lint

```bash
python -m pytest
ruff check src tests scripts
```

## Remote GPU execution

SSH support is intentionally retained in `scripts/ssh_common.py`, `scripts/ssh_exec.py`, `scripts/ssh_upload.py`, and `scripts/ssh_download.py`. The remote host, port, and username are defined in `scripts/ssh_common.py`; credentials remain outside the repository.

Remote commands should use the retained GPU policy helper and write only to the authorized project directory on the remote host.
