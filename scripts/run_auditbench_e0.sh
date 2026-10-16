#!/usr/bin/env bash
# AuditBench-DD E0: fetch + audit + normalize + freeze (CIFAR-10 only).
# Usage: bash scripts/run_auditbench_e0.sh [config]
# Env:   E0_FETCH=0 disables network fetch (staged file:// copies only).
set -euo pipefail

CONFIG="${1:-configs/auditbench_e0_cifar10.yaml}"
FETCH="${E0_FETCH:-1}"

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python" ]]; then
  PY=.venv/bin/python
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=.venv/Scripts/python.exe
else
  PY=python
fi

echo "[e0] config: ${CONFIG} (fetch=${FETCH})"
"$PY" -m ttd.cli auditbench-e0 --config "${CONFIG}" \
  --set "fetch_enabled=${FETCH}"
echo "[e0] summary:"
"$PY" scripts/show_auditbench_e0.py