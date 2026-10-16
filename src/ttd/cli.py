"""Command line interface for the consolidated AuditBench-DD research core."""

from __future__ import annotations

import argparse

from .audit import run_audit_analyze, run_audit_extract
from .auditbench import run_auditbench_e0
from .config import load_auditbench_config, load_config
from .data import build_datasets
from .pipeline import (
    distill,
    evaluate,
    extract_targets,
    fit_latent_statistics,
    train_autoencoder,
    train_backbone,
    train_diffusion,
)
from .phase1 import run_phase1_diagnostics
from .smoke import run_phase0_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttd", description="AuditBench-DD research pipeline")
    parser.add_argument(
        "command",
        choices=[
            "validate", "prepare-data", "train-autoencoder", "fit-latent-stats",
            "train-diffusion", "train-backbone", "extract-targets", "distill", "evaluate",
            "audit-extract", "audit-analyze", "auditbench-e0", "smoke-audit", "phase1-diagnose",
        ],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--synthetic", help="synthetic .pt artifact for evaluate")
    parser.add_argument("--targets", help="target .pt artifact for distill")
    parser.add_argument("--runs-dir", help="candidate run directory for audit")
    parser.add_argument("--cache", help="score cache directory for audit")
    parser.add_argument("--artifact", help="real-reference phase1 artifact for audit extraction")
    parser.add_argument("--out-dir", help="output directory for a diagnostic command")
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "auditbench-e0":
        print(run_auditbench_e0(load_auditbench_config(args.config, args.set)))
        return
    cfg = load_config(args.config, args.set)
    if args.command == "validate":
        print(f"valid: {args.config}")
    elif args.command == "prepare-data":
        train, test = build_datasets(cfg.dataset)
        print(f"prepared {cfg.dataset.name}: train={len(train)}, test={len(test)}")
    elif args.command == "train-autoencoder":
        print(train_autoencoder(cfg))
    elif args.command == "fit-latent-stats":
        print(fit_latent_statistics(cfg))
    elif args.command == "train-diffusion":
        print(train_diffusion(cfg))
    elif args.command == "train-backbone":
        print(train_backbone(cfg))
    elif args.command == "extract-targets":
        print(extract_targets(cfg))
    elif args.command == "distill":
        print(distill(cfg, args.targets))
    elif args.command == "evaluate":
        if not args.synthetic:
            raise SystemExit("--synthetic is required for evaluate")
        print(evaluate(cfg, args.synthetic))
    elif args.command == "audit-extract":
        if not args.runs_dir or not args.cache or not args.artifact:
            raise SystemExit("audit-extract requires --runs-dir, --cache and --artifact")
        print(run_audit_extract(cfg, args.runs_dir, args.cache, args.artifact))
    elif args.command == "audit-analyze":
        if not args.runs_dir or not args.cache:
            raise SystemExit("audit-analyze requires --runs-dir and --cache")
        print(run_audit_analyze(cfg, args.runs_dir, args.cache))
    elif args.command == "smoke-audit":
        print(run_phase0_smoke(cfg))
    elif args.command == "phase1-diagnose":
        print(run_phase1_diagnostics(out_dir=args.out_dir or "runs/phase1", bootstrap_reps=args.bootstrap_reps))


if __name__ == "__main__":
    main()
