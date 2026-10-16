"""CPU-only Phase-0 smoke checks for the proxy and analysis paths."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .audit import _per_class_topology_distance, run_audit_analyze
from .config import ExperimentConfig
from .diffusion import DiffusionSchedule
from .models import LatentDiffusion
from .topology import persistence_signature
from .trajectory import trajectory_embeddings
from .utils import save_json, seed_everything


def _write_analysis_fixture(root: Path) -> tuple[Path, Path]:
    runs_dir, cache_dir = root / "analysis_runs", root / "analysis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ipc in (10, 50):
        for rank, method in enumerate(("random", "feature", "ttd"), start=1):
            name = f"{method}_ipc{ipc}"
            run_dir = runs_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"images": torch.zeros(2, 3, 4, 4), "labels": torch.zeros(2)}, run_dir / f"synthetic_{name}.pt")
            utility = 0.2 + 0.1 * rank + 0.001 * ipc
            (run_dir / f"synthetic_{name}.evaluation.json").write_text(
                json.dumps({"summary": [{"architecture": "convnet", "mean": utility}]}), encoding="utf-8"
            )
            quality = float(rank)
            scores = {
                "ridge_proxy": quality,
                "knn_coverage": quality,
                "class_moment_dist": 4.0 - quality,
                "topo_endpoint_dist": 4.0 - quality,
                "topo_traj_dist": 4.0 - quality,
                "betti_traj_dist": 4.0 - quality,
                "runtime_sec": 0.0,
                "nulls": {"labelperm_topo_traj_dist": 5.0 - quality},
            }
            (cache_dir / f"{name}.scores.json").write_text(
                json.dumps({"run": name, "method": method, "ipc": ipc, "scores": scores}), encoding="utf-8"
            )
    return runs_dir, cache_dir


def run_phase0_smoke(cfg: ExperimentConfig) -> Path:
    """Exercise proxy extraction primitives and the complete analysis writer.

    This does not download data, train a student or consume a checkpoint.  It
    is deliberately separate from evidence-producing experiments.
    """
    seed_everything(cfg.seed)
    output = cfg.run_dir / "phase0_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model = LatentDiffusion(latent_channels=4, width=8, embed_dim=16, num_classes=10).eval()
    model.set_latent_stats(torch.zeros(4), torch.ones(4))
    images = torch.randn(6, 3, 32, 32)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    schedule = DiffusionSchedule(10, 1e-4, 2e-2, device)
    noise = torch.randn(6, 4, 8, 8)
    states = trajectory_embeddings(
        model, images, [9, 4, 0], schedule, noise,
        space="latent", direction="reverse", labels=labels,
        conditioning="unconditional", source="inversion",
        representation="predicted_clean", inference_steps=2,
    )
    states_np = states.detach().numpy()
    topology_identity = _per_class_topology_distance(
        states_np, labels.numpy(), states_np, labels.numpy(), "cosine", resamples=1, seed=cfg.seed
    )
    signature = persistence_signature(states[:, -1], signature_size=3, metric="cosine")
    runs_dir, cache_dir = _write_analysis_fixture(output.parent / "phase0_analysis_fixture")
    analysis = run_audit_analyze(cfg, runs_dir, cache_dir, architectures=["convnet"])
    report = json.loads(analysis.read_text(encoding="utf-8"))
    payload = {
        "kind": "phase0_cpu_smoke",
        "network": "not used",
        "proxy": {
            "states_shape": list(states.shape),
            "finite": bool(torch.isfinite(states).all()),
            "signature_shape": list(signature.shape),
            "topology_identity_distance": topology_identity,
        },
        "analysis": {
            "report": str(analysis),
            "candidate_count": report["protocol"]["candidates"],
            "spearman": report["spearman_within_ipc"]["convnet"],
        },
    }
    if topology_identity is None or abs(topology_identity) > 1e-7:
        raise RuntimeError("topology identity smoke failed")
    if not payload["proxy"]["finite"] or payload["analysis"]["candidate_count"] != 6:
        raise RuntimeError("Phase-0 proxy/analysis smoke failed")
    save_json(payload, output)
    return output
