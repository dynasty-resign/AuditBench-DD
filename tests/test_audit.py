import json

import numpy as np
import torch

from ttd.audit import (
    _class_moment_distance,
    _effective_rank,
    _fid,
    _manifold_pr,
    _per_class_topology_distance,
    parse_run_name,
    run_audit_analyze,
)
from ttd.models import classifier


def test_parse_run_name():
    assert parse_run_name("mtt_topo_ipc10") == ("mtt_topo", 10, 0)
    assert parse_run_name("random_ipc1") == ("random", 1, 0)
    assert parse_run_name("mtt_topo_ipc10_s2") == ("mtt_topo", 10, 2)


def test_fid_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(400, 16))
    assert _fid(real, real.copy()) < 1e-6
    shifted = real + 3.0
    assert _fid(real, shifted) > 10.0


def test_effective_rank_detects_low_rank():
    rng = np.random.default_rng(0)
    low_rank = rng.normal(size=(200, 2)) @ rng.normal(size=(2, 16))
    full_rank = rng.normal(size=(200, 16))
    assert _effective_rank(low_rank) < 2.5
    assert _effective_rank(full_rank) > 8.0


def test_manifold_pr_perfect_overlap():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(300, 8))
    precision, recall, coverage = _manifold_pr(real, real.copy(), k=3)
    assert precision == 1.0
    assert coverage == 1.0
    assert recall > 0.9


def test_class_moment_distance_label_sensitivity():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 50)
    real = rng.normal(size=(200, 6)) + labels[:, None] * 5.0
    syn = real[::2]
    syn_y = labels[::2]
    good = _class_moment_distance(syn, syn_y, real, labels)
    bad = _class_moment_distance(syn, rng.permutation(syn_y), real, labels)
    assert bad > good


def test_topology_distance_identical_is_zero_and_none_for_singletons():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 8)
    states = rng.normal(size=(24, 2, 6))
    dist = _per_class_topology_distance(
        states, labels, states, labels, "cosine", resamples=2, seed=0
    )
    assert dist < 1e-8
    single = states[:3]
    single_y = np.arange(3)
    assert (
        _per_class_topology_distance(single, single_y, states, labels, "cosine", 2, 0) is None
    )


def test_new_architectures_forward():
    x = torch.randn(2, 3, 32, 32)
    for name in ("vgg11", "mobilenet_v2"):
        model = classifier(name, 10, 32)
        assert model(x).shape == (2, 10)


def _fake_candidate(runs_dir, cache_dir, name, ipc, utility, scores):
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    torch.save(
        {"images": torch.zeros(2, 3, 32, 32), "labels": torch.zeros(2, dtype=torch.long)},
        run_dir / f"synthetic_{name}.pt",
    )
    summary = [
        {"architecture": arch, "mean": value, "std": 0.0, "n": 3}
        for arch, value in utility.items()
    ]
    (run_dir / f"synthetic_{name}.evaluation.json").write_text(
        json.dumps({"runs": [], "summary": summary}), encoding="utf-8"
    )
    method, ipc_parsed, synthesis_seed = parse_run_name(name)
    topo = scores.get("topo_traj_dist")
    nulls = {"labelperm_topo_traj_dist": (topo + 0.1) if topo is not None else None}
    record = {
        "run": name,
        "method": method,
        "ipc": ipc_parsed,
        "synthesis_seed": synthesis_seed,
        "n_synthetic": 2,
        "scores": {**scores, "nulls": nulls},
    }
    (cache_dir / f"{name}.scores.json").write_text(json.dumps(record), encoding="utf-8")


def test_audit_analyze_report(tmp_path):
    runs_dir = tmp_path / "runs"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rng = np.random.default_rng(0)
    names = ["mtt", "mtt_topo", "mtt_tv", "ttd", "feature", "random"]
    for ipc in (10, 50):
        for index, method in enumerate(names):
            quality = (index + 1) / len(names) + 0.05 * rng.normal()
            utility = {"convnet": 0.2 + 0.3 * quality, "vgg11": 0.25 + 0.25 * quality}
            scores = {
                "objective_loss": 2.0 - quality,
                "ridge_proxy": 0.1 + 0.5 * quality,
                "class_moment_dist": 1.0 - 0.5 * quality,
                "fid_endpoint": 3.0 - 2.0 * quality,
                "knn_precision": 0.2 + 0.6 * quality,
                "knn_recall": 0.1 + 0.3 * quality,
                "knn_coverage": 0.05 + 0.2 * quality,
                "effective_rank": 10 + 50 * quality,
                "short_run_5ep": 0.15 + 0.4 * quality,
                "topo_endpoint_dist": 1.0 - 0.8 * quality,
                "topo_traj_dist": 1.2 - 0.9 * quality,
                "betti_traj_dist": 0.8 - 0.6 * quality,
                "runtime_sec": 12.0,
            }
            _fake_candidate(runs_dir, cache_dir, f"{method}_ipc{ipc}", ipc, utility, scores)
    from ttd.config import ExperimentConfig

    cfg = ExperimentConfig(output_dir=str(tmp_path))
    out = run_audit_analyze(cfg, runs_dir, cache_dir)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["protocol"]["candidates"] == 12
    for arch in ("convnet", "vgg11"):
        for ipc in ("10", "50"):
            table = report["spearman_within_ipc"][arch][ipc]
            assert table["topo_traj_dist"] > 0.9
            assert table["objective_loss"] > 0.9
            regret = report["top1_regret"][arch][ipc]
            assert regret["topo_traj_dist"] == 0.0 or regret["topo_traj_dist"] >= 0.0
            assert "random_expected_regret" in regret
    assert "lomo" in report and "convnet" in report["lomo"]
    nulls = report["null_checks"]
    assert nulls["labelperm_topo_traj_dist"]["fraction_degraded"] == 1.0
