from __future__ import annotations

from ttd.phase1 import _oof_residuals, _rank_matrix


def _row(family: str, entry: int, architecture: str) -> dict:
    quality = entry / 10.0 + (0.02 if architecture == "convnet" else 0.0)
    return {
        "dataset": "toy",
        "entry_id": f"{family}.{entry}",
        "release_lineage": f"lineage_{family}",
        "method_family": family,
        "ipc": 10 if entry % 2 else 50,
        "architecture": architecture,
        "utility": quality,
        "scores": {
            "ridge_proxy": quality,
            "knn_coverage": quality / 2,
            "class_moment_dist": 1.0 - quality,
            "effective_rank": 10 + entry,
            "topo_endpoint_dist": 1.0 - quality / 2,
            "topo_traj_dist": 1.0 - quality,
            "betti_traj_dist": 1.0 - quality / 3,
        },
        "nulls": {},
    }


def test_phase1_rank_table_and_oof_residuals():
    rows = [
        _row(family, index, architecture)
        for family in "abc"
        for index in (1, 2, 3, 4, 5, 6)
        for architecture in ("convnet", "vgg11")
    ]
    table = _rank_matrix(rows, ("ridge_proxy", "topo_traj_dist"))
    assert table["dataset"]["toy"]["10"]["convnet"]["ridge_proxy"]["rho"] is not None
    records, coefficients = _oof_residuals(rows, "topo_traj_dist")
    assert len(records) == len(rows)
    assert len(coefficients["toy"]) == 3
