"""iter-0003 frozen greedy forward-selection search engine.

Implements exactly the algorithm frozen in
runs/evidence_packs/iter0003_search/search_preregistration.json:

* combination = unweighted rank-average (scipy rankdata, method='average');
* step rule = add the component with maximal mean-Spearman gain;
* stop rule = gain < epsilon (0.01), all components used, or budget exhausted;
* singleton constraint = member-only components (grad_norm) skipped at step 1;
* selection = primary key mean Spearman, tie-break min Spearman;
* gate = >=2/3 datasets strictly above the control-arm frozen max AND
  combination null |rho| < 0.3 on each exceeding dataset.

Screening-level only: Spearman rho values without p-values; single label
permutation nulls. No inferential statistics anywhere in this module.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .components import rank_average

EPSILON_GAIN = 0.01
EVALUATION_BUDGET_MAX = 64
NULL_ABS_THRESHOLD = 0.3


def _spearman(values: np.ndarray, utility: np.ndarray) -> float:
    return float(spearmanr(np.asarray(values, dtype=np.float64), np.asarray(utility, dtype=np.float64)).statistic)


def evaluate_combination(
    members: tuple[str, ...],
    observed: dict[str, dict[str, np.ndarray]],
    nulls: dict[str, dict[str, np.ndarray]],
    utility: dict[str, np.ndarray],
    control_max: dict[str, float],
) -> dict:
    """Evaluate one combination on all datasets.

    ``observed``/``nulls`` map component id -> dataset -> per-candidate values;
    ``utility`` maps dataset -> per-candidate utility; ``control_max`` maps
    dataset -> old-three frozen max.
    """
    datasets = sorted(utility)
    per_dataset: dict[str, dict] = {}
    for ds in datasets:
        combo = rank_average([observed[m][ds] for m in members])
        combo_null = rank_average([nulls[m][ds] for m in members])
        rho = _spearman(combo, utility[ds])
        rho_null = _spearman(combo_null, utility[ds])
        per_dataset[ds] = {
            "spearman": rho,
            "null_spearman": rho_null,
            "exceeds_control_max": bool(rho > control_max[ds]),
        }
    mean_rho = float(np.mean([per_dataset[ds]["spearman"] for ds in datasets]))
    min_rho = float(np.min([per_dataset[ds]["spearman"] for ds in datasets]))
    exceeding = [ds for ds in datasets if per_dataset[ds]["exceeds_control_max"]]
    null_ok = all(abs(per_dataset[ds]["null_spearman"]) < NULL_ABS_THRESHOLD for ds in exceeding)
    qualifies = len(exceeding) >= 2 and null_ok
    return {
        "members": list(members),
        "per_dataset": per_dataset,
        "mean_spearman": mean_rho,
        "min_spearman": min_rho,
        "datasets_exceeding_control_max": exceeding,
        "num_datasets_exceeding": len(exceeding),
        "null_ok_on_exceeding": bool(null_ok),
        "qualifies": bool(qualifies),
    }


def greedy_forward_selection(
    components: list[str],
    member_only: set[str],
    observed: dict[str, dict[str, np.ndarray]],
    nulls: dict[str, dict[str, np.ndarray]],
    utility: dict[str, np.ndarray],
    control_max: dict[str, float],
    *,
    epsilon: float = EPSILON_GAIN,
    budget: int = EVALUATION_BUDGET_MAX,
) -> dict:
    """Run the frozen greedy forward selection. Returns the full trace."""
    evaluations: list[dict] = []
    current: list[str] = []
    current_mean: float | None = None
    steps: list[dict] = []
    spent = 0

    def _evaluate(members: list[str]) -> dict:
        nonlocal spent
        if spent >= budget:
            raise RuntimeError("evaluation budget exhausted")
        result = evaluate_combination(tuple(members), observed, nulls, utility, control_max)
        spent += 1
        evaluations.append(result)
        return result

    while True:
        remaining = [c for c in components if c not in current]
        if not current:
            remaining = [c for c in remaining if c not in member_only]
        if not remaining:
            steps.append({"step": len(steps) + 1, "action": "stop", "reason": "no remaining components"})
            break
        best_result: dict | None = None
        best_component: str | None = None
        candidates_this_step: list[dict] = []
        for component in remaining:
            result = _evaluate(current + [component])
            gain = None if current_mean is None else result["mean_spearman"] - current_mean
            candidates_this_step.append(
                {"component": component, "mean_spearman": result["mean_spearman"], "gain": gain}
            )
            if best_result is None or (
                result["mean_spearman"], result["min_spearman"]
            ) > (best_result["mean_spearman"], best_result["min_spearman"]):
                best_result = result
                best_component = component
        assert best_result is not None and best_component is not None
        if current_mean is not None and best_result["mean_spearman"] - current_mean < epsilon:
            steps.append(
                {
                    "step": len(steps) + 1,
                    "action": "stop",
                    "reason": f"maximal gain {best_result['mean_spearman'] - current_mean:.6f} < epsilon={epsilon}",
                    "candidates": candidates_this_step,
                }
            )
            break
        current = current + [best_component]
        current_mean = best_result["mean_spearman"]
        steps.append(
            {
                "step": len(steps) + 1,
                "action": "add",
                "component": best_component,
                "mean_spearman": current_mean,
                "candidates": candidates_this_step,
            }
        )
    return {
        "evaluations": evaluations,
        "steps": steps,
        "final_members": current,
        "evaluations_spent": spent,
        "budget": budget,
        "epsilon": epsilon,
    }


def select_winner(evaluations: list[dict]) -> dict:
    """Apply the frozen selection rule over all evaluated combinations."""
    ranked = sorted(
        evaluations,
        key=lambda r: (r["mean_spearman"], r["min_spearman"]),
        reverse=True,
    )
    qualifying = [r for r in ranked if r["qualifies"]]
    if qualifying:
        return {
            "outcome": "QUALIFYING_COMBINATION_FOUND",
            "winner": qualifying[0],
            "num_qualifying": len(qualifying),
            "ranked_top": ranked[:5],
        }
    return {
        "outcome": "SEARCH_NO_QUALIFYING_COMBINATION",
        "winner": None,
        "num_qualifying": 0,
        "ranked_top": ranked[:5],
    }