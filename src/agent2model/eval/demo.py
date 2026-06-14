"""Build a demo :class:`EvalRunResult` from canned scores — no API, no GPU.

The eval report (per-criterion bar charts with CIs, failure rates, cost breakdown,
significance tests) is the library's headline differentiator, but it is otherwise
unreachable without an Anthropic key *and* a served compiled model. This module
synthesises per-conversation scores around the paper's published travel-booking
means and runs them through the **real** aggregation/statistics pipeline
(:func:`summarize_condition`, :func:`compare_conditions`), so ``agent2model eval
--demo`` can render an authentic-looking report for free.

The numbers are illustrative (sampled around Dennis et al. 2026's reported means),
NOT a measurement produced by this library — the CLI labels them as such.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from agent2model.eval.judge import JudgeVerdict
from agent2model.eval.rubric import RUBRIC, CriterionName
from agent2model.eval.runner import (
    ConditionResult,
    EvalRunResult,
    compare_conditions,
    summarize_condition,
)


class _DemoCondition(NamedTuple):
    """An illustrative condition spec: per-criterion target means + cost/latency."""

    name: str
    means: tuple[float, float, float, float, float]
    cost_per_conv: float
    wall_clock_s: float


#: Per-condition targets transcribed from the paper's travel-booking table. The
#: means are in the rubric's reporting order. Illustrative only — NOT a measurement.
_DEMO_CONDITIONS: tuple[_DemoCondition, ...] = (
    _DemoCondition("compiled", (4.11, 4.75, 4.34, 4.07, 4.12), 0.0010, 1.8),
    _DemoCondition("same_model_orch", (3.93, 4.69, 4.12, 3.87, 3.96), 0.052, 6.0),
    _DemoCondition("langgraph", (4.17, 4.21, 4.32, 4.62, 4.84), 0.077, 7.1),
    _DemoCondition("in_context", (4.53, 4.64, 4.96, 4.96, 5.00), 0.133, 8.4),
)


def _sample_verdicts(
    means: tuple[float, ...], n: int, rng: np.random.Generator
) -> list[JudgeVerdict]:
    """Sample ``n`` integer (1-5) verdicts whose per-criterion means approximate ``means``."""
    names: tuple[CriterionName, ...] = RUBRIC.names()
    verdicts: list[JudgeVerdict] = []
    for _ in range(n):
        scores: dict[CriterionName, int] = {}
        for name, mean in zip(names, means, strict=True):
            value = int(np.clip(round(rng.normal(mean, 0.55)), 1, 5))
            scores[name] = value
        verdicts.append(JudgeVerdict(scores=scores, user_posed_challenge=True))
    return verdicts


def demo_eval_result(flowchart_name: str, *, n: int = 50, seed: int = 0) -> EvalRunResult:
    """Construct an illustrative :class:`EvalRunResult` with no API calls.

    Args:
        flowchart_name: Procedure name to stamp on the report.
        n: Conversations per condition.
        seed: RNG seed for reproducible sampled scores.

    Returns:
        A fully-populated result (conditions + pairwise comparisons + costs) that
        renders through the normal report writers. Numbers are illustrative.
    """
    rng = np.random.default_rng(seed)
    conditions: list[ConditionResult] = []
    for spec in _DEMO_CONDITIONS:
        verdicts = _sample_verdicts(spec.means, n, rng)
        conditions.append(
            summarize_condition(
                spec.name,
                verdicts,
                cost_usd=spec.cost_per_conv * n,
                wall_clock_s=[spec.wall_clock_s] * n,
                seed=seed,
            )
        )

    # Compare each baseline against the compiled condition (paired — shared scenarios).
    compiled = next(c for c in conditions if c.condition == "compiled")
    comparisons = [
        compare_conditions(compiled, other, paired=True)
        for other in conditions
        if other.condition != "compiled"
    ]
    return EvalRunResult(
        flowchart_name=flowchart_name,
        n=n,
        conditions=conditions,
        comparisons=comparisons,
        total_cost_usd=sum(c.cost_usd for c in conditions),
    )
