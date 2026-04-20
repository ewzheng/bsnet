"""
validator.py

Heuristic evaluator for CheckResult objects.

This module determines whether a fact-check result passes or fails
based on evidence snippet scores (support / contradict / neutral).

"""

from typing import List, Literal

# >>> Adjust this import path to your project structure <<<
from model import CheckResult, EvidenceScore, ScoredClaim


Verdict = Literal["pass", "fail"]


def evaluate_check_result(result: CheckResult) -> Verdict:
    """
    Evaluate a CheckResult and return "pass" or "fail".

    This function applies a deterministic heuristic based on:
    - strongest support vs contradiction
    - number of supporting snippets
    - aggregate support vs contradiction
    - margin consistency across snippets
    - weak influence from the pipeline label

    Args:
        result: CheckResult object from the pipeline

    Returns:
        "pass" if evidence supports the claim, otherwise "fail"
    """
    if not result or not result.scored:
        return "fail"

    scores: List[EvidenceScore] = result.scored.scores or []

    if not scores:
        return "fail"

    _validate_scores(scores)

    metrics = [_score_snippet(s) for s in scores]

    strongest_support = max(m["support"] for m in metrics)
    strongest_contradiction = max(m["contradict"] for m in metrics)
    best_margin = max(m["margin"] for m in metrics)

    moderate_support_count = sum(1 for m in metrics if m["support"] >= 0.60)
    strong_support_count = sum(1 for m in metrics if m["support"] >= 0.75)
    contradiction_count = sum(1 for m in metrics if m["contradict"] >= 0.60)

    total_support = sum(m["support"] for m in metrics)
    total_contradiction = sum(m["contradict"] for m in metrics)

    avg_margin = sum(m["margin"] for m in metrics) / len(metrics)

    # ---------- HARD FAIL CONDITIONS ----------
    if strongest_contradiction >= 0.90:
        return "fail"

    if contradiction_count >= 2 and total_contradiction > total_support:
        return "fail"

    # ---------- STRONG PASS CONDITIONS ----------
    if (
        strongest_support >= 0.85
        and best_margin >= 0.25
        and strongest_contradiction < 0.70
    ):
        return "pass"

    if (
        strong_support_count >= 2
        and total_support > total_contradiction * 1.25
    ):
        return "pass"

    # ---------- AGGREGATE SCORING ----------
    aggregate_score = _aggregate_score(
        strongest_support=strongest_support,
        strongest_contradiction=strongest_contradiction,
        moderate_support_count=moderate_support_count,
        strong_support_count=strong_support_count,
        contradiction_count=contradiction_count,
        total_support=total_support,
        total_contradiction=total_contradiction,
        avg_margin=avg_margin,
        label=result.label,
        snippet_count=len(metrics),
    )

    return "pass" if aggregate_score >= 0.55 else "fail"


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _score_snippet(score: EvidenceScore) -> dict:
    """
    Compute derived metrics for a snippet.

    Returns:
        dict with:
            support
            contradict
            neutral
            margin (support - contradict)
            confidence (gap between top two probabilities)
    """
    values = [score.support, score.contradict, score.neutral]
    ordered = sorted(values, reverse=True)

    return {
        "support": score.support,
        "contradict": score.contradict,
        "neutral": score.neutral,
        "margin": score.support - score.contradict,
        "confidence": ordered[0] - ordered[1],
    }


def _aggregate_score(
    *,
    strongest_support: float,
    strongest_contradiction: float,
    moderate_support_count: int,
    strong_support_count: int,
    contradiction_count: int,
    total_support: float,
    total_contradiction: float,
    avg_margin: float,
    label: str,
    snippet_count: int,
) -> float:
    """
    Compute final heuristic score in [0, 1].

    Higher means stronger evidence alignment.
    """
    score = 0.50

    # Strongest signals
    score += 0.25 * strongest_support
    score -= 0.30 * strongest_contradiction

    # Repetition of support
    score += min(moderate_support_count, 3) * 0.05
    score += min(strong_support_count, 2) * 0.07

    # Repetition of contradiction
    score -= min(contradiction_count, 3) * 0.08

    # Overall balance
    if snippet_count > 0:
        score += 0.20 * ((total_support - total_contradiction) / snippet_count)

    # Margin consistency
    score += 0.20 * avg_margin

    # Weak label bias
    score += _label_bias(label)

    return _clamp(score, 0.0, 1.0)


def _label_bias(label: str) -> float:
    """
    Small adjustment based on pipeline label.
    """
    normalized = (label or "").strip().lower()

    if normalized in {"supported", "true", "entails", "pass"}:
        return 0.04

    if normalized in {"contradicted", "false", "fail", "refuted"}:
        return -0.04

    return 0.0


def _validate_scores(scores: List[EvidenceScore]) -> None:
    """
    Ensure all probabilities are within [0, 1].
    """
    for s in scores:
        for val in (s.support, s.contradict, s.neutral):
            if val < 0.0 or val > 1.0:
                raise ValueError("EvidenceScore values must be between 0 and 1")


def _clamp(value: float, low: float, high: float) -> float:
    """
    Clamp value to [low, high].
    """
    return max(low, min(value, high))