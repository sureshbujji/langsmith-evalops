"""Evaluators: exact-match scorer, 1-5 rubric scorer, latency/regression detector.

These are deterministic, dependency-free scorers used by the offline pipeline.
``rubric_score`` is a transparent heuristic (reference keyword recall mapped to
a 1-5 scale) -- it stands in for an LLM-as-judge rubric so the whole suite runs
with no API key. Swap in a real judge behind the same function signature when
you point the adapter at LangSmith.
"""

from __future__ import annotations

import re
from typing import Dict, List

_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the
    this to was were will with one two three what which when how why do does
    did not no yes true false
    """.split()
)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for comparison."""
    return " ".join(_WORD_RE.findall(text.lower()))


def exact_match_score(predicted: str, reference: str) -> float:
    """1.0 if normalized prediction equals normalized reference, else 0.0."""
    return 1.0 if normalize(predicted) == normalize(reference) else 0.0


def _keywords(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def token_recall(predicted: str, reference: str) -> float:
    """Fraction of reference keywords present in the prediction (0.0-1.0)."""
    ref_tokens = _keywords(reference)
    if not ref_tokens:
        return 1.0
    pred_tokens = set(_WORD_RE.findall(predicted.lower()))
    hits = sum(1 for t in ref_tokens if t in pred_tokens)
    return hits / len(ref_tokens)


def rubric_score(predicted: str, reference: str) -> float:
    """Heuristic 1-5 rubric score.

    Maps reference keyword recall onto a 1-5 scale:
      5 = fully covers the reference answer
      3 = partially covers it
      1 = misses it entirely / empty
    Empty predictions always score 1.0. Deterministic and explainable --
    a stand-in for an LLM judge in the offline path.
    """
    if not predicted or not predicted.strip():
        return 1.0
    recall = token_recall(predicted, reference)
    return round(1.0 + 4.0 * recall, 2)


def latency_ok(latency_ms: float, budget_ms: float) -> bool:
    """True when a run completed within the latency budget."""
    return latency_ms <= budget_ms


def detect_regressions(
    baseline_scores: Dict[str, float],
    candidate_scores: Dict[str, float],
    baseline_latency_ms: float,
    candidate_latency_ms: float,
    *,
    rubric_drop_threshold: float = 1.0,
    latency_budget_ms: float = 1500.0,
) -> List[Dict[str, float | str]]:
    """Compare one example's baseline vs candidate run; list regressions.

    A regression is reported when:
      * exact_match drops from 1.0 to 0.0 (correctness regression), or
      * rubric score drops by >= rubric_drop_threshold (quality regression), or
      * candidate latency exceeds the budget while the baseline was within it
        (or latency jumped by >500ms past the budget).
    """
    regressions: List[Dict[str, float | str]] = []

    b_exact = float(baseline_scores.get("exact_match", 0.0))
    c_exact = float(candidate_scores.get("exact_match", 0.0))
    if b_exact >= 1.0 > c_exact:
        regressions.append(
            {
                "metric": "exact_match",
                "baseline": b_exact,
                "candidate": c_exact,
                "delta": round(c_exact - b_exact, 3),
                "severity": "high",
                "reason": "previously exact answer is now wrong",
            }
        )

    b_rub = float(baseline_scores.get("rubric", 1.0))
    c_rub = float(candidate_scores.get("rubric", 1.0))
    rub_drop = b_rub - c_rub
    if rub_drop >= rubric_drop_threshold:
        regressions.append(
            {
                "metric": "rubric",
                "baseline": b_rub,
                "candidate": c_rub,
                "delta": round(c_rub - b_rub, 3),
                "severity": "high" if rub_drop >= 2.0 else "medium",
                "reason": f"rubric score dropped by {rub_drop:.2f} "
                f"(threshold {rubric_drop_threshold})",
            }
        )

    lat_delta = candidate_latency_ms - baseline_latency_ms
    if not latency_ok(candidate_latency_ms, latency_budget_ms) and (
        latency_ok(baseline_latency_ms, latency_budget_ms) or lat_delta > 500.0
    ):
        regressions.append(
            {
                "metric": "latency_ms",
                "baseline": round(baseline_latency_ms, 1),
                "candidate": round(candidate_latency_ms, 1),
                "delta": round(lat_delta, 1),
                "severity": "medium",
                "reason": f"latency {candidate_latency_ms:.0f}ms exceeds "
                f"budget {latency_budget_ms:.0f}ms",
            }
        )

    return regressions
