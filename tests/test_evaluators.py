"""Tests for the evaluators: exact match, rubric, latency, regression detection."""

from src import evaluators


def test_exact_match_normalizes():
    assert evaluators.exact_match_score("Paris", "paris") == 1.0
    assert evaluators.exact_match_score("  Paris! ", "paris") == 1.0
    assert evaluators.exact_match_score("Lyon", "Paris") == 0.0


def test_rubric_bounds_and_monotonicity():
    ref = "Timing issues and shared test state."
    full = "Two common causes are timing issues and shared test state."
    partial = "A test plan defines scope and schedule for testing activities."
    empty = ""
    s_full = evaluators.rubric_score(full, ref)
    s_partial = evaluators.rubric_score(partial, ref)
    assert s_full == 5.0
    assert 1.0 <= s_partial <= 5.0
    assert s_full > s_partial
    assert evaluators.rubric_score(empty, ref) == 1.0


def test_latency_ok():
    assert evaluators.latency_ok(1500.0, 1500.0)
    assert not evaluators.latency_ok(1500.1, 1500.0)


def _scores(exact, rubric):
    return {"exact_match": exact, "rubric": rubric}


def test_detect_exact_match_regression():
    regs = evaluators.detect_regressions(
        _scores(1.0, 5.0), _scores(0.0, 2.0), 300.0, 350.0
    )
    metrics = {r["metric"] for r in regs}
    assert "exact_match" in metrics
    assert "rubric" in metrics  # 3.0 drop >= 1.0 threshold
    assert all(r["severity"] == "high" for r in regs)


def test_detect_latency_regression_only():
    regs = evaluators.detect_regressions(
        _scores(1.0, 5.0), _scores(1.0, 5.0), 840.0, 2450.0,
        latency_budget_ms=1500.0,
    )
    assert [r["metric"] for r in regs] == ["latency_ms"]
    assert regs[0]["severity"] == "medium"


def test_no_regression_within_threshold():
    # rubric dip of 0.5 is below the 1.0 threshold: no regression
    regs = evaluators.detect_regressions(
        _scores(1.0, 5.0), _scores(1.0, 4.5), 300.0, 320.0
    )
    assert regs == []


def test_seeded_v2_regressions_are_caught():
    """The seeded prompt versions must trip the detector on task_05/09/03."""
    from src.prompts import simulate_response

    cases = {
        # task_id -> (reference, expect_regression)
        "task_05": ("Software", True),   # wrong answer
        "task_09": ("Timing issues and shared test state.", True),  # vague answer
        "task_03": (
            "To catch unintended behavior changes quickly before they "
            "reach production.", True),  # latency blowout
        "task_01": ("Paris", False),     # control: unchanged
    }
    for task_id, (reference, expect) in cases.items():
        task = {"task_id": task_id}
        out_b, lat_b = simulate_response(task, "v1")
        out_c, lat_c = simulate_response(task, "v2")
        regs = evaluators.detect_regressions(
            {"exact_match": evaluators.exact_match_score(out_b, reference),
             "rubric": evaluators.rubric_score(out_b, reference)},
            {"exact_match": evaluators.exact_match_score(out_c, reference),
             "rubric": evaluators.rubric_score(out_c, reference)},
            lat_b, lat_c,
        )
        assert bool(regs) is expect, f"{task_id}: {regs}"
