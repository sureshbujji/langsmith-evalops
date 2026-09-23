"""Tests for the SQLite EvalStore: datasets -> examples -> runs -> scores."""

import pytest

from src.evalstore import EvalStore


@pytest.fixture()
def store():
    s = EvalStore(":memory:")
    yield s
    s.close()


def _seed(store):
    ds = store.create_dataset("qa-golden", "golden tasks")
    ex_ids = store.create_examples(
        ds,
        [
            {"task_id": "t1", "input": "Capital of France?",
             "reference": "Paris", "category": "factual"},
            {"task_id": "t2", "input": "2+2?", "reference": "4",
             "category": "factual"},
        ],
    )
    return ds, ex_ids


def test_create_dataset_and_get_or_create(store):
    ds_id = store.create_dataset("d1", "desc")
    assert isinstance(ds_id, int)
    assert store.get_or_create_dataset("d1") == ds_id  # idempotent


def test_create_examples_round_trip(store):
    ds, ex_ids = _seed(store)
    assert len(ex_ids) == 2
    examples = store.list_examples(ds)
    assert [e["task_id"] for e in examples] == ["t1", "t2"]
    assert examples[0]["reference_text"] == "Paris"


def test_log_run_and_attach_scores(store):
    ds, ex_ids = _seed(store)
    exp = store.create_experiment(ds, "exp-v1", version="v1")
    run_id = store.log_run(exp, ex_ids[0], "Paris", latency_ms=123.0)
    store.attach_scores(run_id, {"exact_match": 1.0, "rubric": 5.0})
    runs = store.get_experiment_runs(exp)
    assert len(runs) == 1
    assert runs[0]["output_text"] == "Paris"
    assert runs[0]["latency_ms"] == 123.0
    assert runs[0]["scores"] == {"exact_match": 1.0, "rubric": 5.0}


def test_compare_experiments_finds_regression(store):
    ds, ex_ids = _seed(store)
    exp_a = store.create_experiment(ds, "baseline", version="v1")
    exp_b = store.create_experiment(ds, "candidate", version="v2")
    # baseline: both correct; candidate: t2 wrong + slower
    for exp, outputs in (
        (exp_a, ["Paris", "4"]),
        (exp_b, ["Paris", "5"]),
    ):
        for eid, out in zip(ex_ids, outputs):
            rid = store.log_run(exp, eid, out, latency_ms=300.0)
            store.attach_scores(
                rid,
                {
                    "exact_match": 1.0 if out in ("Paris", "4") else 0.0,
                    "rubric": 5.0 if out in ("Paris", "4") else 1.0,
                },
            )
    comp = store.compare_experiments(exp_a, exp_b)
    assert comp["summary"]["n_examples"] == 2
    assert comp["summary"]["verdict"] == "REGRESSION DETECTED"
    metrics = {(r["task_id"], r["metric"]) for r in comp["regressions"]}
    assert ("t2", "exact_match") in metrics
    assert ("t2", "rubric") in metrics
    assert comp["aggregates"]["delta"]["mean_exact_match"] == pytest.approx(-0.5)


def test_compare_experiments_no_regression(store):
    ds, ex_ids = _seed(store)
    exp_a = store.create_experiment(ds, "baseline", version="v1")
    exp_b = store.create_experiment(ds, "candidate", version="v2")
    for exp in (exp_a, exp_b):
        for eid, out in zip(ex_ids, ["Paris", "4"]):
            rid = store.log_run(exp, eid, out, latency_ms=300.0)
            store.attach_scores(rid, {"exact_match": 1.0, "rubric": 5.0})
    comp = store.compare_experiments(exp_a, exp_b)
    assert comp["summary"]["verdict"] == "NO REGRESSION"
    assert comp["regressions"] == []
