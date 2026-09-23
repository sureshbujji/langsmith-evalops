"""End-to-end evaluation pipeline.

    python -m src.run_experiment

Seeds the golden dataset, runs prompt v1 (baseline) and v2 (candidate) as two
experiments, scores every run with the evaluators, diffs the experiments, and
writes ``reports/regression_report.md``.

The pipeline talks to ``adapter.get_client()`` only, so the exact same code
path runs offline (SQLite ``EvalStore``) or against real LangSmith -- the
switch is the ``LANGSMITH_API_KEY`` environment variable, nothing else.

Exit code is 0 on success. A detected regression is an expected, reported
outcome (that's the point of the demo), not a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adapter import get_client  # noqa: E402
from src.evaluators import exact_match_score, rubric_score  # noqa: E402
from src.prompts import PROMPT_VERSIONS, simulate_response  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "qa_tasks.jsonl"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "regression_report.md"

LATENCY_BUDGET_MS = 1500.0
RUBRIC_DROP_THRESHOLD = 1.0


def load_tasks(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def run_version(
    client: Any, dataset_id: Any, example_ids: Dict[str, Any], version: str
) -> Any:
    """Run one prompt version over every example; return the experiment id."""
    info = PROMPT_VERSIONS[version]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_id = client.create_experiment(
        dataset_id,
        name=f"{info['name']}-{stamp}",
        version=version,
        metadata={"system_prompt": info["system_prompt"]},
    )
    for task_id, example_id in example_ids.items():
        task = next(t for t in load_tasks() if t["task_id"] == task_id)
        output, latency_ms = simulate_response(task, version)
        run_id = client.log_run(
            experiment_id,
            example_id,
            output,
            latency_ms=latency_ms,
            tokens_in=len(task["input"].split()),
            tokens_out=len(output.split()),
        )
        client.attach_scores(
            run_id,
            {
                "exact_match": exact_match_score(output, task["reference"]),
                "rubric": rubric_score(output, task["reference"]),
            },
        )
    return experiment_id


def render_report(
    comparison: Dict[str, Any], backend: str, versions: List[str]
) -> str:
    agg = comparison["aggregates"]
    summary = comparison["summary"]
    lines: List[str] = []
    a = lines.append

    a("# Experiment Regression Report")
    a("")
    a(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Backend: `{backend}`_")
    a("")
    verdict = summary["verdict"]
    badge = "🔴" if "REGRESSION" in verdict else "🟢"
    a(f"## Verdict: {badge} {verdict}")
    a("")
    a(
        f"Compared **{summary['n_examples']}** golden tasks: "
        f"baseline `{comparison['baseline']['name']}` "
        f"(prompt {versions[0]}) vs candidate "
        f"`{comparison['candidate']['name']}` (prompt {versions[1]}). "
        f"**{summary['n_regressions']} regression(s)** detected."
    )
    a("")
    a("### Configuration")
    a("")
    a(f"* Latency budget: `{comparison['thresholds']['latency_budget_ms']:.0f}ms`")
    a(
        f"* Rubric drop threshold: "
        f"`{comparison['thresholds']['rubric_drop_threshold']}`"
    )
    a("")

    a("### Aggregate scores")
    a("")
    a("| Metric | Baseline (v1) | Candidate (v2) | Delta |")
    a("|---|---|---|---|")
    for metric, label in (
        ("mean_exact_match", "Mean exact-match"),
        ("mean_rubric", "Mean rubric (1-5)"),
        ("mean_latency_ms", "Mean latency (ms)"),
    ):
        b = agg["baseline"][metric]
        c = agg["candidate"][metric]
        d = agg["delta"][metric]
        flag = " ⚠️" if (metric != "mean_latency_ms" and d < 0) or (
            metric == "mean_latency_ms" and d > 0
        ) else ""
        a(f"| {label} | {b} | {c} | {d:+}{flag} |")
    a("")

    a("### Regressions caught")
    a("")
    if not comparison["regressions"]:
        a("None -- candidate is at parity with or better than baseline.")
    else:
        a("| Task | Metric | Baseline | Candidate | Delta | Severity | Reason |")
        a("|---|---|---|---|---|---|---|")
        for r in comparison["regressions"]:
            a(
                f"| `{r['task_id']}` | {r['metric']} | {r['baseline']} | "
                f"{r['candidate']} | {r['delta']:+} | {r['severity']} | "
                f"{r['reason']} |"
            )
    a("")

    a("### Per-task detail")
    a("")
    a(
        "| Task | Category | Exact Δ | Rubric Δ | Latency Δ (ms) | "
        "Regressed |"
    )
    a("|---|---|---|---|---|---|")
    for row in comparison["per_example"]:
        d = row["deltas"]
        mark = "❌" if row["regressed"] else "✅"
        a(
            f"| `{row['task_id']}` | {row['category']} | {d['exact_match']:+} | "
            f"{d['rubric']:+} | {d['latency_ms']:+} | {mark} |"
        )
    a("")
    a("---")
    a(
        "_Offline-first demo: the same pipeline runs against real LangSmith "
        "by setting `LANGSMITH_API_KEY` -- see `src/adapter.py` and the "
        "README's 'Offline vs live' section._"
    )
    a("")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the v1 vs v2 evaluation pipeline and write a "
        "regression report."
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite file for the offline store "
        "(default: $EVALOPS_DB or ./evalstore.db; ignored for live backend)",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Where to write the markdown regression report.",
    )
    parser.add_argument(
        "--dataset-name",
        default="qa-golden-tasks",
        help="Dataset name (reused across runs).",
    )
    args = parser.parse_args(argv)

    live = bool(os.environ.get("LANGSMITH_API_KEY"))
    if args.db_path and not live:
        from src.evalstore import EvalStore

        client = EvalStore(db_path=args.db_path)
        backend = "EvalStore (SQLite, offline)"
    else:
        client = get_client()
        backend = (
            "LangSmith (live)"
            if live
            else "EvalStore (SQLite, offline)"
        )

    versions = ["v1", "v2"]
    tasks = load_tasks()
    dataset_id = client.get_or_create_dataset(
        args.dataset_name,
        "Golden QA task set for prompt regression tracking.",
    )
    example_ids_list = client.create_examples(dataset_id, tasks)
    example_ids = {
        t["task_id"]: eid for t, eid in zip(tasks, example_ids_list)
    }

    experiment_ids = [run_version(client, dataset_id, example_ids, v) for v in versions]
    comparison = client.compare_experiments(
        experiment_ids[0],
        experiment_ids[1],
        rubric_drop_threshold=RUBRIC_DROP_THRESHOLD,
        latency_budget_ms=LATENCY_BUDGET_MS,
    )

    report = render_report(comparison, backend, versions)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    summary = comparison["summary"]
    print(f"backend:        {backend}")
    print(f"dataset:        {args.dataset_name} ({len(tasks)} tasks)")
    print(f"baseline:       {comparison['baseline']['name']}")
    print(f"candidate:      {comparison['candidate']['name']}")
    print(f"verdict:        {summary['verdict']}")
    print(f"regressions:    {summary['n_regressions']}")
    for r in comparison["regressions"]:
        print(
            f"  - {r['task_id']} [{r['metric']}] "
            f"{r['baseline']} -> {r['candidate']} ({r['severity']})"
        )
    print(f"report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
