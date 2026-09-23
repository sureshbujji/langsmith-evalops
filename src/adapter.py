"""One switch between the offline store and real LangSmith.

``get_client()`` is the only entry point the pipeline uses:

    from src.adapter import get_client
    client = get_client()   # EvalStore (offline) or LangSmithAdapter (live)

Switch behavior:
    * ``LANGSMITH_API_KEY`` unset  -> :class:`EvalStore` (SQLite, stdlib only).
      Zero network, zero third-party deps. This is what CI runs.
    * ``LANGSMITH_API_KEY`` set    -> :class:`LangSmithAdapter`, which wraps
      ``langsmith.Client`` with the *same method signatures* as ``EvalStore``
      (``create_dataset``, ``create_examples``, ``create_experiment``,
      ``log_run``, ``attach_scores``, ``compare_experiments``).

Concept mapping (documented, because the two systems name things differently):

    EvalStore dataset      -> LangSmith dataset (client.create_dataset)
    EvalStore example      -> LangSmith example  (client.create_examples)
    EvalStore experiment   -> LangSmith project  (client.create_project);
                              LangSmith has no first-class "experiment" object
                              on the client, so each experiment becomes a
                              project named after the experiment.
    EvalStore run          -> LangSmith run      (client.create_run)
    EvalStore score        -> LangSmith feedback (client.create_feedback)
    compare_experiments    -> list_runs + list_feedback, then the same
                              evaluators.detect_regressions used offline.

The ``langsmith`` package is imported lazily inside ``LangSmithAdapter`` so
the offline path never requires it. Install it only for the live path::

    pip install "langsmith==0.14.0"   # see requirements.txt (optional section)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_client():
    """Return an EvalStore-compatible client.

    Offline (no ``LANGSMITH_API_KEY``): local SQLite ``EvalStore``.
    Live (``LANGSMITH_API_KEY`` set): ``LangSmithAdapter`` around the real
    LangSmith client. Identical method signatures either way.
    """
    if os.environ.get("LANGSMITH_API_KEY"):
        return LangSmithAdapter()
    from .evalstore import EvalStore

    db_path = os.environ.get("EVALOPS_DB") or str(REPO_ROOT / "evalstore.db")
    return EvalStore(db_path=db_path)


class LangSmithAdapter:
    """EvalStore-compatible facade over ``langsmith.Client``.

    Requires the ``langsmith`` package and ``LANGSMITH_API_KEY`` (plus
    optionally ``LANGSMITH_ENDPOINT`` / ``LANGSMITH_PROJECT``) in the
    environment. Method signatures intentionally mirror ``EvalStore`` so
    ``run_experiment.py`` runs unchanged against either backend.
    """

    def __init__(self) -> None:
        try:
            from langsmith import Client
        except ImportError as exc:
            raise RuntimeError(
                "LANGSMITH_API_KEY is set but the 'langsmith' package is not "
                "installed. Install it with: pip install langsmith"
            ) from exc
        self._client = Client()
        # experiment_id (project id str) -> project name, for run logging
        self._projects: Dict[str, str] = {}
        # example_id -> {"input": ...} cache so log_run can attach inputs
        self._examples: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # datasets / examples
    # ------------------------------------------------------------------ #
    def create_dataset(self, name: str, description: str = "") -> str:
        ds = self._client.create_dataset(
            dataset_name=name, description=description or None
        )
        return str(ds.id)

    def get_or_create_dataset(self, name: str, description: str = "") -> str:
        for ds in self._client.list_datasets(dataset_name=name):
            return str(ds.id)
        return self.create_dataset(name, description)

    def create_examples(
        self, dataset_id: str, examples: List[Dict[str, Any]]
    ) -> List[str]:
        """Each example: {task_id, input, reference, category?}."""
        payload = [
            {
                "inputs": {"input": ex["input"], "task_id": ex["task_id"]},
                "outputs": {
                    "reference": ex["reference"],
                    "category": ex.get("category", ""),
                },
            }
            for ex in examples
        ]
        self._client.create_examples(dataset_id=dataset_id, examples=payload)
        # create_examples is a bulk upsert without per-example ids in the
        # response; resolve ids by listing so later calls can reference them.
        ids: List[str] = []
        for ex in self._client.list_examples(dataset_id=dataset_id):
            ex_id = str(ex.id)
            ids.append(ex_id)
            self._examples[ex_id] = {"input": ex.inputs.get("input", "")}
        return ids

    # ------------------------------------------------------------------ #
    # experiments / runs / scores
    # ------------------------------------------------------------------ #
    def create_experiment(
        self,
        dataset_id: str,
        name: str,
        version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Map an experiment onto a LangSmith project (same signature)."""
        project = self._client.create_project(
            project_name=name,
            description=f"experiment version={version} dataset={dataset_id}",
            metadata_=metadata or {},
        )
        project_id = str(project.id)
        self._projects[project_id] = name
        return project_id

    def log_run(
        self,
        experiment_id: str,
        example_id: str,
        output: str,
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> str:
        run_id = str(uuid.uuid4())
        example = self._examples.get(example_id, {})
        self._client.create_run(
            id=run_id,
            name=f"run-{example_id}",
            inputs={"input": example.get("input", "")},
            outputs={"output": output},
            run_type="llm",
            project_name=self._projects.get(experiment_id),
            extra={
                "metadata": {
                    "latency_ms": latency_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }
            },
        )
        return run_id

    def attach_scores(self, run_id: str, scores: Dict[str, float]) -> None:
        for name, value in scores.items():
            self._client.create_feedback(
                run_id=run_id, key=name, score=float(value)
            )

    # ------------------------------------------------------------------ #
    # comparison (same shape as EvalStore.compare_experiments)
    # ------------------------------------------------------------------ #
    def compare_experiments(
        self,
        baseline_experiment_id: str,
        candidate_experiment_id: str,
        *,
        rubric_drop_threshold: float = 1.0,
        latency_budget_ms: float = 1500.0,
    ) -> Dict[str, Any]:
        from .evaluators import detect_regressions

        def _runs(experiment_id: str) -> Dict[str, Dict[str, Any]]:
            project = self._projects.get(experiment_id, experiment_id)
            out: Dict[str, Dict[str, Any]] = {}
            for run in self._client.list_runs(project_name=project):
                feedback = {
                    f.key: f.score
                    for f in self._client.list_feedback(run_ids=[run.id])
                    if f.score is not None
                }
                meta = (run.extra or {}).get("metadata", {}) or {}
                task_id = (run.inputs or {}).get("task_id", str(run.id))
                out[task_id] = {
                    "output": (run.outputs or {}).get("output", ""),
                    "latency_ms": float(meta.get("latency_ms", 0.0)),
                    "scores": {
                        "exact_match": float(feedback.get("exact_match", 0.0)),
                        "rubric": float(feedback.get("rubric", 1.0)),
                    },
                }
            return out

        baseline = _runs(baseline_experiment_id)
        candidate = _runs(candidate_experiment_id)
        regressions: List[Dict[str, Any]] = []
        for task_id in sorted(set(baseline) & set(candidate)):
            b, c = baseline[task_id], candidate[task_id]
            for reg in detect_regressions(
                b["scores"],
                c["scores"],
                b["latency_ms"],
                c["latency_ms"],
                rubric_drop_threshold=rubric_drop_threshold,
                latency_budget_ms=latency_budget_ms,
            ):
                reg["task_id"] = task_id
                regressions.append(reg)
        return {
            "baseline": {"experiment_id": baseline_experiment_id},
            "candidate": {"experiment_id": candidate_experiment_id},
            "thresholds": {
                "rubric_drop_threshold": rubric_drop_threshold,
                "latency_budget_ms": latency_budget_ms,
            },
            "regressions": regressions,
            "summary": {
                "n_examples": len(set(baseline) & set(candidate)),
                "n_regressions": len(regressions),
                "verdict": "REGRESSION DETECTED" if regressions else "NO REGRESSION",
            },
        }
