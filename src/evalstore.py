"""Offline evaluation store (SQLite, stdlib only).

EvalStore mirrors the subset of the LangSmith client interface this repo uses:

    create_dataset(name, description="")      -> dataset id
    create_examples(dataset_id, examples)     -> list of example ids
    create_experiment(dataset_id, name, ...)  -> experiment id
    log_run(experiment_id, example_id, ...)   -> run id
    attach_scores(run_id, scores)             -> None
    compare_experiments(baseline_id, candidate_id, ...) -> comparison dict

Because the method signatures match ``adapter.LangSmithAdapter``, the same
pipeline code in ``run_experiment.py`` works unchanged against real LangSmith
once ``LANGSMITH_API_KEY`` is set. Everything here runs on stdlib sqlite3:
no network, no API key, no third-party dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .evaluators import detect_regressions

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS examples (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id     INTEGER NOT NULL REFERENCES datasets(id),
    task_id        TEXT NOT NULL,
    input_text     TEXT NOT NULL,
    reference_text TEXT NOT NULL,
    category       TEXT DEFAULT '',
    UNIQUE (dataset_id, task_id)
);
CREATE TABLE IF NOT EXISTS experiments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    name       TEXT NOT NULL,
    version    TEXT DEFAULT '',
    metadata   TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    example_id    INTEGER NOT NULL REFERENCES examples(id),
    output_text   TEXT NOT NULL,
    latency_ms    REAL DEFAULT 0.0,
    tokens_in     INTEGER DEFAULT 0,
    tokens_out    INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name   TEXT NOT NULL,
    value  REAL NOT NULL,
    UNIQUE (run_id, name)
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class EvalStore:
    """SQLite-backed evaluation store with a LangSmith-like interface."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------ #
    # datasets / examples
    # ------------------------------------------------------------------ #
    def create_dataset(self, name: str, description: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO datasets (name, description, created_at) VALUES (?, ?, ?)",
            (name, description, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_dataset_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM datasets WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_or_create_dataset(self, name: str, description: str = "") -> int:
        existing = self.get_dataset_by_name(name)
        if existing:
            return int(existing["id"])
        return self.create_dataset(name, description)

    def create_examples(
        self, dataset_id: int, examples: List[Dict[str, Any]]
    ) -> List[int]:
        """Add examples. Each example: {task_id, input, reference, category?}."""
        ids: List[int] = []
        for ex in examples:
            cur = self.conn.execute(
                """INSERT OR REPLACE INTO examples
                   (dataset_id, task_id, input_text, reference_text, category)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    dataset_id,
                    ex["task_id"],
                    ex["input"],
                    ex["reference"],
                    ex.get("category", ""),
                ),
            )
            ids.append(int(cur.lastrowid))
        self.conn.commit()
        return ids

    def list_examples(self, dataset_id: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM examples WHERE dataset_id = ? ORDER BY id", (dataset_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # experiments / runs / scores
    # ------------------------------------------------------------------ #
    def create_experiment(
        self,
        dataset_id: int,
        name: str,
        version: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO experiments (dataset_id, name, version, metadata, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (dataset_id, name, version, json.dumps(metadata or {}), _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_run(
        self,
        experiment_id: int,
        example_id: int,
        output: str,
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO runs
               (experiment_id, example_id, output_text, latency_ms,
                tokens_in, tokens_out, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                example_id,
                output,
                float(latency_ms),
                int(tokens_in),
                int(tokens_out),
                _now(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def attach_scores(self, run_id: int, scores: Dict[str, float]) -> None:
        """Attach evaluator scores, e.g. {"exact_match": 1.0, "rubric": 4.5}."""
        for name, value in scores.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO scores (run_id, name, value)
                   VALUES (?, ?, ?)""",
                (run_id, name, float(value)),
            )
        self.conn.commit()

    def get_experiment_runs(self, experiment_id: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT r.id AS run_id, r.output_text, r.latency_ms,
                      r.tokens_in, r.tokens_out,
                      e.id AS example_id, e.task_id, e.input_text,
                      e.reference_text, e.category
               FROM runs r
               JOIN examples e ON e.id = r.example_id
               WHERE r.experiment_id = ?
               ORDER BY e.id""",
            (experiment_id,),
        ).fetchall()
        runs: List[Dict[str, Any]] = []
        for row in rows:
            run = dict(row)
            score_rows = self.conn.execute(
                "SELECT name, value FROM scores WHERE run_id = ?",
                (run["run_id"],),
            ).fetchall()
            run["scores"] = {s["name"]: s["value"] for s in score_rows}
            runs.append(run)
        return runs

    # ------------------------------------------------------------------ #
    # comparison / regression tracking
    # ------------------------------------------------------------------ #
    def compare_experiments(
        self,
        baseline_experiment_id: int,
        candidate_experiment_id: int,
        *,
        rubric_drop_threshold: float = 1.0,
        latency_budget_ms: float = 1500.0,
    ) -> Dict[str, Any]:
        """Compare two experiments run over the same dataset.

        Returns per-example deltas, aggregate stats, and a list of detected
        regressions. This is the offline equivalent of diffing two LangSmith
        experiments run over a shared dataset.
        """
        baseline_runs = {
            r["task_id"]: r for r in self.get_experiment_runs(baseline_experiment_id)
        }
        candidate_runs = {
            r["task_id"]: r for r in self.get_experiment_runs(candidate_experiment_id)
        }
        baseline_exp = self._get_experiment(baseline_experiment_id)
        candidate_exp = self._get_experiment(candidate_experiment_id)

        task_ids = sorted(set(baseline_runs) & set(candidate_runs))
        per_example: List[Dict[str, Any]] = []
        regressions: List[Dict[str, Any]] = []

        for task_id in task_ids:
            b, c = baseline_runs[task_id], candidate_runs[task_id]
            b_scores = {
                "exact_match": float(b["scores"].get("exact_match", 0.0)),
                "rubric": float(b["scores"].get("rubric", 1.0)),
            }
            c_scores = {
                "exact_match": float(c["scores"].get("exact_match", 0.0)),
                "rubric": float(c["scores"].get("rubric", 1.0)),
            }
            found = detect_regressions(
                b_scores,
                c_scores,
                float(b["latency_ms"]),
                float(c["latency_ms"]),
                rubric_drop_threshold=rubric_drop_threshold,
                latency_budget_ms=latency_budget_ms,
            )
            for reg in found:
                reg["task_id"] = task_id
                reg["category"] = b["category"]
                regressions.append(reg)
            per_example.append(
                {
                    "task_id": task_id,
                    "category": b["category"],
                    "baseline": {
                        "output": b["output_text"],
                        "latency_ms": b["latency_ms"],
                        "scores": b_scores,
                    },
                    "candidate": {
                        "output": c["output_text"],
                        "latency_ms": c["latency_ms"],
                        "scores": c_scores,
                    },
                    "deltas": {
                        "exact_match": round(
                            c_scores["exact_match"] - b_scores["exact_match"], 3
                        ),
                        "rubric": round(c_scores["rubric"] - b_scores["rubric"], 3),
                        "latency_ms": round(
                            float(c["latency_ms"]) - float(b["latency_ms"]), 1
                        ),
                    },
                    "regressed": bool(found),
                }
            )

        aggregates = {
            side: self._aggregate(list(runs.values()))
            for side, runs in (
                ("baseline", baseline_runs),
                ("candidate", candidate_runs),
            )
        }
        aggregates["delta"] = {
            k: round(
                aggregates["candidate"].get(k, 0.0)
                - aggregates["baseline"].get(k, 0.0),
                3,
            )
            for k in ("mean_exact_match", "mean_rubric", "mean_latency_ms")
        }

        return {
            "baseline": {
                "experiment_id": baseline_experiment_id,
                "name": baseline_exp["name"],
                "version": baseline_exp["version"],
            },
            "candidate": {
                "experiment_id": candidate_experiment_id,
                "name": candidate_exp["name"],
                "version": candidate_exp["version"],
            },
            "thresholds": {
                "rubric_drop_threshold": rubric_drop_threshold,
                "latency_budget_ms": latency_budget_ms,
            },
            "per_example": per_example,
            "aggregates": aggregates,
            "regressions": regressions,
            "summary": {
                "n_examples": len(task_ids),
                "n_regressions": len(regressions),
                "verdict": "REGRESSION DETECTED"
                if regressions
                else "NO REGRESSION",
            },
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _get_experiment(self, experiment_id: int) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"experiment {experiment_id} not found")
        return dict(row)

    @staticmethod
    def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(runs)
        if n == 0:
            return {
                "n": 0,
                "mean_exact_match": 0.0,
                "mean_rubric": 0.0,
                "mean_latency_ms": 0.0,
                "pass_rate": 0.0,
            }
        exact = [float(r["scores"].get("exact_match", 0.0)) for r in runs]
        rubric = [float(r["scores"].get("rubric", 1.0)) for r in runs]
        lat = [float(r["latency_ms"]) for r in runs]
        return {
            "n": n,
            "mean_exact_match": round(sum(exact) / n, 3),
            "mean_rubric": round(sum(rubric) / n, 3),
            "mean_latency_ms": round(sum(lat) / n, 1),
            "pass_rate": round(sum(1 for e in exact if e >= 1.0) / n, 3),
        }

    def close(self) -> None:
        self.conn.close()
