"""langsmith-evalops: offline-first LangSmith evaluation operations.

Modules:
    evalstore   SQLite-backed store mirroring the LangSmith client interface.
    evaluators  Exact-match, 1-5 rubric, and latency/regression evaluators.
    prompts     Prompt versions and deterministic simulated model responses.
    adapter     Env-var switch between EvalStore and real LangSmith.
    run_experiment  End-to-end pipeline: dataset -> experiments -> report.
"""

__version__ = "0.1.0"
