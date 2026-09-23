# langsmith-evalops

Offline-first evaluation operations for LLM prompts: datasets → experiments →
evaluators → regression tracking, with a one-env-var switch to real LangSmith.

I built this to demonstrate the full LangSmith evaluation workflow the way I
run it as a QA lead: a golden dataset, two prompt versions run as experiments,
deterministic evaluators scoring every run, and an automated diff that catches
regressions before a prompt ships. Everything runs locally with zero API keys
and zero network — CI included — and the exact same code path points at real
LangSmith when `LANGSMITH_API_KEY` is set.

## Architecture

```
                        ┌──────────────────────────────┐
                        │        run_experiment.py      │
                        │  pipeline: seed → run →       │
                        │  score → compare → report     │
                        └──────────────┬───────────────┘
                                       │  get_client()
                    ┌──────────────────┴───────────────────┐
                    │           src/adapter.py               │
                    │  LANGSMITH_API_KEY unset → EvalStore  │
                    │  LANGSMITH_API_KEY set   → LangSmith  │
                    └──────┬───────────────────────┬───────┘
                           │                       │
              ┌────────────▼──────────┐  ┌──────────▼──────────────┐
              │  EvalStore (SQLite)   │  │ LangSmithAdapter        │
              │  stdlib sqlite3,      │  │ wraps langsmith.Client  │
              │  zero deps, zero      │  │ (lazy import)           │
              │  network              │  │                         │
              └────────────┬──────────┘  └──────────┬──────────────┘
                           │  identical signatures │
                           └──────────┬────────────┘
                                      │
        ┌─────────────┬───────────────┼────────────────┬──────────────┐
        │             │               │                │              │
   create_dataset  create_examples  log_run   attach_scores  compare_experiments
        │             │               │                │              │
        └─────────────┴───────┬───────┴────────────────┴──────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  src/evaluators.py │
                    │  • exact_match     │
                    │  • rubric (1-5)    │
                    │  • latency /       │
                    │    regression      │
                    │    detector        │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────────────────┐
                    │ reports/regression_report.md   │
                    │ verdict + per-task deltas      │
                    └────────────────────────────────┘

  data/qa_tasks.jsonl ──▶ 12 golden tasks ──▶ prompt v1 (baseline)
                                              prompt v2 (candidate, seeded
                                              to regress on 3 tasks)
```

## Quickstart

```bash
# 1. Clone and enter
git clone <your-fork> langsmith-evalops && cd langsmith-evalops

# 2. Install (only pytest; the offline path needs nothing else)
pip install -r requirements.txt

# 3. Run the full pipeline -- no API key, no network
python -m src.run_experiment

# 4. Run the tests
python -m pytest -q
```

Sample console output from step 3:

```
backend:        EvalStore (SQLite, offline)
dataset:        qa-golden-tasks (12 tasks)
baseline:       qa-assistant-v1-20260923-040757
candidate:      qa-assistant-v2-20260923-040759
verdict:        REGRESSION DETECTED
regressions:    4
  - task_03 [latency_ms] 840.0 -> 2450.0 (medium)
  - task_05 [exact_match] 1.0 -> 0.0 (high)
  - task_05 [rubric] 5.0 -> 1.0 (high)
  - task_09 [rubric] 5.0 -> 1.0 (high)
report written: /home/hatch/workspace/github-projects/langsmith-evalops/reports/regression_report.md
```

## Sample output

The pipeline writes `reports/regression_report.md`. Here's the key section from a
real run — the v2 prompt was seeded to regress, and the detector caught it:

| Task | Metric | Baseline | Candidate | Delta | Severity | Reason |
|---|---|---|---|---|---|---|
| `task_03` | latency_ms | 840.0 | 2450.0 | +1610.0 | medium | latency 2450ms exceeds budget 1500ms |
| `task_05` | exact_match | 1.0 | 0.0 | -1.0 | high | previously exact answer is now wrong |
| `task_05` | rubric | 5.0 | 1.0 | -4.0 | high | rubric score dropped by 4.00 (threshold 1.0) |
| `task_09` | rubric | 5.0 | 1.0 | -4.0 | high | rubric score dropped by 4.00 (threshold 1.0) |

What each seeded regression represents:

- **task_05** — the v2 prompt answered "System" instead of "Software" for what
  the "S" in SDET stands for. Exact-match caught it instantly.
- **task_09** — v2 returned a vague non-answer ("Sometimes tests are flaky for
  random reasons.") instead of the two reference causes. The rubric scorer
  collapsed from 5.0 to 1.0.
- **task_03** — v2's answer was correct, but its chatty style blew the 1500ms
  latency budget (2450ms). The latency detector flagged it even though quality
  scores were unchanged — exactly the kind of silent regression that ships when
  you only watch accuracy.

## Offline vs live: the honest breakdown

**What runs offline (default, zero config):**

- `EvalStore` — a SQLite database (stdlib `sqlite3`) implementing
  `create_dataset`, `create_examples`, `create_experiment`, `log_run`,
  `attach_scores`, and `compare_experiments`.
- All three evaluators (exact-match, 1–5 rubric, latency/regression detector).
- The full pipeline and the entire pytest suite, including CI.
- Model responses are **deterministic simulated stand-ins** in
  `src/prompts.py` (`simulate_response(task, version) -> (output, latency_ms)`),
  and the rubric scorer is a **transparent heuristic** (reference keyword
  recall mapped to 1–5), not an LLM judge. That's deliberate: it keeps the demo
  reproducible with no model API. Swap `simulate_response` for your LLM call —
  the pipeline only depends on the `(output, latency_ms)` contract.

**The exact switch to real LangSmith:**

```bash
export LANGSMITH_API_KEY="lsv2_..."   # that's it -- nothing else changes
python -m src.run_experiment
```

`src/adapter.py::get_client()` checks `LANGSMITH_API_KEY`. Unset → local
`EvalStore`. Set → `LangSmithAdapter`, which wraps `langsmith.Client` with
**identical method signatures** (`create_dataset`, `create_examples`,
`create_experiment`, `log_run`, `attach_scores`, `compare_experiments`).
`run_experiment.py` calls only those methods, so the pipeline is backend-blind.
The `langsmith` package is imported lazily inside the adapter — the offline
path never requires it (it's an optional, pinned line in `requirements.txt`).

Concept mapping the adapter documents (LangSmith names things differently):

| EvalStore concept | LangSmith equivalent |
|---|---|
| dataset | dataset (`client.create_dataset`) |
| example | example (`client.create_examples`) |
| experiment | project (`client.create_project`) — LangSmith has no first-class client-side "experiment" object, so each experiment becomes a project |
| run | run (`client.create_run`) |
| score | feedback (`client.create_feedback`) |
| `compare_experiments` | `list_runs` + `list_feedback`, then the same `detect_regressions` evaluator |

## Project layout

```
langsmith-evalops/
├── src/
│   ├── evalstore.py      # SQLite EvalStore mirroring the LangSmith interface
│   ├── evaluators.py     # exact-match, 1-5 rubric, latency/regression detector
│   ├── prompts.py        # prompt v1/v2 + deterministic simulated responses
│   ├── adapter.py        # LANGSMITH_API_KEY switch; LangSmithAdapter facade
│   └── run_experiment.py # pipeline: seed → run → score → compare → report
├── tests/                # pytest suite (store, evaluators, adapter, pipeline)
├── data/qa_tasks.jsonl   # 12 golden QA tasks
├── reports/regression_report.md  # sample output from a real run
├── .github/workflows/ci.yml      # pytest + mock-mode smoke test
├── requirements.txt      # pytest required; langsmith optional (pinned)
├── .env.example          # placeholder keys only
└── LICENSE               # MIT
```

## Roadmap

- [ ] LLM-as-judge rubric option behind the same `rubric_score` signature
      (used automatically when `LANGSMITH_API_KEY` is set).
- [ ] Threshold config file (`evalops.yaml`) instead of CLI constants.
- [ ] Trend tracking: persist aggregate scores per experiment and chart drift
      over time, not just pairwise diffs.
- [ ] Dataset versioning support (LangSmith dataset versions ↔ EvalStore
      snapshots).
- [ ] Slack/webhook notifier on `REGRESSION DETECTED`.
- [ ] More evaluators: JSON-schema validity, toxicity/keyword blocklist, and a
      semantic-similarity scorer.

## Why I built this

Prompt changes are code changes, and I wanted the same discipline I expect
from software QA: golden data, versioned experiments, automated scoring, and a
regression gate that fails loudly. LangSmith is the industry reference for
that workflow — this repo shows I can operate it, and more importantly, that I
understand the machinery underneath well enough to rebuild it offline.
