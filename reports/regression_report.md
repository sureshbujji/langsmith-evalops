# Experiment Regression Report

_Generated: 2026-09-23 04:08:22 | Backend: `EvalStore (SQLite, offline)`_

## Verdict: 🔴 REGRESSION DETECTED

Compared **12** golden tasks: baseline `qa-assistant-v1-20260923-040818` (prompt v1) vs candidate `qa-assistant-v2-20260923-040820` (prompt v2). **4 regression(s)** detected.

### Configuration

* Latency budget: `1500ms`
* Rubric drop threshold: `1.0`

### Aggregate scores

| Metric | Baseline (v1) | Candidate (v2) | Delta |
|---|---|---|---|
| Mean exact-match | 0.667 | 0.583 | -0.084 ⚠️ |
| Mean rubric (1-5) | 4.944 | 4.277 | -0.667 ⚠️ |
| Mean latency (ms) | 472.5 | 646.7 | +174.2 ⚠️ |

### Regressions caught

| Task | Metric | Baseline | Candidate | Delta | Severity | Reason |
|---|---|---|---|---|---|---|
| `task_03` | latency_ms | 840.0 | 2450.0 | +1610.0 | medium | latency 2450ms exceeds budget 1500ms |
| `task_05` | exact_match | 1.0 | 0.0 | -1.0 | high | previously exact answer is now wrong |
| `task_05` | rubric | 5.0 | 1.0 | -4.0 | high | rubric score dropped by 4.00 (threshold 1.0) |
| `task_09` | rubric | 5.0 | 1.0 | -4.0 | high | rubric score dropped by 4.00 (threshold 1.0) |

### Per-task detail

| Task | Category | Exact Δ | Rubric Δ | Latency Δ (ms) | Regressed |
|---|---|---|---|---|---|
| `task_01` | factual | +0.0 | +0.0 | +90.0 | ✅ |
| `task_02` | factual | +0.0 | +0.0 | +70.0 | ✅ |
| `task_03` | summary | +0.0 | +0.0 | +1610.0 | ❌ |
| `task_04` | classification | +0.0 | +0.0 | +90.0 | ✅ |
| `task_05` | factual | -1.0 | -4.0 | +50.0 | ❌ |
| `task_06` | summary | +0.0 | +0.0 | +40.0 | ✅ |
| `task_07` | factual | +0.0 | +0.0 | +70.0 | ✅ |
| `task_08` | classification | +0.0 | +0.0 | +80.0 | ✅ |
| `task_09` | summary | +0.0 | -4.0 | -260.0 | ❌ |
| `task_10` | factual | +0.0 | +0.0 | +120.0 | ✅ |
| `task_11` | factual | +0.0 | +0.0 | +110.0 | ✅ |
| `task_12` | summary | +0.0 | +0.0 | +20.0 | ✅ |

---
_Offline-first demo: the same pipeline runs against real LangSmith by setting `LANGSMITH_API_KEY` -- see `src/adapter.py` and the README's 'Offline vs live' section._
