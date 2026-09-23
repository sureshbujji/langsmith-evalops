"""Prompt versions and deterministic simulated model responses.

In production these "prompts" would be sent to a real LLM and the latencies
would be measured wall-clock. For an offline-first demo the responses are
deterministic stand-ins keyed by task id, so the evaluation pipeline --
datasets, experiments, evaluators, regression tracking -- runs end to end
with no API key and no network.

The two versions are seeded so that v2 REGRESSES on purpose:
  * task_05: v2 answers "System" instead of "Software" (correctness regression)
  * task_09: v2 gives a vague non-answer instead of the two causes (quality regression)
  * task_03: v2 is correct but blows the 1500ms latency budget (latency regression)

The regression detector in evaluators.py is expected to catch all three.
To use real model calls, replace ``simulate_response`` with your LLM call and
keep the signature ``(task, version) -> (output, latency_ms)``.
"""

from __future__ import annotations

from typing import Dict, Tuple

PROMPT_VERSIONS: Dict[str, Dict[str, str]] = {
    "v1": {
        "name": "qa-assistant-v1",
        "description": "Baseline prompt: direct instruction, answer concisely.",
        "system_prompt": (
            "You are a precise QA assistant. Answer the user's question "
            "directly and concisely. For factual questions give the exact "
            "answer with no extra prose."
        ),
    },
    "v2": {
        "name": "qa-assistant-v2",
        "description": (
            "Candidate prompt: adds a 'be conversational' style instruction. "
            "Seeded to regress on task_05, task_09 (quality) and task_03 "
            "(latency) to demonstrate regression detection."
        ),
        "system_prompt": (
            "You are a friendly, conversational QA assistant. Chat with the "
            "user naturally and feel free to elaborate. Keep answers helpful "
            "and personable."
        ),
    },
}

# task_id -> (output, latency_ms)
_RESPONSES: Dict[str, Dict[str, Tuple[str, float]]] = {
    "v1": {
        "task_01": ("Paris", 320.0),
        "task_02": ("GET", 280.0),
        "task_03": (
            "Automated regression suites catch unintended behavior changes "
            "quickly before they reach production.",
            840.0,
        ),
        "task_04": ("negative", 300.0),
        "task_05": ("Software", 310.0),
        "task_06": (
            "Contract testing catches breaking API changes early without "
            "deploying the full system.",
            760.0,
        ),
        "task_07": ("443", 260.0),
        "task_08": ("True", 290.0),
        "task_09": (
            "Two common causes are timing issues and shared test state.",
            880.0,
        ),
        "task_10": ("Continuous Integration", 300.0),
        "task_11": ("pytest -m smoke", 340.0),
        "task_12": (
            "A test plan defines the scope, approach, resources, and "
            "schedule for testing.",
            790.0,
        ),
    },
    "v2": {
        "task_01": ("Paris", 410.0),
        "task_02": ("GET", 350.0),
        # Correct answer, but the chatty style blew the latency budget.
        "task_03": (
            "Automated regression suites catch unintended behavior changes "
            "quickly before they reach production.",
            2450.0,
        ),
        "task_04": ("negative", 390.0),
        # REGRESSION: wrong expansion of the acronym.
        "task_05": ("System", 360.0),
        "task_06": (
            "Contract testing catches breaking API changes early without "
            "deploying the full system.",
            800.0,
        ),
        "task_07": ("443", 330.0),
        "task_08": ("True", 370.0),
        # REGRESSION: vague non-answer, misses both reference causes.
        "task_09": ("Sometimes tests are flaky for random reasons.", 620.0),
        "task_10": ("Continuous Integration", 420.0),
        "task_11": ("pytest -m smoke", 450.0),
        "task_12": (
            "A test plan defines scope, approach, resources, and schedule "
            "for testing activities.",
            810.0,
        ),
    },
}


def simulate_response(task: Dict[str, str], version: str) -> Tuple[str, float]:
    """Return (output, latency_ms) for a task under a prompt version.

    Swap this for a real LLM call in production; the pipeline only depends
    on the ``(output, latency_ms)`` contract.
    """
    if version not in _RESPONSES:
        raise ValueError(f"unknown prompt version: {version!r}")
    try:
        return _RESPONSES[version][task["task_id"]]
    except KeyError as exc:
        raise KeyError(
            f"no simulated response for {task['task_id']!r} under {version!r}"
        ) from exc


def list_versions() -> Dict[str, Dict[str, str]]:
    return PROMPT_VERSIONS
