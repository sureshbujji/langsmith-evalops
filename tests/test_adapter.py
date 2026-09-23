"""Tests for the adapter's env-var switch (offline vs live backend)."""

import pytest

from src import adapter
from src.evalstore import EvalStore


def test_no_key_returns_offline_evalstore(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("EVALOPS_DB", ":memory:")
    client = adapter.get_client()
    assert isinstance(client, EvalStore)
    # the offline client exposes the full mirrored interface
    for method in (
        "create_dataset", "create_examples", "create_experiment",
        "log_run", "attach_scores", "compare_experiments",
    ):
        assert callable(getattr(client, method))
    client.close()


def test_key_returns_live_adapter_or_helpful_error(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key-for-tests")
    try:
        import langsmith  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="pip install langsmith"):
            adapter.get_client()
    else:
        client = adapter.get_client()
        assert isinstance(client, adapter.LangSmithAdapter)
        for method in (
            "create_dataset", "create_examples", "create_experiment",
            "log_run", "attach_scores", "compare_experiments",
        ):
            assert callable(getattr(client, method))
