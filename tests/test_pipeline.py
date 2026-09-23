"""End-to-end smoke test: full pipeline with no API key and no network."""

from src.run_experiment import main


def test_full_pipeline_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    db_path = tmp_path / "smoke.db"
    report_path = tmp_path / "regression_report.md"

    rc = main(
        [
            "--db-path", str(db_path),
            "--report-path", str(report_path),
            "--dataset-name", "smoke-dataset",
        ]
    )
    assert rc == 0
    assert db_path.exists()
    assert report_path.exists()

    report = report_path.read_text(encoding="utf-8")
    assert "REGRESSION DETECTED" in report
    # the seeded regressions must be visible in the sample report
    assert "task_05" in report  # wrong answer (exact_match 1.0 -> 0.0)
    assert "task_09" in report  # vague answer (rubric collapse)
    assert "task_03" in report  # latency budget blowout
    assert "qa-assistant-v1" in report
    assert "qa-assistant-v2" in report
