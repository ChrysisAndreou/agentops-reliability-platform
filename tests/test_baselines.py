"""
Tests for baseline persistence (save/load/list evaluation baselines).
"""

import json
import tempfile
from pathlib import Path

import pytest

from agentops.evals.baselines import (
    BaselineManifest,
    save_baseline,
    load_baseline,
    list_baselines,
)


class TestBaselineManifest:
    """BaselineManifest dataclass tests."""

    def test_create_empty(self):
        m = BaselineManifest(name="v0.6", created_at="2026-01-01", profile="production")
        assert m.name == "v0.6"
        assert m.profile == "production"
        assert m.benchmarks == {}

    def test_to_dict(self):
        m = BaselineManifest(
            name="v0.6",
            created_at="2026-01-01",
            profile="production",
            benchmarks={
                "support-tickets": {
                    "summary": {"composite_mean": 0.85},
                    "per_task": [{"run_id": "st-001", "composite": 0.75}],
                }
            },
        )
        d = m.to_dict()
        assert d["name"] == "v0.6"
        assert d["profile"] == "production"
        assert d["benchmarks"]["support-tickets"]["summary"]["composite_mean"] == 0.85

    def test_from_dict(self):
        data = {
            "name": "v0.6",
            "created_at": "2026-01-01",
            "profile": "production",
            "benchmarks": {},
        }
        m = BaselineManifest.from_dict(data)
        assert m.name == "v0.6"
        assert m.profile == "production"

    def test_from_dict_partial(self):
        """from_dict should handle missing optional fields."""
        m = BaselineManifest.from_dict({"name": "v0.6"})
        assert m.name == "v0.6"
        assert m.profile == "production"
        assert m.benchmarks == {}


class TestSaveBaseline:
    """save_baseline tests."""

    def test_save_and_reload(self):
        benchmark_results = {
            "support-tickets": [
                {
                    "run_id": "st-001",
                    "task_id": "st-001",
                    "groundedness": 0.85,
                    "citation_precision": 0.90,
                    "verification_pass_rate": 1.0,
                    "verification_passed": True,
                    "tool_success_rate": 1.0,
                    "answer_completeness": 0.80,
                    "composite": 0.88,
                    "latency_ms": 5000.0,
                    "latency_score": 0.96,
                    "tool_calls_count": 2,
                    "key_terms_found": 4,
                    "key_terms_total": 5,
                    "grounded_claims_count": 3,
                    "ungrounded_claims_count": 1,
                    "citations_used_count": 3,
                    "retrieved_chunks_count": 5,
                },
                {
                    "run_id": "st-002",
                    "task_id": "st-002",
                    "groundedness": 0.60,
                    "citation_precision": 0.40,
                    "verification_pass_rate": 1.0,
                    "verification_passed": True,
                    "tool_success_rate": 1.0,
                    "answer_completeness": 0.60,
                    "composite": 0.70,
                    "latency_ms": 6000.0,
                    "latency_score": 0.95,
                    "tool_calls_count": 1,
                    "key_terms_found": 3,
                    "key_terms_total": 5,
                    "grounded_claims_count": 2,
                    "ungrounded_claims_count": 1,
                    "citations_used_count": 2,
                    "retrieved_chunks_count": 5,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_baseline(
                benchmark_results=benchmark_results,
                name="v0.6",
                profile="production",
                output_path=tmpdir,
            )

            assert path.exists()
            data = json.loads(path.read_text())
            assert data["name"] == "v0.6"
            assert data["profile"] == "production"
            assert "support-tickets" in data["benchmarks"]

            # Check summary
            summary = data["benchmarks"]["support-tickets"]["summary"]
            assert summary["tasks_evaluated"] == 2
            assert summary["verification_pass_count"] == 2
            assert summary["composite_mean"] == pytest.approx(0.79, abs=0.01)
            assert summary["groundedness_mean"] == pytest.approx(0.725, abs=0.01)

    def test_save_empty_benchmarks(self):
        """Empty benchmark results should produce valid baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_baseline(
                benchmark_results={"empty": []},
                name="empty-baseline",
                profile="production",
                output_path=tmpdir,
            )
            assert path.exists()
            data = json.loads(path.read_text())
            # Empty benchmark list results in no benchmark entry
            assert "empty" not in data["benchmarks"]

    def test_save_multiple_benchmarks(self):
        benchmark_results = {
            "support-tickets": [
                {"groundedness": 0.8, "citation_precision": 0.7, "verification_pass_rate": 1.0,
                 "verification_passed": True, "tool_success_rate": 1.0, "answer_completeness": 0.5,
                 "composite": 0.75, "latency_ms": 5000.0, "latency_score": 0.96},
            ],
            "systems-quality": [
                {"groundedness": 0.7, "citation_precision": 0.6, "verification_pass_rate": 1.0,
                 "verification_passed": True, "tool_success_rate": 1.0, "answer_completeness": 0.4,
                 "composite": 0.65, "latency_ms": 6000.0, "latency_score": 0.95},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_baseline(
                benchmark_results=benchmark_results,
                name="multi-bench",
                profile="production",
                output_path=tmpdir,
            )
            data = json.loads(path.read_text())
            assert len(data["benchmarks"]) == 2
            assert "support-tickets" in data["benchmarks"]
            assert "systems-quality" in data["benchmarks"]

    def test_default_output_path(self, monkeypatch, tmp_path):
        """Default output goes to eval_results/baselines/."""
        monkeypatch.chdir(tmp_path)

        benchmark_results = {
            "test-bench": [
                {"groundedness": 0.5, "citation_precision": 0.5, "verification_pass_rate": 1.0,
                 "verification_passed": True, "tool_success_rate": 1.0, "answer_completeness": 0.5,
                 "composite": 0.5, "latency_ms": 5000.0, "latency_score": 0.96},
            ]
        }
        path = save_baseline(
            benchmark_results=benchmark_results,
            name="test-baseline",
            profile="production",
            output_path=None,
        )
        assert path.exists()
        assert "eval_results/baselines" in str(path)


class TestLoadBaseline:
    """load_baseline tests."""

    def test_load_existing(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text(json.dumps({
            "name": "v0.6",
            "created_at": "2026-01-01",
            "profile": "production",
            "benchmarks": {},
        }))

        loaded = load_baseline(str(path))
        assert loaded.name == "v0.6"
        assert loaded.profile == "production"

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_baseline("/nonexistent/baseline.json")

    def test_load_by_name_in_baselines_dir(self, tmp_path):
        """Short name lookup in baselines dir passes baselines_dir param."""
        baselines_dir = tmp_path / "eval_results" / "baselines"
        baselines_dir.mkdir(parents=True)
        (baselines_dir / "v0.6.json").write_text(json.dumps({
            "name": "v0.6",
            "created_at": "2026-01-01",
            "profile": "production",
            "benchmarks": {},
        }))

        loaded = load_baseline("v0.6", baselines_dir=baselines_dir)
        assert loaded.name == "v0.6"


class TestListBaselines:
    """list_baselines tests."""

    def test_empty_dir(self, tmp_path):
        baselines = list_baselines(tmp_path)
        assert baselines == []

    def test_nonexistent_dir(self):
        baselines = list_baselines("/nonexistent/path")
        assert baselines == []

    def test_list_baselines(self, tmp_path):
        (tmp_path / "v0.5.json").write_text(json.dumps({
            "name": "v0.5",
            "created_at": "2026-06-01",
            "profile": "production",
            "benchmarks": {"a": {}, "b": {}},
        }))
        (tmp_path / "v0.6.json").write_text(json.dumps({
            "name": "v0.6",
            "created_at": "2026-06-28",
            "profile": "production",
            "benchmarks": {"a": {}, "b": {}, "c": {}},
        }))

        baselines = list_baselines(tmp_path)
        assert len(baselines) == 2
        assert baselines[0]["name"] == "v0.5"
        assert baselines[0]["benchmark_count"] == 2
        assert baselines[1]["name"] == "v0.6"
        assert baselines[1]["benchmark_count"] == 3

    def test_skip_invalid_json(self, tmp_path):
        (tmp_path / "valid.json").write_text(json.dumps({
            "name": "valid", "created_at": "", "profile": "", "benchmarks": {}
        }))
        (tmp_path / "invalid.json").write_text("not json")

        baselines = list_baselines(tmp_path)
        assert len(baselines) == 1
        assert baselines[0]["name"] == "valid"
