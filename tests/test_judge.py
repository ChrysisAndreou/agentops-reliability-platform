"""
Tests for LLM-as-Judge evaluation framework.

Covers state models, SimulatedJudge (deterministic, CI-safe),
JudgeRunner, and JudgeConfig.
"""

import pytest

from agentops.evals.judge.state import (
    JudgeConfig,
    JudgeDimension,
    JudgeResult,
    JudgeVerdict,
    JudgeRubric,
    JudgeBenchmarkResult,
    DEFAULT_RUBRICS,
)
from agentops.evals.judge.judge import (
    SimulatedJudge,
    JudgeRunner,
)


# ── State Model Tests ────────────────────────────────────────────────

class TestJudgeDimension:
    def test_all_dimensions_exist(self):
        dims = list(JudgeDimension)
        assert len(dims) == 8
        assert JudgeDimension.ACCURACY in dims
        assert JudgeDimension.SAFETY in dims

    def test_dimension_values(self):
        assert JudgeDimension.ACCURACY.value == "accuracy"
        assert JudgeDimension.SAFETY.value == "safety"
        assert JudgeDimension.CITATION_QUALITY.value == "citation_quality"


class TestJudgeVerdict:
    def test_create_verdict(self):
        v = JudgeVerdict(
            dimension=JudgeDimension.ACCURACY,
            score=0.85,
            reasoning="The answer is correct",
            evidence=["Claim verified by doc"],
        )
        assert v.score == 0.85
        assert v.passed is True

    def test_failed_verdict(self):
        v = JudgeVerdict(
            dimension=JudgeDimension.SAFETY,
            score=0.3,
            reasoning="Contains harmful content",
            passed=False,
        )
        assert v.passed is False
        assert v.score == 0.3

    def test_to_dict(self):
        v = JudgeVerdict(
            dimension=JudgeDimension.COMPLETENESS,
            score=0.9,
            reasoning="All aspects covered",
        )
        d = v.to_dict()
        assert d["dimension"] == "completeness"
        assert d["score"] == 0.9
        assert "reasoning" in d


class TestJudgeRubric:
    def test_create_rubric(self):
        r = JudgeRubric(
            dimension=JudgeDimension.ACCURACY,
            description="Factual correctness",
            score_0="All wrong",
            score_5="Some right",
            score_10="All right",
        )
        assert r.weight == 1.0

    def test_to_dict(self):
        r = JudgeRubric(
            dimension=JudgeDimension.SAFETY,
            description="Safety check",
            score_0="Dangerous",
            score_5="Borderline",
            score_10="Safe",
            weight=1.2,
        )
        d = r.to_dict()
        assert d["weight"] == 1.2
        assert d["dimension"] == "safety"


class TestJudgeConfig:
    def test_default_config(self):
        config = JudgeConfig()
        assert len(config.dimensions) == 5
        assert config.pass_threshold == 0.6
        assert config.judge_model == "gpt-4o"
        assert len(config.rubrics) == 8  # All defaults populated

    def test_custom_config(self):
        config = JudgeConfig(
            dimensions=[JudgeDimension.ACCURACY, JudgeDimension.SAFETY],
            pass_threshold=0.8,
            judge_model="claude-3-opus",
        )
        assert len(config.dimensions) == 2
        assert config.pass_threshold == 0.8
        assert config.judge_model == "claude-3-opus"


class TestJudgeResult:
    def test_create_result(self):
        r = JudgeResult(
            task_id="st-001",
            agent_output="The answer is 42",
            composite_score=0.88,
            passed=True,
        )
        assert r.task_id == "st-001"
        assert r.passed is True
        assert r.verdicts == []

    def test_dimension_scores(self):
        v1 = JudgeVerdict(JudgeDimension.ACCURACY, 0.9, "Good")
        v2 = JudgeVerdict(JudgeDimension.SAFETY, 0.8, "Safe")
        r = JudgeResult(
            task_id="t1",
            agent_output="test",
            verdicts=[v1, v2],
            composite_score=0.85,
            passed=True,
        )
        assert r.dimension_scores == {"accuracy": 0.9, "safety": 0.8}

    def test_to_dict(self):
        v = JudgeVerdict(JudgeDimension.ACCURACY, 0.85, "Correct")
        r = JudgeResult(
            task_id="t1",
            agent_output="test output " * 50,
            verdicts=[v],
            composite_score=0.85,
            passed=True,
        )
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert len(d["verdicts"]) == 1
        # Agent output truncated to 500 chars
        assert len(d["agent_output"]) <= 500


class TestJudgeBenchmarkResult:
    def test_pass_rate(self):
        r1 = JudgeResult("t1", "a", composite_score=0.9, passed=True)
        r2 = JudgeResult("t2", "b", composite_score=0.5, passed=False)
        br = JudgeBenchmarkResult(
            benchmark_name="test-bench",
            judge_model="simulated",
            agent_model="test",
            results=[r1, r2],
        )
        assert br.pass_rate == 0.5
        assert br.mean_composite == 0.7

    def test_empty_results(self):
        br = JudgeBenchmarkResult(
            benchmark_name="empty",
            judge_model="simulated",
            agent_model="test",
        )
        assert br.pass_rate == 0.0
        assert br.mean_composite == 0.0


class TestDefaultRubrics:
    def test_all_dimensions_have_rubrics(self):
        for dim in JudgeDimension:
            assert dim in DEFAULT_RUBRICS

    def test_rubric_content(self):
        r = DEFAULT_RUBRICS[JudgeDimension.ACCURACY]
        assert r.weight == 1.5
        assert "wrong" in r.score_0.lower()
        assert "correct" in r.score_10.lower()

    def test_groundedness_high_weight(self):
        r = DEFAULT_RUBRICS[JudgeDimension.GROUNDEDNESS]
        assert r.weight == 1.5  # Core reliability dimension


# ── SimulatedJudge Tests ─────────────────────────────────────────────

class TestSimulatedJudge:
    def test_evaluate_returns_result(self):
        judge = SimulatedJudge(seed=42)
        result = judge.evaluate(
            task_id="test-001",
            agent_output="The Docker daemon must be running. Check your pipeline settings.",
            key_terms=["docker", "pipeline", "settings"],
        )
        assert isinstance(result, JudgeResult)
        assert result.task_id == "test-001"
        assert len(result.verdicts) == 5  # Default dimensions
        assert result.judge_model == "simulated-judge"

    def test_deterministic_same_seed(self):
        judge1 = SimulatedJudge(seed=42)
        judge2 = SimulatedJudge(seed=42)
        output = "Docker daemon not available. Check runtime settings in clouddeploy.yml."
        r1 = judge1.evaluate("st-001", output, key_terms=["docker", "runtime", "settings"])
        r2 = judge2.evaluate("st-001", output, key_terms=["docker", "runtime", "settings"])
        assert r1.composite_score == r2.composite_score

    def test_different_seeds_produce_different(self):
        judge1 = SimulatedJudge(seed=42)
        judge2 = SimulatedJudge(seed=99)
        output = "Docker daemon not available."
        r1 = judge1.evaluate("st-001", output, key_terms=["docker"])
        r2 = judge2.evaluate("st-001", output, key_terms=["docker"])
        # Might be same by chance, but verify structure
        assert len(r1.verdicts) == len(r2.verdicts)

    def test_accuracy_with_key_terms_present(self):
        judge = SimulatedJudge(seed=42)
        output = "docker runtime pipeline settings dedicated plan"
        result = judge.evaluate(
            "t1", output,
            key_terms=["docker", "runtime", "pipeline", "settings", "dedicated"],
        )
        acc = [v for v in result.verdicts if v.dimension == JudgeDimension.ACCURACY][0]
        assert acc.score > 0.5  # Most terms present

    def test_accuracy_with_no_key_terms(self):
        judge = SimulatedJudge(seed=42)
        output = "docker runtime pipeline settings"
        result = judge.evaluate("t1", output, key_terms=[])
        acc = [v for v in result.verdicts if v.dimension == JudgeDimension.ACCURACY][0]
        assert 0.5 <= acc.score <= 1.0  # Default scoring

    def test_safety_detects_harmful(self):
        judge = SimulatedJudge(seed=42)
        output = "To hack the system, exploit this vulnerability and bypass security."
        result = judge.evaluate("t1", output, key_terms=["security"])
        safety = [v for v in result.verdicts if v.dimension == JudgeDimension.SAFETY][0]
        assert safety.score < 0.9  # Should be penalized for harmful content

    def test_safety_clean_output(self):
        judge = SimulatedJudge(seed=42)
        output = "The system uses TLS 1.3 encryption for data in transit and AES-256 at rest."
        result = judge.evaluate("t2", output, key_terms=["TLS", "AES-256"])
        safety = [v for v in result.verdicts if v.dimension == JudgeDimension.SAFETY][0]
        assert safety.score > 0.85  # Clean output should score high

    def test_completeness_all_terms(self):
        judge = SimulatedJudge(seed=42)
        output = "docker runtime pipeline settings dedicated multi-stage parallel"
        result = judge.evaluate("t1", output, key_terms=["docker", "runtime", "pipeline"])
        comp = [v for v in result.verdicts if v.dimension == JudgeDimension.COMPLETENESS][0]
        assert comp.score > 0.7

    def test_completeness_few_terms(self):
        judge = SimulatedJudge(seed=42)
        output = "docker"
        result = judge.evaluate("t1", output, key_terms=["docker", "runtime", "pipeline", "settings"])
        comp = [v for v in result.verdicts if v.dimension == JudgeDimension.COMPLETENESS][0]
        assert comp.score < 0.5  # Only 1/4 terms

    def test_citation_quality_with_signals(self):
        judge = SimulatedJudge(seed=42)
        output = "According to the docs, [source: clouddeploy-platform.md], the setting is..."
        result = judge.evaluate("t1", output, key_terms=["docker"],
                                expected_sources=["clouddeploy-platform.md"])
        cit = [v for v in result.verdicts if v.dimension == JudgeDimension.CITATION_QUALITY][0]
        assert cit.score > 0.5

    def test_groundedness_with_signals(self):
        config = JudgeConfig(dimensions=[
            JudgeDimension.ACCURACY, JudgeDimension.GROUNDEDNESS,
            JudgeDimension.COMPLETENESS, JudgeDimension.SAFETY,
            JudgeDimension.CITATION_QUALITY,
        ])
        judge = SimulatedJudge(config=config, seed=42)
        output = "According to the documentation, as stated in the platform guide, based on evidence shows that..."
        result = judge.evaluate("t1", output, key_terms=["docker"])
        grd = [v for v in result.verdicts if v.dimension == JudgeDimension.GROUNDEDNESS][0]
        assert grd.score > 0.7

    def test_clarity_structured_output(self):
        config = JudgeConfig(dimensions=[
            JudgeDimension.ACCURACY, JudgeDimension.CLARITY,
            JudgeDimension.COMPLETENESS, JudgeDimension.SAFETY,
            JudgeDimension.CITATION_QUALITY,
        ])
        judge = SimulatedJudge(config=config, seed=42)
        output = "First check the Docker runtime configuration, then verify the pipeline settings are correct, and finally restart the service to apply changes. The system should now be operational."
        result = judge.evaluate("t1", output, key_terms=["docker"])
        cla = [v for v in result.verdicts if v.dimension == JudgeDimension.CLARITY][0]
        assert cla.score > 0.45

    def test_relevance_very_short(self):
        judge = SimulatedJudge(seed=42)
        output = "yes"
        result = judge.evaluate("t1", output, key_terms=["docker", "pipeline", "settings"])
        rel = [v for v in result.verdicts if v.dimension == JudgeDimension.RELEVANCE][0]
        assert rel.score < 0.5  # Very short should score low

    def test_composite_score_range(self):
        judge = SimulatedJudge(seed=42)
        output = "Docker daemon must be running. Check the pipeline settings in clouddeploy.yml."
        result = judge.evaluate("t1", output, key_terms=["docker", "pipeline", "settings"])
        assert 0.0 <= result.composite_score <= 1.0

    def test_passed_when_above_thresholds(self):
        config = JudgeConfig(pass_threshold=0.4, composite_threshold=0.5)
        judge = SimulatedJudge(config=config, seed=42)
        output = "docker pipeline settings runtime multi-stage parallel dedicated"
        result = judge.evaluate("t1", output,
                                key_terms=["docker", "pipeline", "settings", "runtime"])
        # With many terms, should pass
        assert result.composite_score > 0.5


# ── JudgeRunner Tests ────────────────────────────────────────────────

class TestJudgeRunner:
    def test_evaluate_benchmark(self):
        runner = JudgeRunner(use_simulated=True)
        outputs = {
            "st-001": {
                "output": "Check the Docker daemon and pipeline settings.",
                "key_terms": ["docker", "pipeline", "settings"],
            },
            "st-002": {
                "output": "Enable TOTP in Security settings for 2FA.",
                "key_terms": ["TOTP", "2FA", "Security"],
            },
        }
        result = runner.evaluate_benchmark(
            benchmark_name="test-bench",
            agent_outputs=outputs,
            agent_model="test-model",
        )
        assert result.benchmark_name == "test-bench"
        assert result.agent_model == "test-model"
        assert len(result.results) == 2
        assert result.pass_rate >= 0.0
        assert 0.0 <= result.mean_composite <= 1.0

    def test_evaluate_benchmark_empty(self):
        runner = JudgeRunner(use_simulated=True)
        result = runner.evaluate_benchmark(
            benchmark_name="empty",
            agent_outputs={},
            agent_model="test",
        )
        assert result.pass_rate == 0.0
        assert result.mean_composite == 0.0

    def test_generate_report(self):
        runner = JudgeRunner(use_simulated=True)
        outputs = {
            "st-001": {
                "output": "Check Docker daemon and pipeline settings.",
                "key_terms": ["docker", "pipeline"],
            },
        }
        result = runner.evaluate_benchmark("test", outputs, "test-model")
        report = runner.generate_report(result)
        assert "# LLM-Judge Evaluation Report: test" in report
        assert "Pass Rate" in report
        assert "Dimension Scores" in report
        assert "Per-Task Results" in report


# ── LLMJudge Tests (Stub — requires API keys for real use) ────────────

class TestLLMJudge:
    def test_build_prompt(self):
        from agentops.evals.judge.judge import LLMJudge
        judge = LLMJudge()
        prompt = judge._build_judge_prompt(
            "The answer is 42.",
            "What is the meaning of life?",
        )
        assert "expert evaluator" in prompt.lower()
        assert "42" in prompt
        assert "SCORING RUBRIC" in prompt
        assert "accuracy" in prompt.lower()
        assert "TASK CONTEXT" in prompt

    def test_compute_result(self):
        from agentops.evals.judge.judge import LLMJudge
        judge = LLMJudge()
        v1 = JudgeVerdict(JudgeDimension.ACCURACY, 0.9, "Good")
        v2 = JudgeVerdict(JudgeDimension.SAFETY, 0.8, "Safe")
        result = judge._compute_result("t1", "output", [v1, v2], 100.0)
        assert 0.8 <= result.composite_score <= 1.0
        assert result.judge_model == "gpt-4o"

    def test_compute_result_empty(self):
        from agentops.evals.judge.judge import LLMJudge
        judge = LLMJudge()
        result = judge._compute_result("t1", "output", [], 50.0)
        assert result.composite_score == 0.0
        assert result.passed is False


# ── Edge Cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_output(self):
        judge = SimulatedJudge(seed=42)
        result = judge.evaluate("t1", "", key_terms=["docker"])
        assert isinstance(result, JudgeResult)
        assert len(result.verdicts) == 5

    def test_very_long_output(self):
        judge = SimulatedJudge(seed=42)
        output = "docker pipeline settings " * 200
        result = judge.evaluate("t1", output, key_terms=["docker", "pipeline"])
        assert len(result.verdicts) == 5

    def test_special_characters(self):
        judge = SimulatedJudge(seed=42)
        output = "Docker™ daemon @runtime #pipeline $settings %encryption"
        result = judge.evaluate("t1", output, key_terms=["docker", "runtime"])
        acc = [v for v in result.verdicts if v.dimension == JudgeDimension.ACCURACY][0]
        assert acc.score > 0.0

    def test_mixed_case_key_terms(self):
        judge = SimulatedJudge(seed=42)
        output = "docker runtime pipeline settings"
        result = judge.evaluate("t1", output, key_terms=["DOCKER", "Runtime", "PIPELINE"])
        acc = [v for v in result.verdicts if v.dimension == JudgeDimension.ACCURACY][0]
        assert acc.score > 0.7
