"""
Tests for Prompt Management & Optimization module.

Covers state models, registry, comparator, optimizer, and integration
with the benchmark system.
"""

import json
import tempfile
from pathlib import Path

import pytest

from agentops.prompts.state import (
    PromptTemplate,
    PromptVersion,
    PromptCategory,
    PromptDiff,
    ComparisonConfig,
    ComparisonResult,
    OptimizationRun,
    OptimizationResult,
    DEFAULT_PROMPTS,
)
from agentops.prompts.registry import PromptRegistry
from agentops.prompts.comparator import (
    PromptComparator,
    PromptOptimizer,
    create_comparator,
    create_optimizer,
)
from agentops.evals.benchmarks import (
    ALL_BENCHMARKS,
    PROMPT_ENGINEERING_BENCH,
    get_benchmark,
    list_benchmarks,
)


# ═══════════════════════════════════════════════════════════════════════
# State Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPromptTemplate:
    def test_create_template(self):
        t = PromptTemplate(
            name="test-prompt",
            content="Answer: {{question}} using {{context}}",
            description="A test prompt",
        )
        assert t.name == "test-prompt"
        assert "question" in t.variables
        assert "context" in t.variables

    def test_render_success(self):
        t = PromptTemplate(
            name="test",
            content="Q: {{question}}\nA: {{answer}}",
        )
        result = t.render(question="What is AI?", answer="Artificial Intelligence")
        assert "What is AI?" in result
        assert "Artificial Intelligence" in result

    def test_render_missing_variable(self):
        t = PromptTemplate(name="test", content="{{greeting}}, {{name}}!")
        with pytest.raises(ValueError, match="Missing variables"):
            t.render(greeting="Hello")

    def test_to_dict(self):
        t = PromptTemplate(
            name="test",
            content="{{x}}",
            description="desc",
        )
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "desc"

    def test_variable_extraction(self):
        t = PromptTemplate(
            name="multi",
            content="{{a}} {{b}} {{a}} {{c}}",
        )
        assert t.variables == ["a", "b", "c"]  # sorted, unique


class TestPromptVersion:
    def test_create_version(self):
        v = PromptVersion(
            prompt_name="test",
            version=1,
            content="Be helpful",
            author="chrysis",
            changelog="First version",
        )
        assert v.prompt_name == "test"
        assert v.version == 1
        assert v.author == "chrysis"
        assert len(v.content_hash) == 12

    def test_to_dict(self):
        v = PromptVersion(prompt_name="x", version=2, content="hello")
        d = v.to_dict()
        assert d["version"] == 2
        assert d["content"] == "hello"

    def test_unique_hashes(self):
        v1 = PromptVersion(prompt_name="a", version=1, content="hello")
        v2 = PromptVersion(prompt_name="a", version=2, content="world")
        assert v1.content_hash != v2.content_hash


class TestPromptDiff:
    def test_diff_summary(self):
        d = PromptDiff(
            prompt_name="test",
            version_a=1,
            version_b=2,
            lines_added=["new line"],
            lines_removed=["old line"],
            lines_unchanged=5,
        )
        assert d.total_changes == 2
        assert "+1" in d.to_summary()


class TestComparisonResult:
    def test_comparison_markdown(self):
        config = ComparisonConfig(
            prompt_name="test", version_a=1, version_b=2,
        )
        result = ComparisonResult(
            config=config,
            version_a_scores={"accuracy": 0.8, "completeness": 0.7},
            version_b_scores={"accuracy": 0.9, "completeness": 0.8},
            winner="b",
            confidence=0.85,
            recommendation="Version B wins.",
        )
        md = result.to_markdown()
        assert "Version B" in md
        assert "0.800" in md
        assert "0.900" in md

    def test_comparison_to_dict(self):
        config = ComparisonConfig(prompt_name="t", version_a=1, version_b=2)
        result = ComparisonResult(
            config=config,
            version_a_scores={"a": 0.5},
            version_b_scores={"a": 0.6},
            winner="b",
            confidence=0.7,
            recommendation="Use B.",
        )
        d = result.to_dict()
        assert d["winner"] == "b"


class TestOptimizationResult:
    def test_optimization_markdown(self):
        result = OptimizationResult(
            prompt_name="test",
            initial_version=1,
            final_content="Optimized prompt",
            iterations=[
                OptimizationRun(
                    iteration=1,
                    prompt_content="v1",
                    scores={"accuracy": 0.5, "completeness": 0.4},
                ),
                OptimizationRun(
                    iteration=2,
                    prompt_content="v2",
                    scores={"accuracy": 0.8, "completeness": 0.7},
                ),
            ],
            best_iteration=2,
            best_scores={"accuracy": 0.8, "completeness": 0.7},
            improvement={"accuracy": 0.3, "completeness": 0.3},
        )
        md = result.to_markdown()
        assert "0.800" in md
        assert "Optimized prompt" in md
        assert "+0.300" in md

    def test_optimization_to_dict(self):
        result = OptimizationResult(
            prompt_name="test",
            initial_version=1,
            final_content="x",
            best_scores={"acc": 0.9},
            improvement={"acc": 0.2},
        )
        d = result.to_dict()
        assert d["iterations"] == 0
        assert d["best_scores"]["acc"] == 0.9


class TestDefaultPrompts:
    def test_all_defaults_registered(self):
        assert len(DEFAULT_PROMPTS) == 5
        assert "reliability-agent-system" in DEFAULT_PROMPTS
        assert "support-ticket-triage" in DEFAULT_PROMPTS
        assert "verification-check" in DEFAULT_PROMPTS
        assert "chain-of-thought-reasoning" in DEFAULT_PROMPTS
        assert "tool-use-decision" in DEFAULT_PROMPTS

    def test_defaults_have_variables(self):
        for name, template in DEFAULT_PROMPTS.items():
            assert template.variables, f"{name} should have variables"


# ═══════════════════════════════════════════════════════════════════════
# Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPromptRegistry:
    def test_default_prompts_on_init(self):
        reg = PromptRegistry()
        assert reg.prompt_count == 5
        assert reg.total_versions == 5

    def test_register_new_prompt(self):
        reg = PromptRegistry()
        v = reg.register("Hello {{name}}", name="greeting")
        assert v.version == 1
        assert reg.prompt_count == 6

    def test_register_duplicate(self):
        reg = PromptRegistry()
        reg.register("Hello {{name}}", name="greeting")
        with pytest.raises(ValueError, match="already exists"):
            reg.register("Different", name="greeting")

    def test_update_existing(self):
        reg = PromptRegistry()
        reg.register("Hello {{name}}", name="greeting")
        v2 = reg.update("greeting", "Hi {{name}}!", changelog="More casual")
        assert v2.version == 2
        assert reg.total_versions == 7  # 5 defaults + greeting v1 + greeting v2

    def test_update_nonexistent(self):
        reg = PromptRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.update("nonexistent", "new content")

    def test_get_latest(self):
        reg = PromptRegistry()
        reg.register("v1", name="test")
        reg.update("test", "v2")
        v = reg.get("test")
        assert v.version == 2
        assert v.content == "v2"

    def test_get_specific_version(self):
        reg = PromptRegistry()
        reg.register("v1", name="test")
        reg.update("test", "v2")
        v = reg.get("test", version=1)
        assert v.content == "v1"

    def test_get_bad_version(self):
        reg = PromptRegistry()
        with pytest.raises(KeyError):
            reg.get("reliability-agent-system", version=999)

    def test_list_prompts(self):
        reg = PromptRegistry()
        prompts = reg.list_prompts()
        assert len(prompts) >= 5
        names = [p["name"] for p in prompts]
        assert "reliability-agent-system" in names

    def test_list_versions(self):
        reg = PromptRegistry()
        reg.register("v1", name="test")
        reg.update("test", "v2")
        versions = reg.list_versions("test")
        assert len(versions) == 2
        assert versions[1]["version"] == 2

    def test_diff_versions(self):
        reg = PromptRegistry()
        reg.register("Line 1\nLine 2\nLine 3", name="test")
        reg.update("test", "Line 1\nLine 2 modified\nLine 3")
        diff = reg.diff("test", 1, 2)
        assert diff.total_changes > 0

    def test_diff_auto_previous(self):
        reg = PromptRegistry()
        reg.register("v1", name="test")
        reg.update("test", "v2")
        diff = reg.diff("test", 2)  # compare v2 vs v1
        assert diff.version_a == 1
        assert diff.version_b == 2

    def test_rollback(self):
        reg = PromptRegistry()
        reg.register("v1 content", name="test", changelog="Initial")
        reg.update("test", "v2 content", changelog="Changed")
        v3 = reg.rollback("test", 1)
        assert v3.version == 3
        assert v3.content == "v1 content"
        assert "Rollback" in v3.changelog

    def test_render(self):
        reg = PromptRegistry()
        reg.register("Hello {{username}}, welcome to {{place}}!", name="welcome")
        result = reg.render("welcome", username="Chrysis", place="AgentOps")
        assert "Chrysis" in result
        assert "AgentOps" in result

    def test_render_missing_var(self):
        reg = PromptRegistry()
        reg.register("Hello {{name}}!", name="greeting")
        with pytest.raises(ValueError, match="Missing variables"):
            reg.render("greeting")

    def test_save_and_load(self):
        reg = PromptRegistry()
        reg.register("custom: {{data}}", name="my-prompt")
        
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            reg.save(str(path))
            assert path.exists()
            
            loaded = PromptRegistry.load(str(path))
            assert loaded.prompt_count == reg.prompt_count
            v = loaded.get("my-prompt")
            assert v.content == "custom: {{data}}"

    def test_get_stats(self):
        reg = PromptRegistry()
        stats = reg.get_stats()
        assert stats["total_prompts"] >= 5
        assert stats["total_versions"] >= 5
        assert "categories" in stats

    def test_get_template(self):
        reg = PromptRegistry()
        t = reg.get_template("reliability-agent-system")
        assert t.category == PromptCategory.SYSTEM
        assert "product" in t.variables

    def test_register_with_metadata(self):
        reg = PromptRegistry()
        v = reg.register(
            "Prompt: {{task}}",
            name="meta-test",
            metadata={"model": "gpt-4o", "temperature": 0.7},
        )
        t = reg.get_template("meta-test")
        assert t.metadata["model"] == "gpt-4o"


# ═══════════════════════════════════════════════════════════════════════
# Comparator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPromptComparator:
    def test_create_comparator(self):
        comp = create_comparator()
        assert comp.simulated is True

    def test_simulated_compare_production_wins(self):
        comp = create_comparator()
        config = ComparisonConfig(prompt_name="test", version_a=1, version_b=2)
        
        # Good prompt vs bare prompt
        good = (
            "You are a reliability agent. Follow these rules:\n"
            "1. Retrieve documentation first.\n"
            "2. Cite all sources.\n"
            "3. Verify claims before responding.\n"
            "4. Never invent information."
        )
        bare = "Answer the question."
        
        result = comp.compare(config, good, bare)
        assert result.winner in ("a", "b", "tie")
        assert len(result.version_a_scores) > 0
        assert len(result.per_benchmark) > 0

    def test_simulated_compare_tie(self):
        comp = create_comparator()
        config = ComparisonConfig(prompt_name="test", version_a=1, version_b=2)
        
        same = "You are helpful. Cite sources. Verify claims."
        result = comp.compare(config, same, same)
        assert result.winner == "tie"

    def test_comparison_config_to_dict(self):
        config = ComparisonConfig(
            prompt_name="test",
            version_a=1,
            version_b=2,
            benchmark_names=["support-tickets", "tool-use"],
        )
        d = config.to_dict()
        assert len(d["benchmark_names"]) == 2

    def test_score_prompt_quality_range(self):
        comp = create_comparator()
        score = comp._score_prompt_quality("short")
        assert 0.4 <= score <= 0.95
        
        score2 = comp._score_prompt_quality(
            "You are an AI agent. Follow these rules:\n"
            "1. Retrieve documents first.\n"
            "2. Cite all sources.\n"
            "3. Verify claims with evidence.\n"
            "4. Never invent information.\n"
            "5. Flag dangerous operations for review.\n"
        )
        assert score2 > score, "Better prompts should score higher"


class TestPromptOptimizer:
    def test_create_optimizer(self):
        opt = create_optimizer()
        assert opt.simulated is True

    def test_optimize_improves_prompt(self):
        opt = create_optimizer()
        initial = "Answer the question about the product."
        
        result = opt.optimize(
            prompt_name="test-opt",
            initial_content=initial,
            max_iterations=3,
            target_score=0.95,
        )
        assert result.prompt_name == "test-opt"
        assert len(result.iterations) > 0
        assert result.best_iteration >= 1
        assert len(result.best_scores) == 5  # 5 metrics

    def test_optimize_tracks_improvement(self):
        opt = create_optimizer()
        initial = "Simple prompt."
        
        result = opt.optimize(
            prompt_name="test",
            initial_content=initial,
            max_iterations=3,
        )
        for metric, delta in result.improvement.items():
            assert delta >= 0, f"{metric} should not regress"

    def test_optimize_with_good_starting_prompt(self):
        opt = create_optimizer()
        good = (
            "You are a reliability agent. Follow these rules:\n"
            "1. Retrieve documentation first.\n"
            "2. Cite all sources with section numbers.\n"
            "3. Verify claims with evidence from documents.\n"
            "4. Never invent features or configurations.\n"
            "5. Flag dangerous operations for human review.\n"
            "6. Address edge cases and error scenarios.\n"
        )
        result = opt.optimize(
            prompt_name="good-prompt",
            initial_content=good,
            max_iterations=2,
            target_score=0.80,
        )
        # Should converge quickly
        assert len(result.iterations) <= 2

    def test_evaluate_prompt(self):
        opt = create_optimizer()
        prompt = "Be helpful. Cite sources. Verify claims. Never lie."
        scores = opt._evaluate_prompt(prompt, ["support-tickets"])
        assert "groundedness" in scores
        assert "completeness" in scores
        assert "clarity" in scores
        assert "safety" in scores
        assert "citation_quality" in scores
        for s in scores.values():
            assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPromptEngineeringBenchmark:
    def test_benchmark_exists(self):
        bench = get_benchmark("prompt-engineering")
        assert bench is not None
        assert bench.name == "prompt-engineering"
        assert len(bench.tasks) == 5

    def test_benchmark_in_all(self):
        assert PROMPT_ENGINEERING_BENCH in ALL_BENCHMARKS

    def test_benchmark_count(self):
        assert len(ALL_BENCHMARKS) == 12

    def test_tasks_have_ids(self):
        bench = get_benchmark("prompt-engineering")
        ids = [t.id for t in bench.tasks]
        assert ids == ["pe-001", "pe-002", "pe-003", "pe-004", "pe-005"]

    def test_task_categories(self):
        bench = get_benchmark("prompt-engineering")
        cats = {t.category for t in bench.tasks}
        assert "retrieval" in cats
        assert "multi_step" in cats
        assert "verification" in cats
        assert "tool_use" in cats

    def test_benchmark_listing(self):
        benchmarks = list_benchmarks()
        names = [b["name"] for b in benchmarks]
        assert "prompt-engineering" in names
        pe = next(b for b in benchmarks if b["name"] == "prompt-engineering")
        assert pe["task_count"] == 5
