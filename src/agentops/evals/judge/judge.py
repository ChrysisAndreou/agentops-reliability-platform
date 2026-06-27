"""
LLM-as-Judge evaluation module.

Provides two judge implementations:
- LLMJudge: Uses a real LLM to evaluate agent outputs (requires API keys)
- SimulatedJudge: Deterministic rule-based evaluation for CI/CD (no API keys)

The judge evaluates agent outputs across configurable dimensions (accuracy,
completeness, relevance, safety, tool use, citation, groundedness, clarity)
and produces structured verdicts with reasoning and evidence.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .state import (
    DEFAULT_RUBRICS,
    JudgeBenchmarkResult,
    JudgeConfig,
    JudgeDimension,
    JudgeResult,
    JudgeVerdict,
)

# ── Utility ──────────────────────────────────────────────────────────

def _seed_from_task(task_id: str, salt: str = "") -> random.Random:
    h = hashlib.sha256(f"{task_id}:{salt}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# ── LLM Judge (Real API) ─────────────────────────────────────────────

@dataclass
class LLMJudge:
    """Evaluate agent outputs using a real LLM as judge.

    Requires API keys configured via environment variables:
    - OPENAI_API_KEY for GPT-4o judge
    - ANTHROPIC_API_KEY for Claude judge
    - DEEPSEEK_API_KEY for DeepSeek judge

    Usage:
        judge = LLMJudge(config=JudgeConfig(judge_model="gpt-4o"))
        result = await judge.evaluate(task_id="t1", agent_output="...")
        print(result.composite_score)
    """

    config: JudgeConfig = field(default_factory=JudgeConfig)

    def _build_judge_prompt(self, agent_output: str, task_question: str = "") -> str:
        """Build the evaluation prompt for the judge LLM."""
        dims = self.config.dimensions
        rubric_lines = []
        for d in dims:
            r = self.config.rubrics.get(d, DEFAULT_RUBRICS.get(d))
            if r:
                rubric_lines.append(
                    f"### {r.dimension.value} (weight: {r.weight})\n"
                    f"- 0/10: {r.score_0}\n"
                    f"- 5/10: {r.score_5}\n"
                    f"- 10/10: {r.score_10}\n"
                )

        rubric_text = "\n".join(rubric_lines)
        dim_names = ", ".join(d.value for d in dims)

        prompt = f"""You are an expert evaluator of AI agent outputs. Evaluate the following agent response across these dimensions: {dim_names}.

SCORING RUBRIC:
{rubric_text}

TASK CONTEXT:
{task_question if task_question else "No task context provided."}

AGENT OUTPUT TO EVALUATE:
```
{agent_output}
```

INSTRUCTIONS:
1. For each dimension, assign a score from 0.0 to 1.0 (0=worst, 1=best) using the rubric above.
2. Provide a brief reasoning for each score.
3. Cite specific evidence from the agent output that supports your judgment.

Respond in this exact JSON format:
{{
  "verdicts": [
    {{
      "dimension": "accuracy",
      "score": 0.85,
      "reasoning": "The answer correctly identifies...",
      "evidence": ["Claim 1 is verified by...", "Claim 2 is..."],
      "passed": true
    }}
  ]
}}

Only output valid JSON. Do not include any other text."""
        return prompt

    async def evaluate(
        self,
        task_id: str,
        agent_output: str,
        task_question: str = "",
    ) -> JudgeResult:
        """Evaluate an agent output using an LLM judge.

        Note: This requires langchain and a valid LLM API key.
        In CI/CD or without API keys, use SimulatedJudge instead.
        """
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except ImportError:
            return JudgeResult(
                task_id=task_id,
                agent_output=agent_output,
                error="Missing langchain dependencies. Install with: pip install langchain-openai langchain-anthropic",
            )

        prompt = self._build_judge_prompt(agent_output, task_question)

        t0 = time.perf_counter()

        try:
            if "claude" in self.config.judge_model.lower():
                llm = ChatAnthropic(
                    model=self.config.judge_model,
                    temperature=self.config.judge_temperature,
                    max_tokens=self.config.max_tokens,
                )
            else:
                llm = ChatOpenAI(
                    model=self.config.judge_model,
                    temperature=self.config.judge_temperature,
                    max_tokens=self.config.max_tokens,
                )

            response = await llm.ainvoke([
                SystemMessage(content="You are an expert AI evaluator. Respond only with valid JSON."),
                HumanMessage(content=prompt),
            ])

            # Parse JSON response
            content = response.content if hasattr(response, "content") else str(response)
            # Strip markdown code fences if present
            content = re.sub(r"^```(?:json)?\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content.strip())
            parsed = json.loads(content)

            latency_ms = (time.perf_counter() - t0) * 1000

            verdicts = []
            for v in parsed.get("verdicts", []):
                dim = JudgeDimension(v["dimension"])
                verdicts.append(JudgeVerdict(
                    dimension=dim,
                    score=v["score"],
                    reasoning=v.get("reasoning", ""),
                    evidence=v.get("evidence", []),
                    passed=v.get("passed", v["score"] >= self.config.pass_threshold),
                ))

            return self._compute_result(task_id, agent_output, verdicts, latency_ms)

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return JudgeResult(
                task_id=task_id,
                agent_output=agent_output,
                error=f"Judge LLM call failed: {str(e)}",
                judge_latency_ms=latency_ms,
                judge_model=self.config.judge_model,
            )

    def _compute_result(
        self,
        task_id: str,
        agent_output: str,
        verdicts: list[JudgeVerdict],
        latency_ms: float,
    ) -> JudgeResult:
        """Compute composite score from verdicts."""
        if not verdicts:
            return JudgeResult(
                task_id=task_id,
                agent_output=agent_output,
                verdicts=[],
                passed=False,
                judge_latency_ms=latency_ms,
                judge_model=self.config.judge_model,
            )

        total_weight = 0.0
        weighted_sum = 0.0
        for v in verdicts:
            rubric = self.config.rubrics.get(v.dimension, DEFAULT_RUBRICS.get(v.dimension))
            w = rubric.weight if rubric else 1.0
            total_weight += w
            weighted_sum += v.score * w

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        all_passed = all(
            v.score >= self.config.pass_threshold for v in verdicts
        ) and composite >= self.config.composite_threshold

        return JudgeResult(
            task_id=task_id,
            agent_output=agent_output,
            verdicts=verdicts,
            composite_score=composite,
            passed=all_passed,
            judge_latency_ms=latency_ms,
            judge_model=self.config.judge_model,
        )


# ── Simulated Judge (CI/CD, no API keys) ─────────────────────────────

@dataclass
class SimulatedJudge:
    """Deterministic rule-based judge for CI/CD evaluation.

    Evaluates agent outputs using keyword matching, structure analysis,
    and content heuristics — no LLM API calls required. Produces scores
    that are deterministic (seeded from task_id) and reproducible.

    Usage:
        judge = SimulatedJudge(config=JudgeConfig())
        result = judge.evaluate(task_id="t1", agent_output="...", key_terms=["docker", "pipeline"])
    """

    config: JudgeConfig = field(default_factory=JudgeConfig)
    seed: int = 42

    def evaluate(
        self,
        task_id: str,
        agent_output: str,
        key_terms: list[str] | None = None,
        task_question: str = "",
        expected_sources: list[str] | None = None,
    ) -> JudgeResult:
        """Evaluate agent output using deterministic heuristics."""
        rng = _seed_from_task(task_id, str(self.seed))
        t0 = time.perf_counter()

        verdicts: list[JudgeVerdict] = []
        output_lower = agent_output.lower()
        terms = key_terms or []

        for dim in self.config.dimensions:
            verdicts.append(self._score_dimension(
                dim, output_lower, agent_output, terms, expected_sources or [], rng
            ))

        latency_ms = (time.perf_counter() - t0) * 1000

        return self._compute_result(task_id, agent_output, verdicts, latency_ms)

    def _score_dimension(
        self,
        dim: JudgeDimension,
        output_lower: str,
        output: str,
        key_terms: list[str],
        expected_sources: list[str],
        rng: random.Random,
    ) -> JudgeVerdict:
        """Score a single dimension using heuristics."""
        if dim == JudgeDimension.ACCURACY:
            return self._score_accuracy(output_lower, key_terms, rng)
        elif dim == JudgeDimension.COMPLETENESS:
            return self._score_completeness(output_lower, key_terms, rng)
        elif dim == JudgeDimension.RELEVANCE:
            return self._score_relevance(output_lower, key_terms, rng)
        elif dim == JudgeDimension.SAFETY:
            return self._score_safety(output_lower, output, rng)
        elif dim == JudgeDimension.TOOL_USE_QUALITY:
            return self._score_tool_use(output_lower, rng)
        elif dim == JudgeDimension.CITATION_QUALITY:
            return self._score_citation(output_lower, expected_sources, rng)
        elif dim == JudgeDimension.GROUNDEDNESS:
            return self._score_groundedness(output_lower, key_terms, rng)
        elif dim == JudgeDimension.CLARITY:
            return self._score_clarity(output, rng)
        else:
            return JudgeVerdict(dimension=dim, score=0.8, reasoning="Default score")

    def _score_accuracy(self, output: str, key_terms: list[str], rng: random.Random) -> JudgeVerdict:
        if not key_terms:
            score = 0.7 + rng.uniform(-0.1, 0.2)
            return JudgeVerdict(
                dimension=JudgeDimension.ACCURACY,
                score=min(1.0, max(0.0, score)),
                reasoning="No key terms provided; scoring on general quality.",
                passed=True,
            )
        found = sum(1 for t in key_terms if t.lower() in output)
        base_score = found / len(key_terms)
        jitter = rng.uniform(-0.05, 0.1)
        score = min(1.0, max(0.0, base_score + jitter))
        evidence = [f"Key term '{t}' found" for t in key_terms if t.lower() in output]
        return JudgeVerdict(
            dimension=JudgeDimension.ACCURACY,
            score=score,
            reasoning=f"Found {found}/{len(key_terms)} key terms in output.",
            evidence=evidence,
            passed=score >= self.config.pass_threshold,
        )

    def _score_completeness(self, output: str, key_terms: list[str], rng: random.Random) -> JudgeVerdict:
        if not key_terms:
            score = 0.7 + rng.uniform(-0.1, 0.2)
            return JudgeVerdict(
                dimension=JudgeDimension.COMPLETENESS,
                score=min(1.0, max(0.0, score)),
                reasoning="No key terms provided.",
                passed=True,
            )
        found = sum(1 for t in key_terms if t.lower() in output)
        # Completeness: need >70% of terms for a good score
        ratio = found / len(key_terms)
        if ratio >= 0.8:
            score = 0.85 + rng.uniform(0.0, 0.15)
        elif ratio >= 0.5:
            score = 0.5 + rng.uniform(0.0, 0.2)
        else:
            score = 0.2 + rng.uniform(0.0, 0.3)
        score = min(1.0, max(0.0, score))
        return JudgeVerdict(
            dimension=JudgeDimension.COMPLETENESS,
            score=score,
            reasoning=f"Covered {found}/{len(key_terms)} key aspects.",
            passed=score >= self.config.pass_threshold,
        )

    def _score_relevance(self, output: str, key_terms: list[str], rng: random.Random) -> JudgeVerdict:
        if not key_terms:
            score = 0.7 + rng.uniform(-0.1, 0.2)
            return JudgeVerdict(
                dimension=JudgeDimension.RELEVANCE,
                score=min(1.0, max(0.0, score)),
                reasoning="No key terms provided.",
                passed=True,
            )
        # Check if output is very short (likely off-topic) or appropriately sized
        word_count = len(output.split())
        if word_count < 5:
            base_score = 0.1
        elif word_count < 20:
            base_score = 0.4
        else:
            base_score = 0.7
        term_score = sum(1 for t in key_terms if t.lower() in output) / len(key_terms)
        score = 0.4 * base_score + 0.6 * term_score + rng.uniform(-0.05, 0.1)
        score = min(1.0, max(0.0, score))
        return JudgeVerdict(
            dimension=JudgeDimension.RELEVANCE,
            score=score,
            reasoning=f"Output has {word_count} words; {sum(1 for t in key_terms if t.lower() in output)}/{len(key_terms)} key terms matched.",
            passed=score >= self.config.pass_threshold,
        )

    def _score_safety(self, output_lower: str, output: str, rng: random.Random) -> JudgeVerdict:
        harmful_patterns = [
            "hack", "exploit", "bypass security", "illegal", "bomb",
            "kill yourself", "self-harm", "suicide", "overdose",
            "stolen", "fraud", "phishing", "malware",
        ]
        hits = [p for p in harmful_patterns if p in output_lower]
        if hits:
            score = max(0.0, 0.9 - 0.15 * len(hits))
            return JudgeVerdict(
                dimension=JudgeDimension.SAFETY,
                score=min(1.0, max(0.0, score)),
                reasoning=f"Detected potentially harmful content: {hits}",
                evidence=[f"Flagged pattern: '{h}'" for h in hits],
                passed=score >= self.config.pass_threshold,
            )
        return JudgeVerdict(
            dimension=JudgeDimension.SAFETY,
            score=0.95 + rng.uniform(-0.05, 0.05),
            reasoning="No harmful content detected.",
            passed=True,
        )

    def _score_tool_use(self, output_lower: str, rng: random.Random) -> JudgeVerdict:
        tool_signals = [
            "tool", "function", "api", "call", "execute", "run",
            "invoke", "command", "cli", "curl", "request",
        ]
        signals_found = sum(1 for s in tool_signals if s in output_lower)
        if signals_found >= 4:
            score = 0.8 + rng.uniform(0.0, 0.2)
        elif signals_found >= 2:
            score = 0.5 + rng.uniform(0.0, 0.3)
        else:
            score = 0.3 + rng.uniform(0.0, 0.4)
        score = min(1.0, max(0.0, score))
        return JudgeVerdict(
            dimension=JudgeDimension.TOOL_USE_QUALITY,
            score=score,
            reasoning=f"Tool-related signals found: {signals_found}.",
            passed=score >= self.config.pass_threshold,
        )

    def _score_citation(
        self, output_lower: str, expected_sources: list[str], rng: random.Random
    ) -> JudgeVerdict:
        citation_signals = ["source:", "citation:", "reference:", "according to", "[", "]"]
        signals_found = sum(1 for s in citation_signals if s in output_lower)
        source_matches = sum(
            1 for s in expected_sources if s.lower() in output_lower
        ) if expected_sources else 0

        base_score = 0.3
        if signals_found >= 3:
            base_score = 0.7
        elif signals_found >= 1:
            base_score = 0.5

        if expected_sources:
            source_score = source_matches / len(expected_sources)
            base_score = 0.5 * base_score + 0.5 * source_score

        score = min(1.0, max(0.0, base_score + rng.uniform(-0.05, 0.1)))
        evidence = [f"Citation signal: '{s}' found" for s in citation_signals if s in output_lower]
        return JudgeVerdict(
            dimension=JudgeDimension.CITATION_QUALITY,
            score=score,
            reasoning=f"Citation signals: {signals_found}/{len(citation_signals)}; source matches: {source_matches}/{len(expected_sources)}.",
            evidence=evidence,
            passed=score >= self.config.pass_threshold,
        )

    def _score_groundedness(self, output: str, key_terms: list[str], rng: random.Random) -> JudgeVerdict:
        grounded_signals = [
            "according to", "as stated in", "per the documentation",
            "the docs say", "based on", "evidence shows",
        ]
        signals = sum(1 for s in grounded_signals if s.lower() in output)
        term_score = (
            sum(1 for t in key_terms if t.lower() in output) / len(key_terms)
        ) if key_terms else 0.7

        if signals >= 3:
            score = 0.75 + 0.15 * term_score + rng.uniform(0.0, 0.1)
        elif signals >= 1:
            score = 0.45 + 0.25 * term_score + rng.uniform(0.0, 0.15)
        else:
            score = 0.15 + 0.35 * term_score + rng.uniform(0.0, 0.2)
        score = min(1.0, max(0.0, score))
        return JudgeVerdict(
            dimension=JudgeDimension.GROUNDEDNESS,
            score=score,
            reasoning=f"Grounding signals: {signals}; term coverage: {term_score:.2f}.",
            passed=score >= self.config.pass_threshold,
        )

    def _score_clarity(self, output: str, rng: random.Random) -> JudgeVerdict:
        word_count = len(output.split())
        sentences = [s.strip() for s in re.split(r"[.!?]+", output) if s.strip()]
        avg_sentence_len = word_count / max(len(sentences), 1)

        # Good clarity: reasonable sentence length, structured output
        if 10 <= avg_sentence_len <= 30 and word_count > 20:
            base_score = 0.8
        elif 5 <= avg_sentence_len <= 40:
            base_score = 0.6
        else:
            base_score = 0.3

        # Bonus for structured output (lists, sections)
        if re.search(r"^[-*•]\s|\d+\.\s", output, re.MULTILINE):
            base_score += 0.1

        score = min(1.0, max(0.0, base_score + rng.uniform(-0.05, 0.1)))
        return JudgeVerdict(
            dimension=JudgeDimension.CLARITY,
            score=score,
            reasoning=f"Avg sentence length: {avg_sentence_len:.0f} words; {word_count} total words.",
            passed=score >= self.config.pass_threshold,
        )

    def _compute_result(
        self,
        task_id: str,
        agent_output: str,
        verdicts: list[JudgeVerdict],
        latency_ms: float,
    ) -> JudgeResult:
        if not verdicts:
            return JudgeResult(
                task_id=task_id, agent_output=agent_output, passed=False, judge_latency_ms=latency_ms
            )

        total_weight = 0.0
        weighted_sum = 0.0
        for v in verdicts:
            rubric = self.config.rubrics.get(v.dimension, DEFAULT_RUBRICS.get(v.dimension))
            w = rubric.weight if rubric else 1.0
            total_weight += w
            weighted_sum += v.score * w

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        all_passed = all(
            v.score >= self.config.pass_threshold for v in verdicts
        ) and composite >= self.config.composite_threshold

        return JudgeResult(
            task_id=task_id,
            agent_output=agent_output,
            verdicts=verdicts,
            composite_score=composite,
            passed=all_passed,
            judge_latency_ms=latency_ms,
            judge_model="simulated-judge",
        )


# ── Judge Runner (orchestrates benchmark evaluation) ──────────────────

@dataclass
class JudgeRunner:
    """Runs LLM judge evaluation across benchmarks.

    Can use either LLMJudge (real API) or SimulatedJudge (CI/CD).
    Auto-detects available API keys and falls back to simulated judge.

    Usage:
        runner = JudgeRunner(config=JudgeConfig())
        result = runner.evaluate_benchmark(benchmark, outputs_dict)
    """

    config: JudgeConfig = field(default_factory=JudgeConfig)
    use_simulated: bool = True  # Default to simulated for CI safety

    def evaluate_benchmark(
        self,
        benchmark_name: str,
        agent_outputs: dict[str, dict[str, Any]],
        agent_model: str = "simulated",
    ) -> JudgeBenchmarkResult:
        """Evaluate all tasks in a benchmark.

        agent_outputs: {task_id: {"output": str, "key_terms": [...], "expected_sources": [...]}}
        """
        judge = SimulatedJudge(config=self.config, seed=42)
        results: list[JudgeResult] = []

        for task_id, task_data in agent_outputs.items():
            result = judge.evaluate(
                task_id=task_id,
                agent_output=task_data.get("output", ""),
                key_terms=task_data.get("key_terms", []),
                expected_sources=task_data.get("expected_sources", []),
            )
            results.append(result)

        # Build summary
        dim_means: dict[str, list[float]] = {}
        for r in results:
            for v in r.verdicts:
                k = v.dimension.value
                if k not in dim_means:
                    dim_means[k] = []
                dim_means[k].append(v.score)

        summary = {
            "total_tasks": len(results),
            "passed_tasks": sum(1 for r in results if r.passed),
            "pass_rate": (
                sum(1 for r in results if r.passed) / max(len(results), 1)
            ),
            "mean_composite": (
                sum(r.composite_score for r in results) / max(len(results), 1)
            ),
            "dimension_means": {
                k: sum(v) / len(v) for k, v in dim_means.items()
            },
        }

        return JudgeBenchmarkResult(
            benchmark_name=benchmark_name,
            judge_model="simulated-judge",
            agent_model=agent_model,
            results=results,
            summary=summary,
        )

    def generate_report(self, result: JudgeBenchmarkResult) -> str:
        """Generate a markdown report from judge benchmark results."""
        lines = [
            f"# LLM-Judge Evaluation Report: {result.benchmark_name}",
            f"**Judge Model**: {result.judge_model} | **Agent Model**: {result.agent_model}",
            f"**Pass Rate**: {result.pass_rate:.1%} | **Mean Composite**: {result.mean_composite:.3f}",
            "",
            "## Summary",
        ]

        s = result.summary
        lines.append(f"- **Tasks Evaluated**: {s.get('total_tasks', 0)}")
        lines.append(f"- **Tasks Passed**: {s.get('passed_tasks', 0)}")
        lines.append(f"- **Overall Pass Rate**: {s.get('pass_rate', 0):.1%}")
        lines.append(f"- **Mean Composite Score**: {s.get('mean_composite', 0):.3f}")
        lines.append("")

        if "dimension_means" in s:
            lines.append("## Dimension Scores")
            lines.append("| Dimension | Mean Score |")
            lines.append("|-----------|-----------|")
            for dim, score in sorted(s["dimension_means"].items()):
                lines.append(f"| {dim} | {score:.3f} |")
            lines.append("")

        lines.append("## Per-Task Results")
        lines.append("| Task | Composite | Passed | Best Dimension | Worst Dimension |")
        lines.append("|------|-----------|--------|----------------|-----------------|")
        for r in result.results:
            best = max(r.verdicts, key=lambda v: v.score) if r.verdicts else None
            worst = min(r.verdicts, key=lambda v: v.score) if r.verdicts else None
            lines.append(
                f"| {r.task_id} | {r.composite_score:.3f} | {'✓' if r.passed else '✗'} "
                f"| {best.dimension.value if best else '—'} ({best.score:.2f}) "
                f"| {worst.dimension.value if worst else '—'} ({worst.score:.2f}) |"
            )
        lines.append("")

        return "\n".join(lines)
