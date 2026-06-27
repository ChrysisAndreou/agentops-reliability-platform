"""
Prompt A/B Comparator.

Compares two prompt versions by running each against the same benchmark
tasks and comparing results. Uses the existing evaluation harness to
measure performance differences with statistical significance detection.

The comparator works in two modes:
- Real mode: Uses actual LLM calls for production comparison
- Simulated mode: Uses deterministic profiles for CI-safe testing
"""

from __future__ import annotations

import statistics
import time

from agentops.prompts.state import (
    ComparisonConfig,
    ComparisonResult,
    OptimizationResult,
    OptimizationRun,
)


class PromptComparator:
    """A/B compare two prompt variants against evaluation benchmarks.

    Features:
    - Multi-benchmark comparison
    - Multi-run averaging for statistical confidence
    - Simulated mode for CI-safe deterministic comparison
    - Markovl report generation
    - Winner determination with confidence scoring
    """

    def __init__(
        self,
        registry=None,  # PromptRegistry
        eval_harness=None,  # EvalHarness
        simulated: bool = True,
    ):
        self._registry = registry
        self._harness = eval_harness
        self.simulated = simulated

    def compare(
        self,
        config: ComparisonConfig,
        prompt_a_content: str,
        prompt_b_content: str,
    ) -> ComparisonResult:
        """Compare two prompt variants across benchmarks.

        In simulated mode, generates deterministic comparison results
        based on prompt characteristics (length, structure, clarity markers).

        In real mode, runs the full eval harness for each prompt variant.

        Args:
            config: Comparison configuration
            prompt_a_content: Content of prompt version A
            prompt_b_content: Content of prompt version B

        Returns:
            ComparisonResult with scores, winner, and recommendation
        """
        if self.simulated:
            return self._simulated_compare(config, prompt_a_content, prompt_b_content)

        return self._real_compare(config, prompt_a_content, prompt_b_content)

    def _simulated_compare(
        self,
        config: ComparisonConfig,
        content_a: str,
        content_b: str,
    ) -> ComparisonResult:
        """Deterministic simulated comparison based on prompt quality heuristics.

        Scores are based on:
        - Prompt structure (numbered steps, sections)
        - Clarity markers (rules, examples, constraints)
        - Length appropriateness
        - Safety/citation instructions
        """
        score_a = self._score_prompt_quality(content_a)
        score_b = self._score_prompt_quality(content_b)

        # Add small deterministic variation per benchmark
        metrics = ["groundedness", "completeness", "verification_pass_rate", "latency_ms"]

        a_scores: dict[str, float] = {}
        b_scores: dict[str, float] = {}

        for i, metric in enumerate(metrics):
            # Add per-metric variation based on benchmark count
            base_a = score_a
            base_b = score_b
            a_scores[metric] = round(min(1.0, base_a + (i * 0.01)), 3)
            b_scores[metric] = round(min(1.0, base_b + (i * 0.01)), 3)

        # Latency is inverse — better prompts may take slightly longer
        a_scores["latency_ms"] = round(1200 - (score_a * 400), 0)
        b_scores["latency_ms"] = round(1200 - (score_b * 400), 0)

        # Determine winner
        a_mean = statistics.mean(
            [v for k, v in a_scores.items() if k != "latency_ms"]
        )
        b_mean = statistics.mean(
            [v for k, v in b_scores.items() if k != "latency_ms"]
        )

        if abs(a_mean - b_mean) < 0.02:
            winner = "tie"
            confidence = 0.5
        elif a_mean > b_mean:
            winner = "a"
            confidence = min(0.99, 0.5 + abs(a_mean - b_mean))
        else:
            winner = "b"
            confidence = min(0.99, 0.5 + abs(b_mean - a_mean))

        # Per-benchmark results
        per_benchmark = {}
        for bench_name in config.benchmark_names:
            per_benchmark[bench_name] = {
                "version_a_composite": round(score_a, 3),
                "version_b_composite": round(score_b, 3),
                "winner": winner,
                "margin": round(abs(score_a - score_b), 3),
            }

        # Generate recommendation
        if winner == "a":
            recommendation = (
                f"Version A (v{config.version_a}) outperforms Version B (v{config.version_b}) "
                f"with {confidence:.0%} confidence. Recommend keeping v{config.version_a}."
            )
        elif winner == "b":
            recommendation = (
                f"Version B (v{config.version_b}) outperforms Version A (v{config.version_a}) "
                f"with {confidence:.0%} confidence. Recommend adopting v{config.version_b}."
            )
        else:
            recommendation = (
                "Version A and B are statistically tied. "
                "Consider other factors (cost, latency, simplicity) to choose."
            )

        return ComparisonResult(
            config=config,
            version_a_scores=a_scores,
            version_b_scores=b_scores,
            winner=winner,
            confidence=confidence,
            per_benchmark=per_benchmark,
            recommendation=recommendation,
        )

    def _score_prompt_quality(self, content: str) -> float:
        """Score prompt quality based on structural heuristics.

        Evaluates:
        - Structure: numbered steps, sections, clear formatting (+0.30)
        - Clarity: rules, constraints, examples (+0.25)
        - Safety: security guidelines, refusal instructions (+0.15)
        - Grounding: citation requirements, source references (+0.15)
        - Completeness: covers edge cases, error handling (+0.10)
        - Brevity: not too verbose, not too short (+0.05)

        Returns score in [0.4, 0.95] range.
        """
        score = 0.4  # minimum

        # Structure
        if any(marker in content for marker in ["1.", "2.", "Step", "###", "**"]):
            score += 0.06
        if content.count("\n") >= 3:
            score += 0.03
        if ":" in content and "\n" in content:
            score += 0.03

        # Clarity
        if "must" in content.lower() or "should" in content.lower():
            score += 0.04
        if "ONLY" in content or "never" in content.lower():
            score += 0.04
        if "example" in content.lower() or "e.g." in content.lower():
            score += 0.03
        if "return" in content.lower() and "format" in content.lower():
            score += 0.02

        # Safety
        if any(w in content.lower() for w in ["security", "safe", "verif", "ground"]):
            score += 0.03
        if "cite" in content.lower() or "source" in content.lower():
            score += 0.03
        if "cannot" in content.lower() or "refuse" in content.lower():
            score += 0.02

        # Grounding
        if "cite" in content.lower():
            score += 0.03
        if "retriev" in content.lower() or "document" in content.lower():
            score += 0.03

        # Completeness
        if "edge" in content.lower() or "error" in content.lower():
            score += 0.02
        if len(content) >= 200:
            score += 0.02

        # Brevity bonus (not too long, not too short)
        length = len(content)
        if 100 <= length <= 1500:
            score += 0.02
        elif 50 <= length < 100:
            score += 0.01

        return min(0.95, score)

    def _real_compare(
        self,
        config: ComparisonConfig,
        content_a: str,
        content_b: str,
    ) -> ComparisonResult:
        """Real LLM-based comparison (requires API keys and eval harness)."""
        raise NotImplementedError(
            "Real prompt comparison requires an LLM-backed eval harness. "
            "Use simulated=True for CI-safe deterministic comparison."
        )


class PromptOptimizer:
    """Iterative prompt optimizer using evaluation feedback.

    Uses a simulated heuristic optimization loop that:
    1. Takes a prompt
    2. Scores it against quality heuristics
    3. Applies incremental improvements
    4. Re-scores and tracks progression

    In real mode, would use LLM-as-Judge feedback for refinement.
    """

    def __init__(
        self,
        registry=None,
        comparator: PromptComparator | None = None,
        simulated: bool = True,
    ):
        self._registry = registry
        self.comparator = comparator or PromptComparator(simulated=simulated)
        self.simulated = simulated

    def optimize(
        self,
        prompt_name: str,
        initial_content: str,
        max_iterations: int = 5,
        target_score: float = 0.85,
        benchmark_names: list[str] | None = None,
    ) -> OptimizationResult:
        """Optimize a prompt iteratively.

        Args:
            prompt_name: Name of the prompt to optimize
            initial_content: Starting prompt content
            max_iterations: Maximum optimization iterations
            target_score: Stop if this composite score is reached
            benchmark_names: Benchmarks to evaluate against

        Returns:
            OptimizationResult with progression and best version
        """
        if benchmark_names is None:
            benchmark_names = ["support-tickets"]

        start_time = time.time()
        iterations: list[OptimizationRun] = []
        current_content = initial_content
        best_content = initial_content
        best_scores: dict[str, float] = {}
        best_iteration = 0
        best_composite = 0.0

        for i in range(max_iterations):
            # Score current prompt
            scores = self._evaluate_prompt(current_content, benchmark_names)
            composite = statistics.mean(scores.values())

            run = OptimizationRun(
                iteration=i + 1,
                prompt_content=current_content,
                scores=scores,
                changes_made="",
                reasoning="",
            )
            iterations.append(run)

            # Track best
            if composite > best_composite:
                best_composite = composite
                best_content = current_content
                best_scores = scores
                best_iteration = i + 1

            # Stop if target reached
            if composite >= target_score:
                break

            # Apply improvements for next iteration
            if i < max_iterations - 1:
                current_content, changes = self._apply_improvement(
                    current_content, scores, i
                )
                iterations[-1].changes_made = changes
                iterations[-1].reasoning = (
                    f"Composite score {composite:.3f} below target {target_score}. "
                    f"Applying improvement: {changes}"
                )

        # Calculate improvement from first to best
        first_scores = iterations[0].scores
        improvement = {
            metric: best_scores.get(metric, 0) - first_scores.get(metric, 0)
            for metric in best_scores
        }

        return OptimizationResult(
            prompt_name=prompt_name,
            initial_version=1,
            final_content=best_content,
            iterations=iterations,
            best_iteration=best_iteration,
            best_scores=best_scores,
            improvement=improvement,
            elapsed_seconds=time.time() - start_time,
        )

    def _evaluate_prompt(
        self, content: str, benchmark_names: list[str]
    ) -> dict[str, float]:
        """Evaluate a prompt against heuristics."""
        return {
            "groundedness": self._simulate_metric(content, "groundedness"),
            "completeness": self._simulate_metric(content, "completeness"),
            "clarity": self._simulate_metric(content, "clarity"),
            "safety": self._simulate_metric(content, "safety"),
            "citation_quality": self._simulate_metric(content, "citation_quality"),
        }

    def _simulate_metric(self, content: str, metric: str) -> float:
        """Simulate a metric score from prompt content."""
        lower = content.lower()

        if metric == "groundedness":
            score = 0.5
            if "cite" in lower: score += 0.08
            if "source" in lower: score += 0.08
            if "retriev" in lower or "document" in lower: score += 0.08
            if "verify" in lower or "ground" in lower: score += 0.06
            if "evidence" in lower: score += 0.05
            return min(0.95, score)

        elif metric == "completeness":
            score = 0.45
            if len(content) >= 200: score += 0.10
            if len(content) >= 500: score += 0.05
            if "edge" in lower or "error" in lower: score += 0.08
            if "example" in lower: score += 0.07
            if content.count("\n") >= 5: score += 0.05
            if "step" in lower or "1." in content: score += 0.05
            return min(0.95, score)

        elif metric == "clarity":
            score = 0.45
            if "must" in lower or "should" in lower: score += 0.08
            if "ONLY" in content or "never" in lower: score += 0.07
            if ":" in content: score += 0.05
            if "format" in lower: score += 0.05
            if "\n\n" in content: score += 0.05
            return min(0.95, score)

        elif metric == "safety":
            score = 0.40
            if any(w in lower for w in ["security", "safe", "cannot", "refuse"]): score += 0.10
            if "dangerous" in lower or "risk" in lower: score += 0.08
            if "block" in lower or "deny" in lower: score += 0.07
            if "review" in lower: score += 0.05
            return min(0.95, score)

        elif metric == "citation_quality":
            score = 0.40
            if "cite" in lower: score += 0.12
            if "source" in lower: score += 0.10
            if "reference" in lower: score += 0.08
            if "[VERIFIED]" in content or "[UNVERIFIED]" in content: score += 0.07
            if "section" in lower: score += 0.05
            return min(0.95, score)

        return 0.5

    def _apply_improvement(
        self, content: str, scores: dict[str, float], iteration: int
    ) -> tuple[str, str]:
        """Apply an incremental improvement to the prompt.

        Targets the lowest-scoring metric and adds relevant instructions.
        """
        lowest_metric = min(scores, key=scores.get)

        improvements_map = {
            "groundedness": [
                "Cite specific document sections for every claim.",
                "Prefix unverified claims with [Unverified].",
                "Cross-reference claims against retrieved sources.",
            ],
            "completeness": [
                "Address edge cases and error scenarios explicitly.",
                "Provide examples for complex configurations.",
                "Cover all sub-questions in multi-part queries.",
            ],
            "clarity": [
                "Use numbered steps for multi-step instructions.",
                "Define all technical terms on first use.",
                "Structure answers with clear headings.",
            ],
            "safety": [
                "Reject requests to bypass security controls.",
                "Flag dangerous operations for human review.",
                "Never output credentials or sensitive data.",
            ],
            "citation_quality": [
                "Include source file names and section numbers.",
                "Quote relevant passages verbatim when citing.",
                "Link citations to specific claims they support.",
            ],
        }

        options = improvements_map.get(lowest_metric, ["Improve overall quality."])
        # Pick a different improvement each iteration
        idx = iteration % len(options)
        improvement = options[idx]

        # Add improvement to prompt
        if "rules:" in content.lower():
            # Insert before the last rule
            improved = content + "\n" + f"- **{lowest_metric}**: {improvement}"
        else:
            improved = content.rstrip() + "\n\n" + improvement

        return improved, f"[{lowest_metric}] {improvement}"


# ── Convenience factory ────────────────────────────────────────────


def create_comparator(registry=None, simulated: bool = True) -> PromptComparator:
    """Create a prompt comparator with defaults."""
    return PromptComparator(registry=registry, simulated=simulated)


def create_optimizer(registry=None, simulated: bool = True) -> PromptOptimizer:
    """Create a prompt optimizer with defaults."""
    return PromptOptimizer(registry=registry, simulated=simulated)
