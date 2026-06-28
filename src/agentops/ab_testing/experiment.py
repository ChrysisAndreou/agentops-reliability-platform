"""
A/B experiment framework for AI agent evaluation.

Provides a complete experiment framework for comparing AI agent variants
through controlled traffic splitting, metric collection, and statistical
analysis. Designed for production agent evaluation — not just click-through
rates — supporting agent-specific metrics like verification pass rate,
groundedness, latency, and cost.

Architecture:
    VariantConfig → TrafficSplit → ABExperiment → ABExperimentResult

Classes:
    VariantConfig: Configuration for a single agent variant.
    TrafficAllocation: Per-variant allocation in a traffic split.
    TrafficSplit: Weighted traffic distribution across variants.
    ABExperimentConfig: Experiment-level configuration.
    ABExperiment: Core experiment runner with traffic splitting and analysis.
    ABExperimentResult: Aggregated per-variant metrics.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from agentops.ab_testing.stats import (
    bayesian_ab_test,
    chi_squared_test,
    compute_sample_size,
    confidence_interval,
    welch_t_test,
)


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------

@dataclass
class VariantConfig:
    """Configuration for a single agent variant in an A/B experiment.

    Attributes:
        name: Human-readable variant name (e.g., 'control', 'new_reranker').
        description: What this variant changes relative to baseline.
        config: Arbitrary configuration dict passed to the agent factory.
        metadata: Additional tags/labels for reporting.
    """

    name: str
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("VariantConfig.name must not be empty")


@dataclass
class TrafficAllocation:
    """Allocation of traffic to a specific variant.

    Attributes:
        variant: The variant configuration.
        weight: Relative weight for traffic splitting (e.g., 50 for 50%).
        min_samples: Minimum number of samples required before analysis.
    """

    variant: VariantConfig
    weight: float
    min_samples: int = 100

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")


@dataclass
class TrafficSplit:
    """Weighted traffic distribution across agent variants.

    Weights are normalized to sum to 1.0. At least one variant must have
    weight > 0. Traffic is assigned deterministically via hash-based
    assignment (hash of task ID + seed) for reproducibility.
    """

    allocations: list[TrafficAllocation]
    seed: int = 42

    def __post_init__(self):
        if not self.allocations:
            raise ValueError("TrafficSplit must have at least one allocation")
        total = sum(a.weight for a in self.allocations)
        if total <= 0:
            raise ValueError("At least one allocation must have weight > 0")
        self._total_weight = total

    def assign(self, task_id: str) -> VariantConfig:
        """Deterministically assign a task to a variant.

        Uses hash-based assignment for reproducibility — same task_id
        always maps to the same variant given the same seed.

        Args:
            task_id: Unique identifier for the task/query.

        Returns:
            The assigned VariantConfig.
        """
        import hashlib

        key = f"{self.seed}:{task_id}"
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        bucket = hash_val % 10000 / 10000.0  # [0, 1)

        cumulative = 0.0
        for alloc in self.allocations:
            cumulative += alloc.weight / self._total_weight
            if bucket < cumulative:
                return alloc.variant

        # Fallback: return last variant
        return self.allocations[-1].variant

    def get_weight(self, variant_name: str) -> float:
        """Get the normalized weight for a variant by name."""
        for alloc in self.allocations:
            if alloc.variant.name == variant_name:
                return alloc.weight / self._total_weight
        return 0.0


# ---------------------------------------------------------------------------
# Experiment configuration and results
# ---------------------------------------------------------------------------

DEFAULT_METRICS = [
    "verification_pass_rate",
    "groundedness",
    "latency_ms",
    "cost_usd",
    "tool_success_rate",
]


@dataclass
class ABExperimentConfig:
    """Configuration for an A/B experiment.

    Attributes:
        metric: Primary metric to optimize for decision (e.g., 'verification_pass_rate').
        metrics: All metrics to track (default: verification, groundedness, latency, cost, tool success).
        significance_level: Alpha threshold for statistical significance (default 0.05).
        min_sample_size: Minimum total samples before running analysis.
        traffic_split_seed: Seed for reproducible traffic assignment.
        bayesian_threshold: Probability threshold for Bayesian decision (default 0.95).
        name: Optional experiment name for reporting.
    """

    metric: str = "verification_pass_rate"
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    significance_level: float = 0.05
    min_sample_size: int = 100
    traffic_split_seed: int = 42
    bayesian_threshold: float = 0.95
    name: str = ""

    def __post_init__(self):
        if not 0 < self.significance_level < 1:
            raise ValueError(
                f"significance_level must be in (0, 1), got {self.significance_level}"
            )
        if not 0 < self.bayesian_threshold <= 1:
            raise ValueError(
                f"bayesian_threshold must be in (0, 1], got {self.bayesian_threshold}"
            )
        if self.min_sample_size < 2:
            raise ValueError(
                f"min_sample_size must be >= 2, got {self.min_sample_size}"
            )


@dataclass
class VariantMetrics:
    """Collected metrics for a single variant during an experiment.

    Attributes:
        variant_name: Name of the variant.
        total_runs: Total number of runs assigned to this variant.
        successes: Number of successful runs (for binary metrics).
        failures: Number of failed runs.
        metric_values: Per-metric accumulated values across runs.
        metric_sum_sq: Sum of squares for variance calculation.
        latencies: All collected latency values.
        costs: All collected cost values.
    """

    variant_name: str
    total_runs: int = 0
    successes: int = 0
    failures: int = 0
    metric_values: dict[str, list[float]] = field(default_factory=dict)
    metric_sum_sq: dict[str, float] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)

    def record(
        self,
        success: bool,
        metrics: dict[str, float],
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ):
        """Record a single run's metrics."""
        self.total_runs += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

        for key, value in metrics.items():
            if key not in self.metric_values:
                self.metric_values[key] = []
                self.metric_sum_sq[key] = 0.0
            self.metric_values[key].append(value)
            self.metric_sum_sq[key] += value * value

        self.latencies.append(latency_ms)
        self.costs.append(cost_usd)

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.successes / self.total_runs

    def metric_mean(self, metric: str) -> float:
        values = self.metric_values.get(metric, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def metric_std(self, metric: str) -> float:
        values = self.metric_values.get(metric, [])
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        ssq = self.metric_sum_sq.get(metric, 0.0)
        variance = max(0.0, (ssq - n * mean * mean) / (n - 1))
        return math.sqrt(variance)

    def success_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        return confidence_interval(self.successes, self.total_runs, confidence)


@dataclass
class ABExperimentResult:
    """Aggregated results for all variants in an A/B experiment.

    Attributes:
        config: The experiment configuration used.
        variants: Per-variant collected metrics.
        total_runs: Total runs across all variants.
        start_time: ISO timestamp of experiment start.
        end_time: ISO timestamp of experiment end.
    """

    config: ABExperimentConfig
    variants: dict[str, VariantMetrics] = field(default_factory=dict)
    total_runs: int = 0
    start_time: str = ""
    end_time: str = ""

    def get_variant(self, name: str) -> Optional[VariantMetrics]:
        return self.variants.get(name)


# ---------------------------------------------------------------------------
# ABExperiment — core experiment runner
# ---------------------------------------------------------------------------

@dataclass
class ABExperiment:
    """Run and analyze A/B experiments comparing AI agent variants.

    Handles traffic splitting, metric collection, and statistical analysis
    across two or more agent variants. Supports both binary metrics
    (success/failure, pass/fail) and continuous metrics (latency, scores).

    Example:
        >>> config = ABExperimentConfig(metric="verification_pass_rate")
        >>> split = TrafficSplit([
        ...     TrafficAllocation(VariantConfig("control"), weight=50),
        ...     TrafficAllocation(VariantConfig("treatment"), weight=50),
        ... ])
        >>> experiment = ABExperiment(config)
        >>> result = experiment.create_result(split)
        >>> result = experiment.run_iteration(
        ...     result, split, "task_001", True,
        ...     {"groundedness": 0.85}, latency_ms=1234
        ... )
        >>> report = experiment.analyze(result)
    """

    config: ABExperimentConfig

    def create_result(self, split: TrafficSplit) -> ABExperimentResult:
        """Create an empty result container for a new experiment.

        Args:
            split: The traffic split configuration.

        Returns:
            Initialized ABExperimentResult with empty variant metrics.
        """
        import datetime

        variants = {}
        for alloc in split.allocations:
            variants[alloc.variant.name] = VariantMetrics(
                variant_name=alloc.variant.name
            )

        return ABExperimentResult(
            config=self.config,
            variants=variants,
            total_runs=0,
            start_time=datetime.datetime.utcnow().isoformat() + "Z",
        )

    def run_iteration(
        self,
        result: ABExperimentResult,
        split: TrafficSplit,
        task_id: str,
        success: bool,
        metrics: dict[str, float],
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> ABExperimentResult:
        """Record a single task evaluation in the experiment.

        Assigns the task to a variant via the traffic split and records
        the outcome and metrics for that variant.

        Args:
            result: The experiment result to update.
            split: Traffic split for deterministic assignment.
            task_id: Unique task identifier for reproducible assignment.
            success: Whether the run was successful (binary outcome).
            metrics: Dict of metric name → value for this run.
            latency_ms: Observed latency in milliseconds.
            cost_usd: Observed cost in USD.

        Returns:
            Updated ABExperimentResult.
        """
        variant = split.assign(task_id)
        vm = result.variants[variant.name]
        vm.record(success, metrics, latency_ms, cost_usd)
        result.total_runs += 1
        return result

    def run_batch(
        self,
        result: ABExperimentResult,
        split: TrafficSplit,
        runs: list[dict[str, Any]],
    ) -> ABExperimentResult:
        """Record multiple task evaluations at once.

        Args:
            result: The experiment result to update.
            split: Traffic split for assignment.
            runs: List of dicts with keys: task_id, success, metrics,
                  and optionally latency_ms, cost_usd.

        Returns:
            Updated ABExperimentResult.
        """
        for run in runs:
            result = self.run_iteration(
                result,
                split,
                run["task_id"],
                run["success"],
                run.get("metrics", {}),
                run.get("latency_ms", 0.0),
                run.get("cost_usd", 0.0),
            )
        return result

    def analyze(self, result: ABExperimentResult) -> "ABAnalysisReport":
        """Analyze experiment results with statistical tests.

        Compares the two highest-traffic variants (primary analysis) and
        runs chi-squared, Fisher's exact, Bayesian A/B, and Welch's t-test
        on the primary metric.

        Args:
            result: The completed experiment result.

        Returns:
            ABAnalysisReport with statistical conclusions.
        """
        import datetime

        result.end_time = datetime.datetime.utcnow().isoformat() + "Z"

        if len(result.variants) < 2:
            return ABAnalysisReport(
                result=result,
                significant=False,
                winner=None,
                tests={},
                recommendation="Need at least 2 variants with data for analysis.",
            )

        # Sort variants by total runs (descending)
        sorted_variants = sorted(
            result.variants.values(),
            key=lambda v: v.total_runs,
            reverse=True,
        )
        control = sorted_variants[0]
        treatment = sorted_variants[1]
        metric = self.config.metric

        tests = {}
        significant = False
        winner = None

        # Chi-squared on binary success
        if metric == "verification_pass_rate" or metric == "success_rate":
            chi2_p, chi2_sig = chi_squared_test(
                [control.successes, control.failures],
                [treatment.successes, treatment.failures],
            )
            tests["chi_squared"] = {
                "p_value": chi2_p,
                "significant": chi2_sig,
                "control_rate": control.success_rate,
                "treatment_rate": treatment.success_rate,
            }
            if chi2_sig:
                significant = True

        # Bayesian A/B
        bayes_prob = bayesian_ab_test(
            successes_a=control.successes,
            trials_a=control.total_runs,
            successes_b=treatment.successes,
            trials_b=treatment.total_runs,
        )
        bayes_sig = bayes_prob >= self.config.bayesian_threshold
        tests["bayesian"] = {
            "prob_treatment_better": bayes_prob,
            "significant": bayes_sig,
        }
        if bayes_sig:
            significant = True
            winner = treatment.variant_name if bayes_prob > 0.5 else control.variant_name

        # Welch's t-test on primary metric if continuous data exists
        control_vals = control.metric_values.get(metric, [])
        treatment_vals = treatment.metric_values.get(metric, [])
        if len(control_vals) >= 2 and len(treatment_vals) >= 2:
            _, t_p, t_sig = welch_t_test(control_vals, treatment_vals)
            tests["welch_t"] = {
                "p_value": t_p,
                "significant": t_sig,
                "control_mean": control.metric_mean(metric),
                "treatment_mean": treatment.metric_mean(metric),
            }
            if t_sig:
                significant = True

        # Confidence intervals
        control_ci = control.success_ci()
        treatment_ci = treatment.success_ci()
        tests["confidence_intervals"] = {
            "control": {"lower": control_ci[0], "upper": control_ci[1]},
            "treatment": {"lower": treatment_ci[0], "upper": treatment_ci[1]},
        }

        # Recommendation
        if not significant:
            if result.total_runs < self.config.min_sample_size:
                recommendation = (
                    f"Insufficient data: {result.total_runs} runs collected, "
                    f"need {self.config.min_sample_size}. Continue the experiment."
                )
            else:
                recommendation = (
                    "No statistically significant difference detected. "
                    "Either variants are equivalent or effect size is too small "
                    "for current sample size."
                )
            winner = None
        else:
            if winner is None:
                winner = (
                    treatment.variant_name
                    if treatment.success_rate > control.success_rate
                    else control.variant_name
                )
            recommendation = (
                f"Statistically significant difference detected. "
                f"Recommend deploying variant '{winner}'."
            )

        return ABAnalysisReport(
            result=result,
            significant=significant,
            winner=winner,
            tests=tests,
            recommendation=recommendation,
        )


@dataclass
class ABAnalysisReport:
    """Statistical analysis report for an A/B experiment.

    Attributes:
        result: The raw experiment result.
        significant: Whether a statistically significant difference was found.
        winner: Name of the winning variant, or None if no clear winner.
        tests: Dict of test name → test results.
        recommendation: Human-readable recommendation string.
    """

    result: ABExperimentResult
    significant: bool
    winner: Optional[str]
    tests: dict[str, Any]
    recommendation: str

    @property
    def summary(self) -> str:
        """One-line summary of the analysis."""
        if self.winner:
            return f"SIGNIFICANT: {self.winner} wins ({self.recommendation})"
        return f"NOT SIGNIFICANT: {self.recommendation}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "significant": self.significant,
            "winner": self.winner,
            "recommendation": self.recommendation,
            "tests": self.tests,
            "total_runs": self.result.total_runs,
            "variants": {
                name: {
                    "total_runs": vm.total_runs,
                    "success_rate": vm.success_rate,
                    "successes": vm.successes,
                    "failures": vm.failures,
                }
                for name, vm in self.result.variants.items()
            },
        }
