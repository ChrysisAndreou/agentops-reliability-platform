"""
Canary deployment framework for AI agents.

Provides a staged rollout system for deploying new agent versions with
gradual traffic shifting, automatic regression detection at each stage,
and configurable rollback conditions. Designed for production agent
deployments where a bad agent version can silently degrade quality.

Architecture:
    CanaryStage → CanaryConfig → CanaryDeployer → CanaryResult

Stages progress from low-traffic canary (e.g., 5%) through medium (25%)
to full rollout (100%), with evaluation gates at each transition.

Rollback triggers:
    - Regression in primary metric (statistically significant drop)
    - Error rate exceeding threshold
    - Latency spike beyond configured multiplier
    - Any custom condition function
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RollbackReason(str, Enum):
    """Reasons a canary deployment was rolled back."""

    REGRESSION_DETECTED = "regression_detected"
    ERROR_RATE_EXCEEDED = "error_rate_exceeded"
    LATENCY_SPIKE = "latency_spike"
    MANUAL = "manual"
    CUSTOM_CONDITION = "custom_condition"
    STAGE_TIMEOUT = "stage_timeout"


class CanaryStatus(str, Enum):
    """Status of a canary deployment."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------

@dataclass
class RollbackCondition:
    """A condition that triggers automatic rollback.

    Attributes:
        reason: The reason enum for reporting.
        description: Human-readable description.
        threshold: Threshold value for the condition.
        metric: Which metric to check (for metric-based conditions).
    """

    reason: RollbackReason
    description: str
    threshold: float = 0.0
    metric: str = ""


@dataclass
class CanaryStage:
    """A single stage in a canary deployment.

    Attributes:
        name: Stage name (e.g., 'canary-5%', 'staged-25%', 'full-100%').
        traffic_percent: Percentage of traffic routed to the new variant.
        min_samples: Minimum samples required before evaluating this stage.
        max_duration_minutes: Maximum time to wait for min_samples.
        evaluation_metric: Primary metric to evaluate at this stage.
        regression_threshold: Maximum allowed drop in evaluation_metric.
        error_rate_threshold: Maximum allowed error/failure rate.
        latency_multiplier_threshold: Maximum latency increase vs baseline (e.g., 2.0 = 2x).
        custom_gate: Optional callable that returns (passed: bool, reason: str).
    """

    name: str
    traffic_percent: float
    min_samples: int = 100
    max_duration_minutes: int = 60
    evaluation_metric: str = "verification_pass_rate"
    regression_threshold: float = 0.05
    error_rate_threshold: float = 0.10
    latency_multiplier_threshold: float = 2.0
    custom_gate: Optional[Callable[[dict[str, Any]], tuple[bool, str]]] = None

    def __post_init__(self):
        if not 0 < self.traffic_percent <= 100:
            raise ValueError(f"traffic_percent must be in (0, 100], got {self.traffic_percent}")
        if self.regression_threshold < 0:
            raise ValueError(f"regression_threshold must be >= 0, got {self.regression_threshold}")


@dataclass
class CanaryConfig:
    """Configuration for a canary deployment.

    Attributes:
        name: Deployment name for reporting.
        stages: Ordered list of canary stages (typically 5% → 25% → 100%).
        baseline_metric: Baseline value of the primary metric (from control).
        baseline_error_rate: Baseline error/failure rate.
        baseline_latency_p50: Baseline median latency in milliseconds.
        baseline_latency_p95: Baseline P95 latency in milliseconds.
        auto_rollback: Whether to automatically rollback on condition trigger.
        require_approval: Whether each stage requires manual approval to proceed.
    """

    name: str = ""
    stages: list[CanaryStage] = field(default_factory=list)
    baseline_metric: float = 0.0
    baseline_error_rate: float = 0.0
    baseline_latency_p50: float = 0.0
    baseline_latency_p95: float = 0.0
    auto_rollback: bool = True
    require_approval: bool = False

    def __post_init__(self):
        if not self.stages:
            raise ValueError("CanaryConfig must have at least one stage")

        # Validate stages are monotonically increasing in traffic
        prev = 0.0
        for stage in self.stages:
            if stage.traffic_percent <= prev:
                raise ValueError(
                    f"Stage traffic_percent must increase monotonically: "
                    f"{stage.traffic_percent} <= {prev}"
                )
            prev = stage.traffic_percent


# ---------------------------------------------------------------------------
# Stage-level metrics
# ---------------------------------------------------------------------------

@dataclass
class StageMetrics:
    """Metrics collected during a canary stage.

    Attributes:
        total_runs: Number of canary runs in this stage.
        successes: Number of successful runs.
        failures: Number of failed runs.
        metric_values: Accumulated values for the primary evaluation metric.
        latencies: All latency values collected.
        errors: List of error messages.
    """

    total_runs: int = 0
    successes: int = 0
    failures: int = 0
    metric_values: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 1.0
        return self.successes / self.total_runs

    @property
    def error_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.failures / self.total_runs

    @property
    def metric_mean(self) -> float:
        if not self.metric_values:
            return 0.0
        return sum(self.metric_values) / len(self.metric_values)

    @property
    def latency_p50(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = len(sorted_l) // 2
        return sorted_l[idx]

    @property
    def latency_p95(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def record(self, success: bool, metric_value: float, latency_ms: float = 0.0, error: str = ""):
        """Record a single canary run."""
        self.total_runs += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.metric_values.append(metric_value)
        self.latencies.append(latency_ms)
        if error:
            self.errors.append(error)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CanaryStageResult:
    """Result of a single canary stage evaluation.

    Attributes:
        stage: The stage configuration.
        metrics: Collected metrics for this stage.
        passed: Whether the stage evaluation gate passed.
        regression_detected: Whether a metric regression was detected.
        rollback_reason: If rolled back, the reason.
        details: Human-readable details about the evaluation.
        duration_seconds: How long this stage took.
    """

    stage: CanaryStage
    metrics: StageMetrics = field(default_factory=StageMetrics)
    passed: bool = False
    regression_detected: bool = False
    rollback_reason: Optional[RollbackReason] = None
    details: str = ""
    duration_seconds: float = 0.0


@dataclass
class CanaryResult:
    """Complete result of a canary deployment.

    Attributes:
        config: The canary configuration used.
        status: Final deployment status.
        stage_results: Per-stage evaluation results.
        completed_stages: Number of stages that passed evaluation.
        final_reason: Explanation of the final outcome.
    """

    config: CanaryConfig
    status: CanaryStatus = CanaryStatus.PENDING
    stage_results: list[CanaryStageResult] = field(default_factory=list)
    completed_stages: int = 0
    final_reason: str = ""

    @property
    def rolled_back(self) -> bool:
        return self.status == CanaryStatus.ROLLED_BACK

    @property
    def completed(self) -> bool:
        return self.status == CanaryStatus.COMPLETED

    @property
    def current_traffic_percent(self) -> float:
        if not self.stage_results:
            return 0.0
        return self.stage_results[-1].stage.traffic_percent


# ---------------------------------------------------------------------------
# CanaryDeployer
# ---------------------------------------------------------------------------

@dataclass
class CanaryDeployer:
    """Execute canary deployments for AI agent variants.

    Progresses through configurable stages, evaluating at each step for
    regressions in the primary metric, error rate spikes, and latency
    degradation. Supports automatic rollback and custom evaluation gates.

    Example:
        >>> stages = [
        ...     CanaryStage("canary-5%", traffic_percent=5.0, min_samples=50),
        ...     CanaryStage("staged-25%", traffic_percent=25.0, min_samples=200),
        ...     CanaryStage("full-100%", traffic_percent=100.0, min_samples=500),
        ... ]
        >>> config = CanaryConfig(
        ...     name="reranker-v2",
        ...     stages=stages,
        ...     baseline_metric=0.85,
        ...     baseline_error_rate=0.05,
        ... )
        >>> deployer = CanaryDeployer(config)
        >>> result = deployer.start()
        >>> for run in canary_runs:
        ...     result = deployer.record_run(result, run)
        ...     if deployer.stage_complete(result):
        ...         result = deployer.evaluate_stage(result)
        ...         if result.rolled_back:
        ...             break
    """

    config: CanaryConfig

    def start(self) -> CanaryResult:
        """Initialize a new canary deployment.

        Returns:
            Fresh CanaryResult with PENDING status.
        """
        result = CanaryResult(
            config=self.config,
            status=CanaryStatus.IN_PROGRESS,
            stage_results=[],
            completed_stages=0,
        )
        return result

    def _current_stage(self, result: CanaryResult) -> Optional[CanaryStage]:
        """Get the currently active stage."""
        idx = result.completed_stages
        if idx < len(self.config.stages):
            return self.config.stages[idx]
        return None

    def record_run(
        self,
        result: CanaryResult,
        success: bool,
        metric_value: float,
        latency_ms: float = 0.0,
        error: str = "",
    ) -> CanaryResult:
        """Record a single canary run against the current stage.

        Args:
            result: Current deployment state.
            success: Whether the run was successful.
            metric_value: Value of the primary evaluation metric.
            latency_ms: Observed latency in milliseconds.
            error: Error message if the run failed.

        Returns:
            Updated CanaryResult.
        """
        stage = self._current_stage(result)
        if stage is None:
            # All stages complete
            if result.status != CanaryStatus.COMPLETED:
                result.status = CanaryStatus.COMPLETED
                result.final_reason = "All canary stages completed."
            return result

        # Ensure stage result exists
        while len(result.stage_results) <= result.completed_stages:
            result.stage_results.append(
                CanaryStageResult(stage=stage, metrics=StageMetrics())
            )

        sr = result.stage_results[-1]
        sr.metrics.record(success, metric_value, latency_ms, error)
        return result

    def stage_complete(self, result: CanaryResult) -> bool:
        """Check if the current stage has collected enough samples.

        Returns True when min_samples is reached for the current stage.
        """
        stage = self._current_stage(result)
        if stage is None:
            return True

        idx = result.completed_stages
        if idx >= len(result.stage_results):
            return False

        sr = result.stage_results[idx]
        return sr.metrics.total_runs >= stage.min_samples

    def evaluate_stage(self, result: CanaryResult) -> CanaryResult:
        """Evaluate the current stage and decide whether to proceed or rollback.

        Checks:
        1. Primary metric regression (drop > baseline by regression_threshold)
        2. Error rate exceeds threshold
        3. Latency spike (P95 > baseline * latency_multiplier_threshold)
        4. Custom gate (if provided)

        Args:
            result: Current deployment state.

        Returns:
            Updated CanaryResult with stage evaluation results.
        """
        import datetime

        stage = self._current_stage(result)
        if stage is None:
            result.status = CanaryStatus.COMPLETED
            result.final_reason = "All canary stages evaluated and passed."
            return result

        idx = result.completed_stages
        if idx >= len(result.stage_results):
            if result.status != CanaryStatus.COMPLETED:
                result.status = CanaryStatus.IN_PROGRESS
            return result

        sr = result.stage_results[idx]

        # --- Check 1: Metric regression ---
        metric_drop = self.config.baseline_metric - sr.metrics.metric_mean
        regression = metric_drop > stage.regression_threshold

        # --- Check 2: Error rate ---
        error_rate_exceeded = sr.metrics.error_rate > stage.error_rate_threshold

        # --- Check 3: Latency spike ---
        latency_p95 = sr.metrics.latency_p95
        latency_baseline = self.config.baseline_latency_p95
        latency_spike = (
            latency_baseline > 0
            and latency_p95 > latency_baseline * stage.latency_multiplier_threshold
        )

        # --- Check 4: Custom gate ---
        custom_failed = False
        custom_reason = ""
        if stage.custom_gate is not None:
            custom_data = {
                "metric_drop": metric_drop,
                "error_rate": sr.metrics.error_rate,
                "latency_p95": latency_p95,
                "success_rate": sr.metrics.success_rate,
                "total_runs": sr.metrics.total_runs,
            }
            custom_passed, custom_reason = stage.custom_gate(custom_data)
            if not custom_passed:
                custom_failed = True

        # Determine outcome
        if error_rate_exceeded:
            sr.passed = False
            sr.rollback_reason = RollbackReason.ERROR_RATE_EXCEEDED
            sr.details = (
                f"Error rate {sr.metrics.error_rate:.1%} exceeds threshold "
                f"{stage.error_rate_threshold:.1%} ({sr.metrics.failures}/{sr.metrics.total_runs} failures)"
            )
            sr.regression_detected = True
        elif latency_spike:
            sr.passed = False
            sr.rollback_reason = RollbackReason.LATENCY_SPIKE
            sr.details = (
                f"P95 latency {latency_p95:.0f}ms exceeds "
                f"{stage.latency_multiplier_threshold:.1f}x baseline "
                f"({self.config.baseline_latency_p95:.0f}ms)"
            )
            sr.regression_detected = True
        elif regression:
            sr.passed = False
            sr.rollback_reason = RollbackReason.REGRESSION_DETECTED
            sr.details = (
                f"Metric '{stage.evaluation_metric}' dropped by {metric_drop:.1%} "
                f"(baseline: {self.config.baseline_metric:.3f}, "
                f"canary: {sr.metrics.metric_mean:.3f}, "
                f"threshold: {stage.regression_threshold:.1%})"
            )
            sr.regression_detected = True
        elif custom_failed:
            sr.passed = False
            sr.rollback_reason = RollbackReason.CUSTOM_CONDITION
            sr.details = custom_reason
            sr.regression_detected = True
        else:
            sr.passed = True
            sr.details = (
                f"Stage '{stage.name}' passed: "
                f"metric={sr.metrics.metric_mean:.3f} (baseline={self.config.baseline_metric:.3f}), "
                f"error_rate={sr.metrics.error_rate:.1%}, "
                f"P95 latency={latency_p95:.0f}ms, "
                f"runs={sr.metrics.total_runs}"
            )

        result.completed_stages += 1

        if not sr.passed and self.config.auto_rollback:
            result.status = CanaryStatus.ROLLED_BACK
            result.final_reason = (
                f"Rolled back at stage '{stage.name}': {sr.details}"
            )
        elif result.completed_stages >= len(self.config.stages):
            result.status = CanaryStatus.COMPLETED
            result.final_reason = "All canary stages passed. Deployment complete."
        else:
            result.status = CanaryStatus.IN_PROGRESS

        return result

    def generate_report(self, result: CanaryResult) -> dict[str, Any]:
        """Generate a structured report of the canary deployment.

        Args:
            result: The completed/failed canary result.

        Returns:
            Dict with summary, stage-by-stage breakdown, and recommendations.
        """
        stages_report = []
        for sr in result.stage_results:
            stages_report.append({
                "stage_name": sr.stage.name,
                "traffic_percent": sr.stage.traffic_percent,
                "passed": sr.passed,
                "total_runs": sr.metrics.total_runs,
                "success_rate": sr.metrics.success_rate,
                "error_rate": sr.metrics.error_rate,
                "metric_mean": sr.metrics.metric_mean,
                "latency_p50": sr.metrics.latency_p50,
                "latency_p95": sr.metrics.latency_p95,
                "regression_detected": sr.regression_detected,
                "rollback_reason": sr.rollback_reason.value if sr.rollback_reason else None,
                "details": sr.details,
            })

        return {
            "deployment_name": self.config.name,
            "status": result.status.value,
            "completed_stages": result.completed_stages,
            "total_stages": len(self.config.stages),
            "final_reason": result.final_reason,
            "baseline": {
                "metric": self.config.baseline_metric,
                "error_rate": self.config.baseline_error_rate,
                "latency_p50": self.config.baseline_latency_p50,
                "latency_p95": self.config.baseline_latency_p95,
            },
            "stages": stages_report,
        }
