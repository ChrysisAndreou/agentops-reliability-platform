"""
Agent A/B Testing & Canary Deployment — statistical experiment framework for AI agents.

Provides a production-grade framework for comparing agent versions through
controlled A/B experiments with rigorous statistical analysis, and for
gradually rolling out new agent versions through canary deployments with
automatic regression detection and rollback.

Modules:
    experiment: ABExperiment with traffic splitting, variant configuration,
                and metric collection across agent versions.
    stats: Statistical tests for A/B evaluation — chi-squared, Fisher's exact,
           Bayesian A/B, Welch's t-test, Mann-Whitney U, sample size estimation.
    canary: CanaryDeployer with staged traffic shifting, automatic regression
            detection, and configurable rollback conditions.
    eval: ABEvalResult with structured reporting, significance summaries,
          and per-variant metric breakdowns.

Example usage:
    >>> from agentops.ab_testing import ABExperiment, chi_squared_test, CanaryDeployer
    >>> config = ABExperimentConfig(metric="verification_pass_rate")
    >>> experiment = ABExperiment(config)
    >>> results = experiment.run_with_traces(traces_a, traces_b)
    >>> report = experiment.analyze(results)
    >>> print(f"Significant: {report.significant}, Winner: {report.winner}")

Statistical tests can be used standalone:
    >>> p_value, significant = chi_squared_test([45, 5], [38, 12])
    >>> prob = bayesian_ab_test(successes_a=45, trials_a=50, successes_b=38, trials_b=50)
"""

from agentops.ab_testing.experiment import (
    ABExperiment,
    ABExperimentConfig,
    ABExperimentResult,
    TrafficSplit,
    TrafficAllocation,
    VariantConfig,
)
from agentops.ab_testing.stats import (
    bayesian_ab_test,
    chi_squared_test,
    compute_sample_size,
    confidence_interval,
    fisher_exact_test,
    mann_whitney_u,
    welch_t_test,
)
from agentops.ab_testing.canary import (
    CanaryConfig,
    CanaryDeployer,
    CanaryResult,
    CanaryStage,
    CanaryStageResult,
    CanaryStatus,
    RollbackCondition,
    RollbackReason,
)
from agentops.ab_testing.eval import (
    ABEvalReport,
    MetricComparison,
    VariantSummary,
    generate_ab_report,
    format_report_markdown,
)

__all__ = [
    # Experiment
    "ABExperiment",
    "ABExperimentConfig",
    "ABExperimentResult",
    "TrafficSplit",
    "TrafficAllocation",
    "VariantConfig",
    # Statistical tests
    "chi_squared_test",
    "fisher_exact_test",
    "bayesian_ab_test",
    "welch_t_test",
    "mann_whitney_u",
    "compute_sample_size",
    "confidence_interval",
    # Canary
    "CanaryConfig",
    "CanaryDeployer",
    "CanaryResult",
    "CanaryStage",
    "CanaryStageResult",
    "RollbackCondition",
    "RollbackReason",
    # Evaluation
    "ABEvalReport",
    "MetricComparison",
    "VariantSummary",
    "generate_ab_report",
]
