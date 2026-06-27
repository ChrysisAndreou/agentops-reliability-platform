"""
Agent regression testing — detect quality regressions across benchmark runs.

Runs all reliability benchmarks against a configurable simulated agent profile,
compares results to a saved baseline, and generates a regression report.
Exits with code 1 when regressions are detected (CI-friendly).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tracing.store import TraceStore
from .baselines import load_baseline
from .benchmarks import ALL_BENCHMARKS, ReliabilityBenchmark
from .comparator import EvalComparator
from .harness import EvalHarness
from .simulator import SimulatedAgent, get_profile

# Per-metric regression thresholds (a drop of this magnitude triggers a regression)
DEFAULT_THRESHOLDS = {
    "composite_mean": 0.05,
    "groundedness_mean": 0.05,
    "citation_precision_mean": 0.10,
    "verification_pass_rate": 0.10,
    "tool_success_rate_mean": 0.05,
    "answer_completeness_mean": 0.10,
    "avg_latency_ms": 0.30,  # 30% increase in latency is a regression
}


@dataclass
class BenchmarkRegression:
    """Per-benchmark regression result."""

    benchmark_name: str
    has_regressions: bool = False
    regressions: list[dict[str, Any]] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)
    stable: list[str] = field(default_factory=list)
    baseline_summary: dict[str, Any] = field(default_factory=dict)
    current_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionResult:
    """Complete regression test result across all benchmarks."""

    baseline_name: str
    profile: str
    benchmarks: list[BenchmarkRegression] = field(default_factory=list)
    has_regressions: bool = False
    run_timestamp: str = ""

    @property
    def exit_code(self) -> int:
        """CI-friendly exit code: 0 if no regressions, 1 if regressions found."""
        return 1 if self.has_regressions else 0

    def to_markdown(self) -> str:
        lines = [
            "# Agent Regression Test Report",
            "",
            f"**Baseline:** {self.baseline_name} | **Profile:** {self.profile} | **{self.run_timestamp}**",
            "",
        ]

        if self.has_regressions:
            lines.append("## ⚠️  REGRESSIONS DETECTED")
            lines.append("")
        else:
            lines.append("## ✅  No Regressions Detected")
            lines.append("")
            lines.append("All benchmarks are within acceptable thresholds of the baseline.")
            lines.append("")

        for br in self.benchmarks:
            status = "⚠️  REGRESSION" if br.has_regressions else "✅  PASS"
            lines.append(f"### {br.benchmark_name} — {status}")
            lines.append("")

            lines.append("| Metric | Baseline | Current | Delta | Status |")
            lines.append("|--------|----------|---------|-------|--------|")

            metric_labels = {
                "composite_mean": "Composite",
                "groundedness_mean": "Groundedness",
                "citation_precision_mean": "Citation Precision",
                "verification_pass_rate": "Verify Rate",
                "tool_success_rate_mean": "Tool Success",
                "answer_completeness_mean": "Completeness",
                "avg_latency_ms": "Avg Latency (ms)",
            }

            for key, label in metric_labels.items():
                base_val = br.baseline_summary.get(key)
                curr_val = br.current_summary.get(key)
                if base_val is None or curr_val is None:
                    continue

                delta = curr_val - base_val
                if key == "avg_latency_ms":
                    # Latency: increase is bad, decrease is good (flip sign for consistency)
                    delta_pct = (curr_val - base_val) / max(base_val, 1) * 100
                    if delta_pct > DEFAULT_THRESHOLDS.get(key, 30):
                        status = "⚠️"
                    elif delta_pct < -5:
                        status = "↑"
                    else:
                        status = "—"
                    sign = "+" if delta >= 0 else ""
                    lines.append(
                        f"| {label} | {base_val:.1f} | {curr_val:.1f} | {sign}{delta_pct:.1f}% | {status} |"
                    )
                else:
                    if key in [r["metric"] for r in br.regressions]:
                        status = "⚠️"
                    elif key in [r["metric"] for r in br.improvements]:
                        status = "↑"
                    else:
                        status = "—"
                    sign = "+" if delta >= 0 else ""
                    lines.append(
                        f"| {label} | {base_val:.3f} | {curr_val:.3f} | {sign}{delta:.4f} | {status} |"
                    )

            lines.append("")

            # Regression details
            if br.regressions:
                lines.append("**Regressions:**")
                for r in br.regressions:
                    lines.append(
                        f"- **{r['metric']}**: {r['baseline']:.4f} → {r['current']:.4f} "
                        f"(Δ {r['delta']:+.4f}, threshold={r['threshold']})"
                    )
                lines.append("")

            if br.improvements:
                lines.append("**Improvements:**")
                for imp in br.improvements:
                    lines.append(
                        f"- **{imp['metric']}**: {imp['baseline']:.4f} → {imp['current']:.4f} "
                        f"(Δ {imp['delta']:+.4f})"
                    )
                lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        import json
        return json.dumps({
            "baseline_name": self.baseline_name,
            "profile": self.profile,
            "has_regressions": self.has_regressions,
            "exit_code": self.exit_code,
            "run_timestamp": self.run_timestamp,
            "benchmarks": [
                {
                    "name": br.benchmark_name,
                    "has_regressions": br.has_regressions,
                    "regressions": br.regressions,
                    "improvements": br.improvements,
                    "stable": br.stable,
                }
                for br in self.benchmarks
            ],
        }, indent=2)


class RegressionRunner:
    """Run regression tests by comparing current benchmark results to a baseline.

    Usage:
        runner = RegressionRunner(profile="production")
        result = await runner.run("baselines/v0.6.json", output_dir="eval_results/")
        print(result.to_markdown())
        sys.exit(result.exit_code)
    """

    def __init__(
        self,
        profile: str = "production",
        baselines_dir: str | Path | None = None,
        seed: int = 42,
    ):
        sim_config = get_profile(profile)
        if sim_config is None:
            raise ValueError(
                f"Unknown profile '{profile}'. Available: perfect, production, development, unreliable"
            )
        self.profile = profile
        self.sim_config = sim_config
        self.seed = seed
        self.baselines_dir = Path(baselines_dir) if baselines_dir else Path("eval_results/baselines")
        self.comparator = EvalComparator(seed=seed)

    async def run(
        self,
        baseline_name: str,
        output_dir: str | Path | None = None,
        benchmarks: list[ReliabilityBenchmark] | None = None,
    ) -> RegressionResult:
        """Run regression tests against a baseline.

        Args:
            baseline_name: Name or path of the baseline to compare against.
            output_dir: Directory for output reports (default: eval_results/).
            benchmarks: Specific benchmarks to test (default: ALL_BENCHMARKS).

        Returns:
            RegressionResult with per-benchmark regression detection.
        """
        baseline = load_baseline(baseline_name, baselines_dir=self.baselines_dir)

        if benchmarks is None:
            benchmarks = ALL_BENCHMARKS

        out = Path(output_dir) if output_dir else Path("eval_results")
        out.mkdir(parents=True, exist_ok=True)

        # Run all benchmarks with the simulated agent
        agent = SimulatedAgent(config=self.sim_config, seed=self.seed)
        trace_store = TraceStore(str(out / "regression_traces.db"))
        harness = EvalHarness(
            agent=agent,
            trace_store=trace_store,
            model=f"sim-{self.profile}",
            output_dir=str(out),
        )

        bench_regressions: list[BenchmarkRegression] = []
        has_any_regressions = False

        for bench in benchmarks:
            bench_name = bench.name

            # Skip benchmarks without baseline data
            if bench_name not in baseline.benchmarks:
                continue

            # Run the benchmark
            report = await harness.run_with_simulator(bench, sim_config=self.sim_config)
            [r.to_dict() for r in report.results]

            # Build current summary
            current_summary = report.summary

            # Compare to baseline
            baseline_data = baseline.benchmarks[bench_name]
            baseline_summary = baseline_data.get("summary", {})

            br = BenchmarkRegression(
                benchmark_name=bench_name,
                baseline_summary=baseline_summary,
                current_summary=current_summary,
            )

            # Detect regressions per metric

            for summary_key, threshold in DEFAULT_THRESHOLDS.items():
                base_val = baseline_summary.get(summary_key)
                curr_val = current_summary.get(summary_key)

                if base_val is None or curr_val is None:
                    continue

                if summary_key == "avg_latency_ms":
                    # Latency: > threshold % increase = regression
                    delta_pct = (curr_val - base_val) / max(base_val, 1)
                    if delta_pct > threshold:
                        br.regressions.append({
                            "metric": summary_key,
                            "baseline": round(base_val, 1),
                            "current": round(curr_val, 1),
                            "delta": round(delta_pct, 4),
                            "threshold": threshold,
                        })
                        br.has_regressions = True
                    elif delta_pct < -0.05:
                        br.improvements.append({
                            "metric": summary_key,
                            "baseline": round(base_val, 1),
                            "current": round(curr_val, 1),
                            "delta": round(delta_pct, 4),
                            "threshold": threshold,
                        })
                    else:
                        br.stable.append(summary_key)
                else:
                    # Score metrics: drop > threshold = regression
                    delta = curr_val - base_val
                    if delta < -threshold:
                        br.regressions.append({
                            "metric": summary_key,
                            "baseline": round(base_val, 4),
                            "current": round(curr_val, 4),
                            "delta": round(delta, 4),
                            "threshold": threshold,
                        })
                        br.has_regressions = True
                    elif delta > threshold:
                        br.improvements.append({
                            "metric": summary_key,
                            "baseline": round(base_val, 4),
                            "current": round(curr_val, 4),
                            "delta": round(delta, 4),
                            "threshold": threshold,
                        })
                    else:
                        br.stable.append(summary_key)

            bench_regressions.append(br)
            if br.has_regressions:
                has_any_regressions = True

        trace_store.close()

        result = RegressionResult(
            baseline_name=baseline.name,
            profile=self.profile,
            benchmarks=bench_regressions,
            has_regressions=has_any_regressions,
            run_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Save reports
        report_md = result.to_markdown()
        report_json = result.to_json()

        (out / "regression_report.md").write_text(report_md)
        (out / "regression_report.json").write_text(report_json)

        return result
