"""
Evaluation harness — runs agents on reliability benchmarks and produces
structured reports with metrics, failure analysis, and recommendations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..tracing.classifier import FailureClassifier
from ..tracing.store import TraceStore
from .benchmarks import ALL_BENCHMARKS, ReliabilityBenchmark
from .metrics import ReliabilityMetrics, compute_metrics


@dataclass
class EvalReport:
    """Complete evaluation report for a benchmark run."""
    benchmark_name: str
    model: str
    total_tasks: int
    results: list[ReliabilityMetrics] = field(default_factory=list)
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    run_timestamp: str = ""

    def to_markdown(self) -> str:
        lines = []
        lines.append(f"# Evaluation Report: {self.benchmark_name}")
        lines.append(f"Model: {self.model} | Tasks: {self.total_tasks} | {self.run_timestamp}")
        lines.append("")

        s = self.summary
        lines.append("## Summary")
        lines.append(f"- **Composite Score**: {s.get('composite_mean', 0):.3f}")
        lines.append(f"- **Groundedness**: {s.get('groundedness_mean', 0):.3f}")
        lines.append(f"- **Verification Pass Rate**: {s.get('verification_pass_rate', 0):.1%}")
        lines.append(f"- **Avg Latency**: {s.get('avg_latency_ms', 0):.0f}ms")
        lines.append(f"- **Citation Precision**: {s.get('citation_precision_mean', 0):.3f}")
        lines.append("")

        if self.failure_patterns:
            lines.append("## Failure Analysis")
            for fp in self.failure_patterns:
                sev = {"critical": "!!!", "high": "!!", "medium": "!"}.get(fp.get("severity", ""), "")
                lines.append(f"- [{sev}] **{fp['name']}** ({fp['count']}): {fp['description']}")
            lines.append("")

        lines.append("## Per-Task Metrics")
        lines.append("| Task | Grounded | Citations | Verified | Latency | Composite |")
        lines.append("|------|----------|-----------|----------|---------|-----------|")
        for r in self.results:
            verified = "✓" if r.verification_passed else "✗"
            lines.append(
                f"| {r.task_id} | {r.groundedness:.2f} | {r.citations_used_count}/{r.retrieved_chunks_count} "
                f"| {verified} | {r.latency_ms:.0f}ms | {r.composite:.3f} |"
            )
        lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "benchmark_name": self.benchmark_name,
            "model": self.model,
            "total_tasks": self.total_tasks,
            "summary": self.summary,
            "failure_patterns": self.failure_patterns,
            "results": [r.to_dict() for r in self.results],
        }, indent=2)


class EvalHarness:
    """Runs reliability agents on benchmarks and produces reports.

    Usage:
        harness = EvalHarness(agent, trace_store, model="gpt-4o")
        report = await harness.run_benchmark(benchmark)
        print(report.to_markdown())
    """

    def __init__(
        self,
        agent,
        trace_store: TraceStore | None = None,
        model: str = "gpt-4o",
        output_dir: str | None = None,
    ):
        self.agent = agent
        self.trace_store = trace_store or TraceStore(":memory:")
        self.model = model
        self.output_dir = Path(output_dir) if output_dir else None
        self.classifier = FailureClassifier()

    async def run_benchmark(self, benchmark: ReliabilityBenchmark) -> EvalReport:
        """Run all tasks in a benchmark and produce a report."""
        results = []
        traces_for_analysis = []

        for task in benchmark.tasks:
            result = await self.agent.run(
                task=task.question,
                task_id=task.id,
                context=f"Category: {task.category}. Difficulty: {task.difficulty}.",
            )

            self.trace_store.save(result)
            metrics = compute_metrics(result, key_terms=task.key_terms)
            results.append(metrics)
            traces_for_analysis.append(result)

        return self._build_report(benchmark, results, traces_for_analysis)

    def _build_report(
        self,
        benchmark: ReliabilityBenchmark,
        results: list[ReliabilityMetrics],
        traces_for_analysis: list,
    ) -> EvalReport:
        composites = [r.composite for r in results]
        groundedness_vals = [r.groundedness for r in results]
        citation_vals = [r.citation_precision for r in results]
        latencies = [r.latency_ms for r in results]
        vp_rate = sum(1 for r in results if r.verification_passed) / max(len(results), 1)

        summary = {
            "composite_mean": round(sum(composites) / len(composites), 3) if composites else 0,
            "composite_std": round(pd.Series(composites).std(), 3) if len(composites) > 1 else 0,
            "groundedness_mean": round(sum(groundedness_vals) / len(groundedness_vals), 3) if groundedness_vals else 0,
            "citation_precision_mean": round(sum(citation_vals) / len(citation_vals), 3) if citation_vals else 0,
            "verification_pass_rate": round(vp_rate, 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "total_tool_calls": sum(r.tool_calls_count for r in results),
            "tasks_evaluated": len(results),
        }

        # Failure analysis
        failure_patterns = self.classifier.to_dict(traces_for_analysis)
        pattern_list = failure_patterns.get("failure_patterns", [])

        report = EvalReport(
            benchmark_name=benchmark.name,
            model=self.model,
            total_tasks=len(results),
            results=results,
            failure_patterns=pattern_list,
            summary=summary,
            run_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Save if output_dir set
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.output_dir / f"{benchmark.name}_report.md"
            report_path.write_text(report.to_markdown())
            json_path = self.output_dir / f"{benchmark.name}_report.json"
            json_path.write_text(report.to_json())

        return report

    async def run_all(self, benchmarks: list[ReliabilityBenchmark] | None = None) -> list[EvalReport]:
        """Run all benchmarks."""
        if benchmarks is None:
            benchmarks = ALL_BENCHMARKS

        reports = []
        for bench in benchmarks:
            report = await self.run_benchmark(bench)
            reports.append(report)

        return reports

    async def run_with_simulator(
        self,
        benchmark: ReliabilityBenchmark,
        sim_config=None,
    ) -> EvalReport:
        """Run benchmark with a simulated agent (no API keys needed)."""
        from .simulator import PRODUCTION_AGENT, SimulatedAgent

        if sim_config is None:
            sim_config = PRODUCTION_AGENT

        sim_agent = SimulatedAgent(config=sim_config, seed=42)
        results = []
        traces_for_analysis = []

        for task in benchmark.tasks:
            result = await sim_agent.run(task.question, task_id=task.id)
            self.trace_store.save(result)
            metrics = compute_metrics(result, key_terms=task.key_terms)
            results.append(metrics)
            traces_for_analysis.append(result)

        return self._build_report(benchmark, results, traces_for_analysis)

    def dry_run(self, benchmark: ReliabilityBenchmark) -> list[dict[str, Any]]:
        """Run benchmark without LLM calls — validates structure only."""
        results = []
        for task in benchmark.tasks:
            metrics = compute_metrics(
                type("FakeResult", (), {
                    "task_id": task.id,
                    "grounded_claims": task.key_terms[:2],
                    "ungrounded_claims": task.key_terms[2:],
                    "citations_used": ["chunk:0:abc"],
                    "retrieved_chunks_count": 5,
                    "total_latency_ms": 5000,
                    "verification_passed": True,
                    "tool_calls_count": 1 if task.requires_tool else 0,
                    "final_answer": " ".join(task.key_terms),
                })(),
                key_terms=task.key_terms,
            )
            results.append(metrics.to_dict())

        return results
