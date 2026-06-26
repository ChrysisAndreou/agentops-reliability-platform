"""
CLI entry point for the AgentOps Reliability Platform.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="agentops",
    help="AgentOps Reliability Platform — build, trace, and evaluate tool-using AI agents",
    no_args_is_help=True,
)


def _get_project_root() -> Path:
    """Find the project root from cwd or environment."""
    return Path.cwd()


async def _build_agent_from_dir(project_dir: Path, model: str = "gpt-4o", otel_enabled: bool = False):
    """Build a reliability agent from a project directory with sample data."""
    from agentops.agent.tool_registry import ToolRegistry, ToolDefinition
    from agentops.agent.implementations import ReliabilityAgent
    from agentops.retrieval.ingest import DocumentIngestor
    from agentops.retrieval.engine import RetrievalEngine
    from agentops.tracing.opentelemetry import OTelObserver

    docs_dir = project_dir / "sample_data" / "docs"
    if not docs_dir.exists():
        docs_dir = Path(__file__).parent.parent.parent.parent / "sample_data" / "docs"

    # Set up retrieval
    ingestor = DocumentIngestor(chunk_size=512, chunk_overlap=64)
    chunks = ingestor.ingest_directory(str(docs_dir))
    engine = RetrievalEngine()
    engine.index(chunks)

    def retrieval_fn(query: str, k: int = 5):
        return [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "source": r.source,
                "source_title": r.source_title,
                "score": r.score,
                "retrieval_method": r.retrieval_method,
            }
            for r in engine.search(query, k=k)
        ]

    # Set up tool registry
    registry = ToolRegistry()

    # Calculator tool
    def calculator(expression: str) -> str:
        import math, operator, ast
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "log": math.log, "sin": math.sin,
            "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
            "pow": pow, "int": int, "float": float, "ceil": math.ceil,
            "floor": math.floor, "exp": math.exp,
        }
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            def _eval(n):
                if isinstance(n, ast.Expression): return _eval(n.body)
                if isinstance(n, ast.Constant): return n.value
                if isinstance(n, ast.Name): return allowed[n.id]
                if isinstance(n, ast.BinOp): return ops[type(n.op)](_eval(n.left), _eval(n.right))
                if isinstance(n, ast.UnaryOp): return -_eval(n.operand)
                if isinstance(n, ast.Call):
                    return allowed[n.func.id](*[_eval(a) for a in n.args])
                raise ValueError(f"Unsupported: {type(n)}")
            return str(_eval(tree))
        except Exception as e:
            return f"Error: {e}"

    registry.register_many([
        ToolDefinition(
            name="calculator",
            description="Evaluate a mathematical expression",
            parameters={"expression": {"type": "string", "description": "Math expression to evaluate"}},
            required=["expression"],
            fn=calculator,
        ),
    ])

    print(f"Loaded {engine.chunk_count} document chunks from {docs_dir}")
    print(f"Tools: {registry.tool_names}")

    # OTEL observer
    otel = None
    if otel_enabled:
        otel = OTelObserver()
        otel.start()
        print(f"OpenTelemetry: enabled (endpoint={otel._otlp_endpoint})")

    return ReliabilityAgent(
        tool_registry=registry,
        retrieval_fn=retrieval_fn,
        model=model,
        otel_observer=otel,
    )


@app.command()
def run(
    task: str = typer.Argument(..., help="Task/question for the agent"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model to use"),
    project_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    otel: bool = typer.Option(False, "--otel", help="Enable OpenTelemetry trace/metric export"),
):
    """Run the reliability agent on a single task."""
    d = Path(project_dir) if project_dir else _get_project_root()

    async def _run():
        agent = await _build_agent_from_dir(d, model, otel_enabled=otel)
        result = await agent.run(task=task, task_id="cli-run")

        if json_output:
            print(json.dumps({
                "task_id": result.task_id,
                "task": result.task,
                "success": result.success,
                "verification_passed": result.verification_passed,
                "final_answer": result.final_answer,
                "grounded_claims": result.grounded_claims,
                "ungrounded_claims": result.ungrounded_claims,
                "citations_used": result.citations_used,
                "latency_ms": result.total_latency_ms,
            }, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"Task: {result.task}")
            print(f"Verification: {'PASSED' if result.verification_passed else 'FAILED'}")
            print(f"Grounded: {len(result.grounded_claims)}, Ungrounded: {len(result.ungrounded_claims)}")
            print(f"Citations: {len(result.citations_used)}, Latency: {result.total_latency_ms:.0f}ms")
            print(f"{'='*60}")
            print(f"\n{result.final_answer}\n")

    asyncio.run(_run())


@app.command()
def eval(
    benchmark: str = typer.Option("all", "--benchmark", "-b", help="Benchmark name or 'all'"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model"),
    project_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for reports"),
    otel: bool = typer.Option(False, "--otel", help="Enable OpenTelemetry trace/metric export"),
):
    """Run evaluation benchmarks."""
    from agentops.evals.benchmarks import ALL_BENCHMARKS, get_benchmark
    from agentops.evals.harness import EvalHarness
    from agentops.tracing.store import TraceStore

    d = Path(project_dir) if project_dir else _get_project_root()
    out = Path(output_dir) if output_dir else d / "eval_results"

    async def _run():
        agent = await _build_agent_from_dir(d, model, otel_enabled=otel)
        trace_store = TraceStore(str(out / "traces.db"))
        harness = EvalHarness(agent=agent, trace_store=trace_store, model=model, output_dir=str(out))

        if benchmark == "all":
            reports = await harness.run_all()
        else:
            bench = get_benchmark(benchmark)
            if not bench:
                print(f"Benchmark '{benchmark}' not found. Available: {[b.name for b in ALL_BENCHMARKS]}")
                raise typer.Exit(1)
            reports = [await harness.run_benchmark(bench)]

        for report in reports:
            print(report.to_markdown())
            print()

    asyncio.run(_run())


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM model"),
    project_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory"),
    otel: bool = typer.Option(False, "--otel", help="Enable OpenTelemetry trace/metric export"),
):
    """Start the FastAPI server."""
    import uvicorn

    d = Path(project_dir) if project_dir else _get_project_root()

    async def _start():
        agent = await _build_agent_from_dir(d, model, otel_enabled=otel)
        from agentops.tracing.store import TraceStore
        trace_store = TraceStore(str(d / "traces.db"))

        from agentops.api.app import create_app
        web_app = create_app(agent=agent, trace_store=trace_store)

        config = uvicorn.Config(web_app, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(_start())


@app.command()
def traces(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of traces to show"),
    failed_only: bool = typer.Option(False, "--failed", "-f", help="Show only failed traces"),
    project_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory"),
):
    """List recent traces."""
    from agentops.tracing.store import TraceStore

    d = Path(project_dir) if project_dir else _get_project_root()
    db_path = d / "traces.db"
    if not db_path.exists():
        print("No traces found. Run some evaluations first.")
        raise typer.Exit(1)

    store = TraceStore(str(db_path))
    traces = store.query(
        verification_passed=None if failed_only else None,
        success=False if failed_only else None,
        limit=limit,
    )

    if not traces:
        print("No traces found.")
    else:
        print(f"{'Run ID':<12} {'Task':<50} {'Verified':<8} {'Latency':<10}")
        print("-" * 80)
        for t in traces:
            v = "✓" if t.verification_passed else "✗"
            print(f"{t.run_id:<12} {t.task[:48]:<50} {v:<8} {t.total_latency_ms:.0f}ms")

    store.close()


@app.command()
def trace(run_id: str = typer.Argument(..., help="Run ID to inspect"), project_dir: Optional[str] = typer.Option(None, "--dir", "-d")):
    """Inspect a specific trace."""
    from agentops.tracing.store import TraceStore

    d = Path(project_dir) if project_dir else _get_project_root()
    store = TraceStore(str(d / "traces.db"))

    replay = store.get_replay(run_id)
    if replay is None:
        print(f"Trace '{run_id}' not found.")
        raise typer.Exit(1)

    print(f"Run ID: {replay['run_id']}")
    print(f"Task: {replay['task']}")
    print(f"Verification: {'PASSED' if replay['verification_passed'] else 'FAILED'}")
    print(f"Tool calls: {replay['tool_calls_count']}")
    print(f"\nPlan:")
    for i, step in enumerate(replay.get('plan', []), 1):
        print(f"  {i}. {step}")
    print(f"\nTrace steps:")
    for step in replay.get('reliability_trace', []):
        print(f"  [{step.get('step_type', '?')}] {step.get('step_name', '?')}: {step.get('output_summary', '')}")
    print(f"\nFinal Answer:\n{replay.get('final_answer', '')}")

    store.close()


@app.command()
def stats(project_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory")):
    """Show aggregate statistics."""
    from agentops.tracing.store import TraceStore

    d = Path(project_dir) if project_dir else _get_project_root()
    store = TraceStore(str(d / "traces.db"))
    s = store.stats()

    print(f"Total runs: {s['total_runs']}")
    if s['total_runs'] > 0:
        print(f"Verification pass rate: {s['verification_pass_rate']:.1%}")
        print(f"Failure rate: {s['failure_rate']:.1%}")
        print(f"Average latency: {s['avg_latency_ms']:.0f}ms")

    store.close()


@app.command()
def index(
    docs_dir: str = typer.Argument(..., help="Directory containing documents"),
    project_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory"),
):
    """Index documents for retrieval (dry-run — checks chunking)."""
    from agentops.retrieval.ingest import DocumentIngestor
    from agentops.retrieval.engine import RetrievalEngine

    ingestor = DocumentIngestor(chunk_size=512, chunk_overlap=64)
    chunks = ingestor.ingest_directory(docs_dir)
    engine = RetrievalEngine()
    engine.index(chunks)

    print(f"Ingested {len(chunks)} chunks from {docs_dir}")
    print(f"Engine ready: {engine.ready}")
    print(f"\nSample chunks:")
    for chunk in chunks[:3]:
        print(f"  [{chunk.chunk_id}] {chunk.source_title}: {chunk.content[:80]}...")

    # Test search
    results = engine.search("deployment strategy", k=3)
    if results:
        print(f"\nTest search 'deployment strategy':")
        for r in results:
            print(f"  [{r.chunk_id}] score={r.score:.3f} ({r.retrieval_method})")


@app.command()
def simulate(
    benchmark: str = typer.Option("all", "--benchmark", "-b", help="Benchmark name or 'all'"),
    profile: str = typer.Option("production", "--profile", "-p", help="Agent profile: perfect, production, development, unreliable"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for reports"),
    project_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory"),
):
    """Run evaluation benchmarks with a simulated agent (no API keys needed)."""
    from agentops.evals.benchmarks import ALL_BENCHMARKS, get_benchmark, list_benchmarks
    from agentops.evals.harness import EvalHarness
    from agentops.evals.simulator import get_profile
    from agentops.tracing.store import TraceStore

    sim_config = get_profile(profile)
    if sim_config is None:
        print(f"Profile '{profile}' not found. Available: perfect, production, development, unreliable")
        raise typer.Exit(1)

    d = Path(project_dir) if project_dir else _get_project_root()
    out = Path(output_dir) if output_dir else d / "eval_results"
    out.mkdir(parents=True, exist_ok=True)

    async def _run():
        trace_store = TraceStore(str(out / "sim_traces.db"))
        harness = EvalHarness(agent=None, trace_store=trace_store, model=f"sim-{profile}", output_dir=str(out))

        benchmarks_to_run = ALL_BENCHMARKS if benchmark == "all" else [get_benchmark(benchmark)]
        benchmarks_to_run = [b for b in benchmarks_to_run if b is not None]

        if not benchmarks_to_run:
            print(f"No benchmarks found. Available: {[b['name'] for b in list_benchmarks()]}")
            raise typer.Exit(1)

        print(f"Profile: {sim_config.name} — {sim_config.description}")
        print(f"Benchmarks: {len(benchmarks_to_run)}")
        print()

        all_summaries = []
        for bench in benchmarks_to_run:
            report = await harness.run_with_simulator(bench, sim_config=sim_config)
            all_summaries.append(report.summary)
            print(f"  {bench.name}: composite={report.summary.get('composite_mean', 0):.3f}, "
                  f"verify_rate={report.summary.get('verification_pass_rate', 0):.1%}")

        print(f"\nReports written to: {out}")

        # Write aggregate summary
        if len(all_summaries) > 1:
            lines = ["# Aggregate Simulation Summary", f"Profile: {sim_config.name}", ""]
            lines.append("| Benchmark | Tasks | Composite | Verify Rate | Grounded | Latency |")
            lines.append("|-----------|-------|-----------|-------------|----------|---------|")
            for bench, summary in zip(benchmarks_to_run, all_summaries):
                lines.append(
                    f"| {bench.name} | {len(bench.tasks)} | {summary.get('composite_mean', 0):.3f} | "
                    f"{summary.get('verification_pass_rate', 0):.1%} | "
                    f"{summary.get('groundedness_mean', 0):.3f} | "
                    f"{summary.get('avg_latency_ms', 0):.0f}ms |"
                )
            (out / "aggregate_summary.md").write_text("\n".join(lines))

    asyncio.run(_run())


@app.command()
def compare(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="Benchmark name to compare on"),
    profile_a: str = typer.Option("production", "--profile-a", "-a", help="First agent profile"),
    profile_b: str = typer.Option("development", "--profile-b", "-c", help="Second agent profile"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    project_dir: Optional[str] = typer.Option(None, "--dir", "-d", help="Project root directory"),
):
    """A/B compare two agent configurations on a benchmark."""
    from agentops.evals.benchmarks import get_benchmark
    from agentops.evals.simulator import get_profile
    from agentops.evals.comparator import EvalComparator

    config_a = get_profile(profile_a)
    config_b = get_profile(profile_b)
    if config_a is None or config_b is None:
        print(f"Profile not found. Available: perfect, production, development, unreliable")
        raise typer.Exit(1)

    bench = get_benchmark(benchmark)
    if bench is None:
        print(f"Benchmark '{benchmark}' not found.")
        raise typer.Exit(1)

    d = Path(project_dir) if project_dir else _get_project_root()
    out = Path(output_dir) if output_dir else d / "eval_results"

    async def _run():
        comparator = EvalComparator()
        result = await comparator.compare(bench, config_a=config_a, config_b=config_b)
        md = result.to_markdown()
        print(md)

        out.mkdir(parents=True, exist_ok=True)
        report_path = out / f"compare_{bench.name}_{config_a.name}_vs_{config_b.name}.md"
        report_path.write_text(md)
        print(f"\nReport: {report_path}")

    asyncio.run(_run())


@app.command()
def benchmarks():
    """List available evaluation benchmarks."""
    from agentops.evals.benchmarks import list_benchmarks

    blist = list_benchmarks()
    print(f"{'Name':<30} {'Tasks':<6} {'Categories':<40} {'Difficulties'}")
    print("-" * 90)
    for b in blist:
        cats = ", ".join(b["categories"])
        diffs = ", ".join(b["difficulties"])
        print(f"{b['name']:<30} {b['task_count']:<6} {cats:<40} {diffs}")


if __name__ == "__main__":
    app()
