"""
Baseline persistence for agent evaluation results.

Enables saving benchmark run results as versioned baselines
and loading them later for regression detection.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BaselineManifest:
    """A saved set of benchmark results used as a regression baseline."""

    name: str
    created_at: str
    profile: str
    benchmarks: dict[str, dict[str, Any]] = field(default_factory=dict)
    # benchmarks dict: benchmark_name -> {
    #     "summary": {composite_mean, groundedness_mean, verification_pass_rate, ...},
    #     "per_task": [{run_id, task_id, groundedness, composite, ...}, ...]
    # }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "profile": self.profile,
            "benchmarks": self.benchmarks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineManifest:
        return cls(
            name=data["name"],
            created_at=data.get("created_at", ""),
            profile=data.get("profile", "production"),
            benchmarks=data.get("benchmarks", {}),
        )


def save_baseline(
    benchmark_results: dict[str, list[dict[str, Any]]],
    name: str,
    profile: str = "production",
    output_path: str | Path | None = None,
) -> Path:
    """Save benchmark results as a named baseline.

    Args:
        benchmark_results: Dict mapping benchmark_name -> list of per-task metric dicts.
        name: Human-readable baseline name (e.g. "v0.6", "production-2026-06-28").
        profile: Agent profile used when the baseline was captured.
        output_path: Path to save the baseline JSON file. Defaults to
                     eval_results/baselines/<name>.json.

    Returns:
        Path to the saved baseline file.
    """
    if output_path is None:
        output_path = Path("eval_results/baselines")
    output_path = Path(output_path)

    # Compute summaries per benchmark
    benchmarks = {}
    for bench_name, tasks in benchmark_results.items():
        if not tasks:
            continue

        # Compute aggregate means
        metric_keys = [
            "groundedness", "citation_precision", "verification_pass_rate",
            "tool_success_rate", "answer_completeness", "composite",
        ]
        summary = {}
        for key in metric_keys:
            vals = [t.get(key, 0) for t in tasks]
            summary[f"{key}_mean"] = round(sum(vals) / len(vals), 4)

        # Latency
        latencies = [t.get("latency_ms", 0) for t in tasks]
        summary["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

        # Verification pass count
        verified = sum(1 for t in tasks if t.get("verification_passed", False))
        summary["verification_pass_count"] = verified
        summary["tasks_evaluated"] = len(tasks)
        summary["verification_pass_rate"] = round(verified / max(len(tasks), 1), 4)

        benchmarks[bench_name] = {
            "summary": summary,
            "per_task": tasks,
        }

    manifest = BaselineManifest(
        name=name,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        profile=profile,
        benchmarks=benchmarks,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{name}.json"
    file_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    return file_path


def load_baseline(path: str | Path, baselines_dir: str | Path | None = None) -> BaselineManifest:
    """Load a saved baseline from a JSON file.

    Args:
        path: Path to a baseline JSON file or a name in the baselines directory.
              If just a name (no path separators), looks in <baselines_dir>/<name>.json.
        baselines_dir: Directory containing baseline JSON files.
                       Defaults to eval_results/baselines/.

    Returns:
        Loaded BaselineManifest.
    """
    path = Path(path)

    # Allow short names: "v0.6" → "<baselines_dir>/v0.6.json"
    if "/" not in str(path) and "\\" not in str(path):
        if not path.exists() and path.suffix != ".json":
            if baselines_dir:
                path = Path(baselines_dir) / f"{path.name}.json"
            else:
                path = Path("eval_results/baselines") / f"{path.name}.json"

    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")

    data = json.loads(path.read_text())
    return BaselineManifest.from_dict(data)


def list_baselines(baselines_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List all available baselines.

    Args:
        baselines_dir: Directory containing baseline JSON files.
                       Defaults to eval_results/baselines/.

    Returns:
        List of dicts with name, created_at, profile, benchmark_count, file_path.
    """
    if baselines_dir is None:
        baselines_dir = Path("eval_results/baselines")
    baselines_dir = Path(baselines_dir)

    if not baselines_dir.exists():
        return []

    baselines = []
    for f in sorted(baselines_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            baselines.append({
                "name": data.get("name", f.stem),
                "created_at": data.get("created_at", ""),
                "profile": data.get("profile", "production"),
                "benchmark_count": len(data.get("benchmarks", {})),
                "file_path": str(f),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return baselines
