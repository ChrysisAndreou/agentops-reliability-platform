"""
W&B Sweeps integration for hyperparameter optimization of AgentOps evaluation pipelines.

Provides sweep configuration, agent execution, and local-grid fallback for
environments without W&B access.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_WANDB_AVAILABLE = False
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]


@dataclass
class SweepParameter:
    """A single hyperparameter in a sweep configuration."""

    name: str
    values: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    distribution: str = "categorical"  # "categorical", "uniform", "log_uniform", "discrete"
    default: Any = None

    def to_wandb_dict(self) -> dict[str, Any]:
        """Convert to W&B sweep parameter format."""
        param: dict[str, Any] = {}
        if self.values is not None:
            param["values"] = self.values
        if self.min is not None:
            param["min"] = self.min
        if self.max is not None:
            param["max"] = self.max
        if self.distribution:
            param["distribution"] = self.distribution
        return param


@dataclass
class SweepConfig:
    """Configuration for a W&B hyperparameter sweep."""

    name: str
    method: str = "grid"  # "grid", "random", "bayes"
    metric: str = "benchmark/support-tickets/composite_mean"
    goal: str = "maximize"  # "maximize" or "minimize"
    parameters: list[SweepParameter] = field(default_factory=list)
    description: str = ""
    project: str = "agentops-reliability-platform"
    entity: str | None = None

    def to_wandb_dict(self) -> dict[str, Any]:
        """Convert to W&B sweep configuration dictionary."""
        parameters_dict = {}
        for p in self.parameters:
            parameters_dict[p.name] = p.to_wandb_dict()
            if p.default is not None:
                parameters_dict[p.name]["default"] = p.default

        return {
            "name": self.name,
            "method": self.method,
            "metric": {"name": self.metric, "goal": self.goal},
            "parameters": parameters_dict,
            "description": self.description,
        }

    def expand_grid(self) -> list[dict[str, Any]]:
        """Expand grid-sweep parameters into all combinations (local fallback)."""
        if self.method != "grid":
            raise ValueError(f"expand_grid only works with method='grid', got '{self.method}'")

        param_names = []
        param_values = []
        for p in self.parameters:
            param_names.append(p.name)
            if p.values is not None:
                param_values.append(p.values)
            elif p.min is not None and p.max is not None:
                # For numeric ranges in grid mode, sample evenly
                steps = 5  # Default discretization
                span = p.max - p.min
                param_values.append([p.min + i * span / (steps - 1) for i in range(steps)])
            else:
                param_values.append([p.default])

        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        return combinations


class WandBSweep:
    """Manage W&B hyperparameter sweeps for AgentOps evaluation.

    Supports grid, random, and Bayesian optimization sweeps. Falls back to
    local grid execution when W&B is unavailable.

    Usage:
        sweep_config = SweepConfig(
            name="agent-profile-sweep",
            method="grid",
            metric="benchmark/support-tickets/composite_mean",
            parameters=[
                SweepParameter("groundedness_target", values=[0.75, 0.85, 0.95]),
                SweepParameter("tool_success_rate", values=[0.80, 0.90, 0.95]),
            ],
        )

        sweeper = WandBSweep(sweep_config)
        sweeper.run_local_grid(
            train_fn=lambda config: {"composite_mean": evaluate(config)},
            output_dir="./sweep_results",
        )
    """

    def __init__(self, config: SweepConfig):
        """Initialize the sweep manager.

        Args:
            config: Sweep configuration.
        """
        self.config = config
        self._sweep_id: str | None = None
        self._results: list[dict[str, Any]] = []
        self._best_result: dict[str, Any] | None = None

    # ── W&B Sweep (requires wandb installed) ────────────────────────

    def create(self) -> str:
        """Create a W&B sweep.

        Returns:
            The sweep ID.
        """
        if not _WANDB_AVAILABLE:
            raise RuntimeError(
                "W&B is not installed. Install with: pip install wandb"
            )

        entity = self.config.entity or os.getenv("WANDB_ENTITY")
        self._sweep_id = wandb.sweep(
            sweep=self.config.to_wandb_dict(),
            project=self.config.project,
            entity=entity,
        )
        return self._sweep_id

    def run_agent(self, train_fn: Callable[[dict[str, Any]], dict[str, Any]], count: int = 10) -> None:
        """Run a W&B sweep agent.

        Args:
            train_fn: Function that takes a config dict and returns a metrics dict.
            count: Maximum number of runs for this agent.
        """
        if not _WANDB_AVAILABLE or self._sweep_id is None:
            raise RuntimeError(
                "Sweep not created or W&B unavailable. Call create() first."
            )

        def _agent():
            run = wandb.init()
            metrics = train_fn(dict(run.config))
            run.log(metrics)
            run.finish()

        wandb.agent(self._sweep_id, function=_agent, count=count)

    # ── Local Grid Search (no W&B needed) ───────────────────────────

    def run_local_grid(
        self,
        train_fn: Callable[[dict[str, Any]], dict[str, Any]],
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a local grid search without W&B.

        Iterates over all hyperparameter combinations, calls train_fn for each,
        and records results. Works in any environment.

        Args:
            train_fn: Function that takes config dict, returns metrics dict.
            output_dir: Directory to save results JSON. None = don't save.

        Returns:
            List of result dicts, each containing config + metrics.
        """
        combinations = self.config.expand_grid()
        metric_name = self.config.metric
        goal = self.config.goal

        for i, params in enumerate(combinations):
            run_name = f"sweep-{self.config.name}-run-{i:03d}"
            start = time.time()

            try:
                metrics = train_fn(params)
            except Exception as exc:
                metrics = {"error": str(exc)}

            duration = time.time() - start
            result = {
                "run_name": run_name,
                "run_id": f"local-{i:03d}",
                "config": deepcopy(params),
                "metrics": metrics,
                "duration_seconds": round(duration, 2),
            }
            self._results.append(result)

        # Determine best run
        valid = [r for r in self._results if "error" not in r.get("metrics", {})]
        if valid:
            key_fn = lambda r: r["metrics"].get(
                metric_name.split("/")[-1], 0
            )
            reverse = goal == "maximize"
            self._results.sort(key=key_fn, reverse=reverse)
            self._best_result = self._results[0]

        # Save results
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            results_path = out_path / f"sweep_{self.config.name}_results.json"
            results_path.write_text(json.dumps(self._results, indent=2))

            # Best result summary
            if self._best_result:
                best_path = out_path / f"sweep_{self.config.name}_best.json"
                best_path.write_text(json.dumps(self._best_result, indent=2))

        return self._results

    @property
    def best_result(self) -> dict[str, Any] | None:
        """Best result from the sweep (highest/lowest metric depending on goal)."""
        return self._best_result

    @property
    def results(self) -> list[dict[str, Any]]:
        """All sweep results."""
        return self._results


# ── Pre-built Sweep Configurations ──────────────────────────────────


def agent_profile_sweep(project: str = "agentops-reliability-platform") -> SweepConfig:
    """Sweep over simulated agent quality parameters."""
    return SweepConfig(
        name="agent-profile-sweep",
        method="grid",
        metric="benchmark/support-tickets/composite_mean",
        goal="maximize",
        project=project,
        parameters=[
            SweepParameter("groundedness_target", values=[0.75, 0.85, 0.95]),
            SweepParameter("verification_pass_rate", values=[0.70, 0.80, 0.90]),
            SweepParameter("tool_success_rate", values=[0.80, 0.90, 0.95]),
            SweepParameter("hallucination_rate", values=[0.01, 0.05, 0.10]),
        ],
        description="Sweep over simulated agent quality profiles to find optimal reliability settings.",
    )


def retrieval_sweep(project: str = "agentops-reliability-platform") -> SweepConfig:
    """Sweep over retrieval engine parameters."""
    return SweepConfig(
        name="retrieval-sweep",
        method="grid",
        metric="benchmark/support-tickets/citation_precision_mean",
        goal="maximize",
        project=project,
        parameters=[
            SweepParameter("chunk_size", values=[256, 512, 1024]),
            SweepParameter("chunk_overlap", values=[32, 64, 128]),
            SweepParameter("top_k", values=[3, 5, 10]),
            SweepParameter("retrieval_method", values=["bm25", "dense", "hybrid"]),
        ],
        description="Sweep over retrieval parameters to optimize citation precision.",
    )
