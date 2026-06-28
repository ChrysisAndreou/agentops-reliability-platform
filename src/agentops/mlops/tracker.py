"""
W&B experiment tracking for AgentOps evaluation runs.

Automatically logs benchmark results, metrics, failure patterns, and artifacts
to Weights & Biases. Falls back to local JSON logging when W&B is unavailable
(e.g., CI environments or local-only execution).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WANDB_AVAILABLE = False
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]


@dataclass
class RunMetadata:
    """Metadata for a tracked experiment run."""

    run_name: str
    project: str
    entity: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    mode: str = "online"  # "online", "offline", "disabled"
    run_id: str = ""
    started_at: str = ""
    status: str = "running"
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


class WandBTracker:
    """Track AgentOps evaluation runs in Weights & Biases.

    Gracefully falls back to local JSON logging when W&B is unavailable.
    Supports both simulated-agent benchmarks and real LLM-backed agent runs.

    Usage:
        tracker = WandBTracker(project="agentops-evals", tags=["v0.19", "simulated"])
        tracker.init_run(run_name="baseline-gpt4o")

        # Log individual metrics
        tracker.log_metrics({"composite": 0.85, "groundedness": 0.92}, step=1)

        # Log a full benchmark report
        tracker.log_benchmark_report(report)

        # Log artifacts
        tracker.log_artifact("reports/support-tickets_report.md")

        tracker.finish()
    """

    DEFAULT_PROJECT = "agentops-reliability-platform"

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str = "online",
        tags: list[str] | None = None,
        notes: str = "",
        local_dir: str | None = None,
    ):
        """Initialize the W&B tracker.

        Args:
            project: W&B project name. Defaults to "agentops-reliability-platform".
            entity: W&B entity (team/user). Defaults to WANDB_ENTITY env var.
            config: Hyperparameters / configuration to log.
            mode: "online", "offline", or "disabled".
            tags: List of tags for the run.
            notes: Run description / notes.
            local_dir: Directory for local JSON fallback. Defaults to "./wandb_local".
        """
        self._project = project or os.getenv("WANDB_PROJECT", self.DEFAULT_PROJECT)
        self._entity = entity or os.getenv("WANDB_ENTITY")
        self._config = config or {}
        self._mode = mode if _WANDB_AVAILABLE else "local"
        self._tags = tags or []
        self._notes = notes
        self._local_dir = Path(local_dir or os.getenv("AGENTOPS_WANDB_LOCAL_DIR", "./wandb_local"))
        self._run: Any = None
        self._metadata: RunMetadata | None = None
        self._step = 0

    # ── Public API ──────────────────────────────────────────────────

    def init_run(self, run_name: str | None = None, run_id: str | None = None) -> RunMetadata:
        """Initialize a tracking run.

        Args:
            run_name: Human-readable run name. Auto-generated if None.
            run_id: Explicit run ID. Auto-generated if None.

        Returns:
            RunMetadata describing the initialized run.
        """
        run_name = run_name or f"agentops-{int(time.time())}"
        self._metadata = RunMetadata(
            run_name=run_name,
            project=self._project,
            entity=self._entity,
            config=self._config,
            tags=self._tags,
            notes=self._notes,
            mode=self._mode,
            run_id=run_id or "",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        if self._mode in ("online", "offline") and _WANDB_AVAILABLE:
            self._run = wandb.init(
                project=self._project,
                entity=self._entity,
                name=run_name,
                id=run_id,
                config=self._config,
                tags=self._tags,
                notes=self._notes,
                mode={"online": "online", "offline": "offline"}.get(self._mode, "offline"),
                reinit=True,
            )
            self._metadata.run_id = self._run.id if self._run else ""
        else:
            self._run = None
            self._metadata.run_id = run_id or _generate_run_id()

        return self._metadata

    def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log scalar metrics for the current step.

        Args:
            metrics: Dict of metric_name -> value.
            step: Global step number. Auto-increments if None.
        """
        if step is not None:
            self._step = step

        if self._metadata is not None:
            entry = {"_step": self._step, "_timestamp": time.time(), **metrics}
            self._metadata.metrics_history.append(entry)

        if self._run is not None and _WANDB_AVAILABLE:
            self._run.log(metrics, step=self._step)

        self._step += 1

    def log_benchmark_report(self, report: Any, benchmark_config: dict[str, Any] | None = None) -> None:
        """Log a full benchmark evaluation report to W&B.

        Args:
            report: EvalReport from the evaluation harness (or dict).
            benchmark_config: Additional config to log alongside metrics.
        """
        # Accept both EvalReport objects and dicts
        if hasattr(report, "summary") and hasattr(report, "results"):
            summary = report.summary
            results = report.results
            benchmark_name = getattr(report, "benchmark_name", "unknown")
            model = getattr(report, "model", "unknown")
            failure_patterns = getattr(report, "failure_patterns", [])
        elif isinstance(report, dict):
            summary = report.get("summary", {})
            results = report.get("results", [])
            benchmark_name = report.get("benchmark_name", "unknown")
            model = report.get("model", "unknown")
            failure_patterns = report.get("failure_patterns", [])
        else:
            raise TypeError(f"Expected EvalReport or dict, got {type(report).__name__}")

        # Log benchmark-level summary metrics
        self.log_metrics(
            {f"benchmark/{benchmark_name}/{k}": v for k, v in summary.items()
             if isinstance(v, (int, float))},
        )

        # Log per-task metric distributions as histograms (when W&B available)
        if results:
            if hasattr(results[0], "to_dict"):
                result_dicts = [r.to_dict() for r in results]
            else:
                result_dicts = results

            # Aggregate per-task metrics
            composites = []
            groundedness_vals = []
            latencies = []
            verification_passes = 0
            for rd in result_dicts:
                if isinstance(rd, dict):
                    if "composite" in rd:
                        composites.append(rd["composite"])
                    if "groundedness" in rd:
                        groundedness_vals.append(rd["groundedness"])
                    if "latency_ms" in rd:
                        latencies.append(rd["latency_ms"])
                    if rd.get("verification_passed"):
                        verification_passes += 1

            if composites and self._run is not None and _WANDB_AVAILABLE:
                self._run.log(
                    {
                        f"benchmark/{benchmark_name}/composite_histogram": wandb.Histogram(composites),
                        f"benchmark/{benchmark_name}/groundedness_histogram": wandb.Histogram(
                            groundedness_vals
                        ),
                    }
                )

        # Log failure patterns as a table
        if failure_patterns and self._run is not None and _WANDB_AVAILABLE:
            columns = ["name", "count", "severity", "description"]
            rows = [
                [
                    fp.get("name", ""),
                    fp.get("count", 0),
                    fp.get("severity", ""),
                    fp.get("description", ""),
                ]
                for fp in failure_patterns
            ]
            self._run.log(
                {f"benchmark/{benchmark_name}/failure_patterns": wandb.Table(columns=columns, data=rows)}
            )

        # Log metadata
        if benchmark_config:
            self.log_config({f"benchmark/{benchmark_name}/config": benchmark_config})

    def log_config(self, config: dict[str, Any]) -> None:
        """Update the run configuration."""
        if self._metadata:
            self._metadata.config.update(config)
        if self._run is not None and _WANDB_AVAILABLE:
            self._run.config.update(config, allow_val_change=True)

    def log_artifact(
        self,
        file_path: str,
        artifact_type: str = "eval-output",
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Log a file artifact to W&B or local storage.

        Args:
            file_path: Path to the file to log.
            artifact_type: Category label for the artifact.
            name: Artifact name. Defaults to the filename.
            metadata: Additional metadata for the artifact.

        Returns:
            Path/URI to the logged artifact, or None on failure.
        """
        path = Path(file_path)
        if not path.exists():
            return None

        name = name or path.name

        if self._run is not None and _WANDB_AVAILABLE:
            artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
            artifact.add_file(str(path))
            self._run.log_artifact(artifact)
            # Construct a W&B URI
            entity = self._entity or os.getenv("WANDB_ENTITY", "unknown")
            project = self._project
            run_id = self._run.id or "unknown"
            return f"wandb://{entity}/{project}/runs/{run_id}/artifacts/{name}"
        else:
            # Local fallback
            self._local_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            dest = self._local_dir / name
            shutil.copy2(str(path), str(dest))
            if self._metadata:
                self._metadata.artifacts.append(str(dest))
            return str(dest)

    def log_model(
        self,
        model_path: str,
        model_name: str,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> str | None:
        """Log a model artifact to the W&B Model Registry.

        Args:
            model_path: Path to the model directory or file.
            model_name: Registered model name.
            metadata: Model metadata (framework, version, metrics).
            aliases: Registry aliases (e.g., ["latest", "production-ready"]).

        Returns:
            Model registry URI or local path.
        """
        path = Path(model_path)
        if not path.exists():
            return None

        if self._run is not None and _WANDB_AVAILABLE:
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                metadata=metadata or {},
            )
            artifact.add_file(str(path)) if path.is_file() else artifact.add_dir(str(path))
            self._run.log_artifact(artifact, aliases=aliases or ["latest"])
            entity = self._entity or os.getenv("WANDB_ENTITY", "unknown")
            return f"wandb://{entity}/model-registry/{model_name}"
        else:
            # Local fallback
            model_dir = self._local_dir / "models" / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            if path.is_file():
                shutil.copy2(str(path), str(model_dir / path.name))
            else:
                shutil.copytree(str(path), str(model_dir), dirs_exist_ok=True)
            if self._metadata:
                self._metadata.artifacts.append(str(model_dir))
            return str(model_dir)

    def finish(self, exit_code: int = 0) -> RunMetadata | None:
        """Finish the run and persist metadata.

        Args:
            exit_code: 0 for success, non-zero for failure.
        """
        if self._metadata:
            self._metadata.status = "finished" if exit_code == 0 else "failed"

        # Save local JSON record
        if self._metadata:
            self._save_local_metadata()

        if self._run is not None and _WANDB_AVAILABLE:
            self._run.finish(exit_code=exit_code)

        return self._metadata

    def summary(self) -> dict[str, Any]:
        """Get a lightweight run summary."""
        if self._run is not None and _WANDB_AVAILABLE:
            return dict(self._run.summary)
        if self._metadata and self._metadata.metrics_history:
            return self._metadata.metrics_history[-1]
        return {}

    @property
    def is_available(self) -> bool:
        """Whether W&B is available for online tracking."""
        return _WANDB_AVAILABLE and self._mode != "local"

    @property
    def run(self) -> Any:
        """The underlying W&B run object (None if unavailable)."""
        return self._run

    @property
    def metadata(self) -> RunMetadata | None:
        """Current run metadata."""
        return self._metadata

    # ── Internal ────────────────────────────────────────────────────

    def _save_local_metadata(self) -> None:
        """Persist run metadata to local JSON for CI/CD environments."""
        if self._metadata is None:
            return
        self._local_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "run_name": self._metadata.run_name,
            "run_id": self._metadata.run_id,
            "project": self._metadata.project,
            "entity": self._metadata.entity,
            "config": self._metadata.config,
            "tags": self._metadata.tags,
            "notes": self._metadata.notes,
            "mode": self._metadata.mode,
            "started_at": self._metadata.started_at,
            "status": self._metadata.status,
            "artifacts": self._metadata.artifacts,
            "metrics_count": len(self._metadata.metrics_history),
            "metrics_history": self._metadata.metrics_history,
        }

        run_file = self._local_dir / f"{self._metadata.run_name}.json"
        run_file.write_text(json.dumps(record, indent=2))


def _generate_run_id() -> str:
    """Generate a unique run ID when W&B is unavailable."""
    import uuid

    return uuid.uuid4().hex[:12]
