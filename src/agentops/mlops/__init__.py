"""
MLOps integration module — experiment tracking, artifact management, and
hyperparameter optimization via Weights & Biases.

Provides:
- WandBTracker: Track evaluation runs, log metrics and artifacts
- WandBSweep: Hyperparameter sweep configuration and execution
- Local file-system fallback for CI/CD environments without W&B credentials
"""

from agentops.mlops.tracker import WandBTracker
from agentops.mlops.sweeps import SweepConfig, WandBSweep

__all__ = ["WandBTracker", "SweepConfig", "WandBSweep"]
