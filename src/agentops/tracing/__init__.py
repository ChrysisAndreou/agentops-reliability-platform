"""
Persistent trace store for agent runs with replay and failure analysis.

Stores complete agent execution traces in SQLite, providing:
- Full trace recording of agent steps (plan, retrieve, execute, verify, respond)
- Queryable trace history by task_id, agent, verification status
- Failure pattern clustering and classification
- Trace replay for deterministic evaluation
"""

from .store import TraceStore
from .classifier import FailureClassifier, FailurePattern
from .opentelemetry import OTelObserver

__all__ = [
    "TraceStore",
    "FailureClassifier",
    "FailurePattern",
    "OTelObserver",
]
