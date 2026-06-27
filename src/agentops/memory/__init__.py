"""
Agent Memory Evaluation — test whether agents maintain context across
multi-turn conversations, recall facts correctly, and resist degradation.

Module for v0.12: provides memory state models, simulated memory agent
with configurable degradation profiles, and systematic memory evaluation
metrics. All testable without API keys via deterministic simulation.
"""

from agentops.memory.metrics import MemoryEvaluator
from agentops.memory.simulator import SimulatedMemoryAgent
from agentops.memory.state import (
    MEMORY_PROFILES,
    MemoryContext,
    MemoryEntry,
    MemoryMetrics,
    MemoryRecallResult,
    MemoryReport,
    MemoryStore,
    MemoryType,
    get_memory_profile,
)

__all__ = [
    "MemoryType",
    "MemoryEntry",
    "MemoryStore",
    "MemoryContext",
    "MemoryRecallResult",
    "MemoryMetrics",
    "MemoryReport",
    "MEMORY_PROFILES",
    "get_memory_profile",
    "SimulatedMemoryAgent",
    "MemoryEvaluator",
]
