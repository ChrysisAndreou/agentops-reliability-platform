"""
Production Readiness Assessment — v0.18.0

Synthesises 17 prior versions of evaluation infrastructure into a unified
production-readiness signal. Computes a composite score across 8 dimensions
(verification, safety, tools, response quality, retrieval, latency, memory,
multi-agent coordination), assigns a readiness tier with evidence, and
generates structured reports for CI gating and stakeholder communication.

This is the "define what ready for production means" layer that caps the
AgentOps platform.
"""

from agentops.readiness.state import (
    ReadinessDimension,
    ReadinessDimensionScore,
    ReadinessReport,
    ReadinessTier,
    READINESS_DIMENSIONS,
    READINESS_THRESHOLDS,
)
from agentops.readiness.assessor import ReadinessAssessor
from agentops.readiness.reporting import (
    format_readiness_json,
    format_readiness_markdown,
    format_readiness_html,
)

__all__ = [
    "ReadinessAssessor",
    "ReadinessDimension",
    "ReadinessDimensionScore",
    "ReadinessReport",
    "ReadinessTier",
    "READINESS_DIMENSIONS",
    "READINESS_THRESHOLDS",
    "format_readiness_json",
    "format_readiness_markdown",
    "format_readiness_html",
]
