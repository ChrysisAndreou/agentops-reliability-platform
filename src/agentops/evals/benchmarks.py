"""
Reliability benchmark definitions.

Each benchmark is a collection of tasks that test an agent's ability to:
- Retrieve relevant information from documentation
- Ground answers in retrieved evidence with citations
- Use tools correctly when needed
- Pass verification checks on factual accuracy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkTask:
    """A single evaluation task for the reliability benchmark."""
    id: str
    question: str
    category: str  # "retrieval", "tool_use", "verification", "multi_step"
    key_terms: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    requires_tool: bool = False
    difficulty: str = "medium"


@dataclass
class ReliabilityBenchmark:
    """A collection of benchmark tasks."""
    name: str
    description: str
    tasks: list[BenchmarkTask]


# ── Support Ticket Resolution Benchmark ─────────────────────────────

SUPPORT_TICKETS_BENCH = ReliabilityBenchmark(
    name="support-tickets",
    description="Resolve CloudDeploy support tickets using product documentation",
    tasks=[
        BenchmarkTask(
            id="st-001",
            question="A user reports their build failed with 'Docker daemon not available'. They are on the Dedicated plan and recently updated their clouddeploy.yml. What should they check?",
            category="retrieval",
            key_terms=["runtime", "docker", "Dedicated", "pipeline settings"],
            expected_sources=["clouddeploy-platform.md"],
            difficulty="easy",
        ),
        BenchmarkTask(
            id="st-002",
            question="How do I enable two-factor authentication on my CloudDeploy account? What options are available?",
            category="retrieval",
            key_terms=["TOTP", "SMS", "Settings", "Security", "recovery codes"],
            expected_sources=["clouddeploy-onboarding.md", "clouddeploy-platform.md"],
            difficulty="easy",
        ),
        BenchmarkTask(
            id="st-003",
            question="A Java monolith build runs out of memory on Shared agents (4 GB limit). What optimisation steps can they try before upgrading to Dedicated?",
            category="retrieval",
            key_terms=["multi-stage", "Docker", "parallel", "4 GB", "Dedicated", "16 GB"],
            expected_sources=["clouddeploy-platform.md"],
            difficulty="medium",
        ),
        BenchmarkTask(
            id="st-004",
            question="A production deployment was automatically rolled back from v3.7.2 to v3.7.1. The health check at /health returned 503 after 30 seconds. What caused this and how can it be prevented?",
            category="verification",
            key_terms=["health check", "503", "auto-rollback", "blue-green", "smoke tests"],
            expected_sources=["clouddeploy-platform.md"],
            difficulty="medium",
        ),
        BenchmarkTask(
            id="st-005",
            question="What encryption standards does CloudDeploy use for data in transit and at rest?",
            category="retrieval",
            key_terms=["TLS 1.3", "AES-256", "encryption", "GDPR"],
            expected_sources=["clouddeploy-security.md", "clouddeploy-platform.md"],
            difficulty="easy",
        ),
    ],
)


# ── Systems Quality Benchmark ───────────────────────────────────────

SYSTEMS_QUALITY_BENCH = ReliabilityBenchmark(
    name="systems-quality",
    description="Evaluate reliability/quality characteristics of the CloudDeploy platform",
    tasks=[
        BenchmarkTask(
            id="sq-001",
            question="What deployment strategies does CloudDeploy support? Compare rolling, blue-green, and canary deployments in terms of risk and rollback behaviour.",
            category="multi_step",
            key_terms=["rolling", "blue-green", "canary", "rollback", "traffic", "health check"],
            expected_sources=["clouddeploy-platform.md"],
            difficulty="medium",
        ),
        BenchmarkTask(
            id="sq-002",
            question="What is CloudDeploy's incident response SLA for critical, high, and medium severity incidents?",
            category="retrieval",
            key_terms=["15-minute", "1-hour", "4-hour", "24-hour", "SLA", "critical"],
            expected_sources=["clouddeploy-security.md"],
            difficulty="medium",
        ),
        BenchmarkTask(
            id="sq-003",
            question="What build agent tiers does CloudDeploy offer, and what are the memory limits and concurrent build limits for each?",
            category="retrieval",
            key_terms=["Shared", "Dedicated", "Enterprise", "4 GB", "16 GB", "64 GB"],
            expected_sources=["clouddeploy-platform.md"],
            difficulty="easy",
        ),
        BenchmarkTask(
            id="sq-004",
            question="A user generated an API token set to never expire, but it returns 401 Unauthorized. What are the possible causes?",
            category="verification",
            key_terms=["revoked", "Authorization", "Bearer", "role", "account disabled"],
            expected_sources=["clouddeploy-onboarding.md"],
            difficulty="medium",
        ),
        BenchmarkTask(
            id="sq-005",
            question="Summarise CloudDeploy's password policy and session management rules.",
            category="retrieval",
            key_terms=["12 characters", "uppercase", "lowercase", "number", "special", "8 hours", "5 concurrent"],
            expected_sources=["clouddeploy-security.md"],
            difficulty="easy",
        ),
    ],
)


# ── All benchmarks ──────────────────────────────────────────────────

ALL_BENCHMARKS = [SUPPORT_TICKETS_BENCH, SYSTEMS_QUALITY_BENCH]


def get_benchmark(name: str) -> ReliabilityBenchmark | None:
    for b in ALL_BENCHMARKS:
        if b.name == name:
            return b
    return None
