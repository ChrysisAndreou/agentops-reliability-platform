"""
Multi-Agent System Benchmarks — tasks that require genuine coordination.

Unlike single-agent benchmarks (which test one agent's retrieval/tool-use ability),
these tasks test a multi-agent system's ability to:

1. Decompose complex tasks into appropriate subtasks
2. Route subtasks to the right specialized workers
3. Coordinate parallel execution without deadlocks or redundancy
4. Resolve conflicts when workers produce contradictory results
5. Synthesize heterogeneous outputs into coherent answers
6. Scale gracefully with worker count

Each benchmark task defines:
- The complex task to solve
- Expected decomposition (which worker roles should be assigned)
- Coordination complexity (how much inter-worker communication is needed)
- Ground-truth key facts that a correct answer must contain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CoordinationPattern(str, Enum):
    """How workers must coordinate to solve the task."""
    INDEPENDENT = "independent"          # Workers can operate in isolation
    SEQUENTIAL = "sequential"            # Worker B needs Worker A's output
    CONSENSUS = "consensus"              # Workers must reconcile conflicting views
    PIPELINE = "pipeline"                # Results flow through workers in order
    FAN_OUT = "fan_out"                  # One result distributed to many for analysis
    FAN_IN = "fan_in"                    # Many outputs aggregated into one synthesis


class BenchmarkDifficulty(str, Enum):
    EASY = "easy"        # 2 workers, no conflicts, straightforward decomposition
    MEDIUM = "medium"    # 3 workers, some dependencies, minor conflicts
    HARD = "hard"        # 4 workers, sequential dependencies, conflicting data
    EXPERT = "expert"    # 4+ workers, complex dependencies, adversarial inputs


@dataclass
class MultiAgentBenchmarkTask:
    """A single benchmark task for multi-agent coordination evaluation.

    Attributes:
        id: Unique task identifier
        name: Human-readable task name
        task: The full task description given to the supervisor agent
        domain: Application domain (security, infrastructure, etc.)
        expected_worker_roles: Worker roles that should be assigned
        expected_decomposition_count: How many subtasks the supervisor should create
        coordination_pattern: How workers must coordinate
        difficulty: Task difficulty tier
        key_facts: Ground-truth facts a correct answer must contain
        requires_tool_use: Whether workers need tools to solve this
        has_contradictions: Whether the task contains intentionally contradictory info
        min_workers: Minimum workers needed to solve correctly
    """
    id: str
    name: str
    task: str
    domain: str
    expected_worker_roles: List[str] = field(default_factory=list)
    expected_decomposition_count: int = 2
    coordination_pattern: CoordinationPattern = CoordinationPattern.INDEPENDENT
    difficulty: BenchmarkDifficulty = BenchmarkDifficulty.EASY
    key_facts: List[str] = field(default_factory=list)
    requires_tool_use: bool = False
    has_contradictions: bool = False
    min_workers: int = 2


@dataclass
class MultiAgentBenchmark:
    """A collection of multi-agent benchmark tasks."""
    name: str
    description: str
    tasks: List[MultiAgentBenchmarkTask] = field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    def by_difficulty(self, difficulty: BenchmarkDifficulty) -> List[MultiAgentBenchmarkTask]:
        return [t for t in self.tasks if t.difficulty == difficulty]

    def by_domain(self, domain: str) -> List[MultiAgentBenchmarkTask]:
        return [t for t in self.tasks if t.domain == domain]


# ═══════════════════════════════════════════════════════════════════
# Standard Multi-Agent Coordination Benchmark (10 tasks)
# ═══════════════════════════════════════════════════════════════════

MULTI_AGENT_BENCHMARK = MultiAgentBenchmark(
    name="multi-agent-coordination-benchmark",
    description=(
        "Comprehensive benchmark for multi-agent coordination quality. "
        "10 tasks across 6 domains, testing decomposition, routing, "
        "conflict resolution, and synthesis across all coordination patterns."
    ),
    tasks=[
        # ═══ EASY (2 tasks) ═══

        MultiAgentBenchmarkTask(
            id="ma-001",
            name="Security Policy Audit",
            domain="security",
            task=(
                "Audit our CloudDeploy platform against the security baseline. "
                "Find the current encryption standards for data in transit and at rest. "
                "Verify whether our SOC 2 compliance documentation is up to date. "
                "List any gaps between our security posture and industry best practices."
            ),
            expected_worker_roles=["retrieval_specialist", "verifier"],
            expected_decomposition_count=2,
            coordination_pattern=CoordinationPattern.INDEPENDENT,
            difficulty=BenchmarkDifficulty.EASY,
            key_facts=[
                "TLS 1.3",
                "AES-256",
                "SOC 2",
                "encryption at rest",
            ],
            min_workers=2,
        ),
        MultiAgentBenchmarkTask(
            id="ma-002",
            name="Documentation Completeness Check",
            domain="documentation",
            task=(
                "Evaluate our CloudDeploy documentation for completeness. "
                "Check whether onboarding docs cover 2FA setup, API key management, "
                "and pipeline configuration. Also verify that the platform docs "
                "document all pricing tiers (Shared, Dedicated, Enterprise) with their "
                "resource limits. Report any missing sections."
            ),
            expected_worker_roles=["retrieval_specialist", "verifier"],
            expected_decomposition_count=2,
            coordination_pattern=CoordinationPattern.INDEPENDENT,
            difficulty=BenchmarkDifficulty.EASY,
            key_facts=[
                "2FA",
                "API key",
                "pipeline",
                "Shared",
                "Dedicated",
                "Enterprise",
                "resource limits",
            ],
            min_workers=2,
        ),

        # ═══ MEDIUM (3 tasks) ═══

        MultiAgentBenchmarkTask(
            id="ma-003",
            name="Infrastructure Cost Optimization",
            domain="infrastructure",
            task=(
                "Analyze our CloudDeploy infrastructure costs. First, retrieve our "
                "current pricing tiers and compute the monthly cost for a team of 25 "
                "developers on the Dedicated plan with 3 concurrent pipelines each. "
                "Then, check the pricing docs for any volume discounts or annual "
                "commitment savings. Finally, verify that the computed cost aligns "
                "with the documented pricing model. Flag any discrepancies."
            ),
            expected_worker_roles=["retrieval_specialist", "tool_executor", "verifier"],
            expected_decomposition_count=3,
            coordination_pattern=CoordinationPattern.SEQUENTIAL,
            difficulty=BenchmarkDifficulty.MEDIUM,
            key_facts=[
                "25 developers",
                "3 concurrent pipelines",
                "Dedicated plan",
                "volume discount",
                "annual commitment",
                "cost computation",
            ],
            requires_tool_use=True,
            min_workers=3,
        ),
        MultiAgentBenchmarkTask(
            id="ma-004",
            name="Cross-Region Incident Analysis",
            domain="infrastructure",
            task=(
                "A production incident affected three regions simultaneously. "
                "Retrieve the incident response documentation and standard procedures. "
                "Analyze error logs showing 'connection timeout' in EU-West, "
                "'rate limit exceeded' in US-East, and 'TLS handshake failure' in APAC. "
                "Verify whether the documented incident procedures cover all three "
                "failure modes. Determine if there's a common root cause and recommend "
                "a unified remediation."
            ),
            expected_worker_roles=["retrieval_specialist", "code_analyst", "verifier"],
            expected_decomposition_count=3,
            coordination_pattern=CoordinationPattern.FAN_IN,
            difficulty=BenchmarkDifficulty.MEDIUM,
            key_facts=[
                "connection timeout",
                "rate limit exceeded",
                "TLS handshake failure",
                "EU-West",
                "US-East",
                "APAC",
                "common root cause",
            ],
            min_workers=3,
        ),
        MultiAgentBenchmarkTask(
            id="ma-005",
            name="Compliance Migration Assessment",
            domain="compliance",
            task=(
                "We're migrating a healthcare application from on-premise to CloudDeploy. "
                "Check our platform docs for HIPAA compliance features and data residency "
                "options. Verify whether the Dedicated plan includes BAA support. Analyze "
                "whether our current encryption standards meet healthcare data requirements. "
                "The verifier has flagged potential issues with PHI storage — reconcile the "
                "retrieval and verification outputs. Produce a migration readiness report."
            ),
            expected_worker_roles=["retrieval_specialist", "code_analyst", "verifier"],
            expected_decomposition_count=3,
            coordination_pattern=CoordinationPattern.CONSENSUS,
            difficulty=BenchmarkDifficulty.MEDIUM,
            key_facts=[
                "HIPAA",
                "data residency",
                "BAA",
                "encryption",
                "PHI",
                "healthcare",
                "migration readiness",
            ],
            has_contradictions=True,
            min_workers=3,
        ),

        # ═══ HARD (3 tasks) ═══

        MultiAgentBenchmarkTask(
            id="ma-006",
            name="Zero-Day Vulnerability Response",
            domain="security",
            task=(
                "A critical zero-day vulnerability (CVE-2026-XXXX) has been announced "
                "affecting container runtimes. Retrieve our security policy for "
                "emergency patching procedures. Analyze our infrastructure to determine "
                "which services are vulnerable — the code analyst reports that our "
                "container runtime version is 1.7.3, but platform docs claim we're "
                "on 1.8.1 (CONTRADICTORY). Compute the risk exposure: how many customer "
                "workloads are on the affected version. Verify that our incident response "
                "plan covers container runtime vulnerabilities. The verifier disagrees "
                "with the code analyst's version assessment — resolve the conflict."
            ),
            expected_worker_roles=[
                "retrieval_specialist", "tool_executor", "code_analyst", "verifier"
            ],
            expected_decomposition_count=4,
            coordination_pattern=CoordinationPattern.CONSENSUS,
            difficulty=BenchmarkDifficulty.HARD,
            key_facts=[
                "CVE-2026",
                "container runtime",
                "1.7.3",
                "1.8.1",
                "risk exposure",
                "incident response",
                "emergency patching",
            ],
            requires_tool_use=True,
            has_contradictions=True,
            min_workers=4,
        ),
        MultiAgentBenchmarkTask(
            id="ma-007",
            name="Multi-Tenant Performance Regression",
            domain="infrastructure",
            task=(
                "After the v4.2 deployment, multiple tenants report 5x latency increases. "
                "Retrieve the deployment changelog and performance benchmarks for v4.1 vs v4.2. "
                "Analyze error logs showing increased database connection pool exhaustion "
                "and query timeouts. Compute the throughput degradation as a percentage "
                "comparing v4.1 and v4.2 metrics. Verify whether the deployment met our "
                "canary requirements (10% traffic for 30 minutes before full rollout). "
                "Determine root cause: is it the new connection pool config, the query "
                "planner change, or a cascading failure from tenant isolation changes?"
            ),
            expected_worker_roles=[
                "retrieval_specialist", "tool_executor", "code_analyst", "verifier"
            ],
            expected_decomposition_count=4,
            coordination_pattern=CoordinationPattern.SEQUENTIAL,
            difficulty=BenchmarkDifficulty.HARD,
            key_facts=[
                "v4.2",
                "5x latency",
                "connection pool",
                "query timeout",
                "throughput degradation",
                "canary requirements",
                "10% traffic",
                "30 minutes",
                "root cause",
            ],
            requires_tool_use=True,
            min_workers=4,
        ),
        MultiAgentBenchmarkTask(
            id="ma-008",
            name="Data Breach Forensic Investigation",
            domain="security",
            task=(
                "A suspected data breach has been detected. Retrieve our data breach "
                "response policy and notification requirements under GDPR and CCPA. "
                "Analyze audit logs showing unusual access patterns: 3,000 API calls "
                "from a single IP in a 5-minute window, all targeting customer PII "
                "endpoints. Compute the scope: how many customer records were potentially "
                "exposed given the API response pagination of 100 records per page. "
                "Verify the access was unauthorized by checking against our access "
                "control policy. The code analyst found that the IP belongs to a "
                "legitimate partner with a valid API key — reconcile this finding "
                "with the unusual volume."
            ),
            expected_worker_roles=[
                "retrieval_specialist", "tool_executor", "code_analyst", "verifier"
            ],
            expected_decomposition_count=4,
            coordination_pattern=CoordinationPattern.CONSENSUS,
            difficulty=BenchmarkDifficulty.HARD,
            key_facts=[
                "GDPR",
                "CCPA",
                "3,000 API calls",
                "5-minute window",
                "PII",
                "100 records per page",
                "access control",
                "valid API key",
                "notification requirements",
            ],
            requires_tool_use=True,
            has_contradictions=True,
            min_workers=4,
        ),

        # ═══ EXPERT (2 tasks) ═══

        MultiAgentBenchmarkTask(
            id="ma-009",
            name="AI Agent Platform Architecture Review",
            domain="architecture",
            task=(
                "We need a comprehensive architecture review of our AI agent platform "
                "before its EU AI Act compliance deadline. Retrieve the EU AI Act "
                "high-risk system requirements (Annex III categories, Article 6 "
                "classification rules). Retrieve our current architecture docs and identify "
                "which components process personal data or make autonomous decisions. "
                "Analyze our observability stack for compliance gaps: do we log all "
                "agent decisions with sufficient detail for regulatory audit? Compute "
                "our compliance readiness score by mapping each Annex IV documentation "
                "requirement against what we currently have. Verify that our human "
                "oversight mechanisms meet the Article 14 requirements. The verifier "
                "has flagged that our 'autonomous decision' classification may trigger "
                "HIGH risk tier — the code analyst disagrees because we have human-in-the-loop. "
                "Reconcile and produce a compliance gap report with prioritized remediation."
            ),
            expected_worker_roles=[
                "retrieval_specialist", "tool_executor", "code_analyst", "verifier"
            ],
            expected_decomposition_count=5,
            coordination_pattern=CoordinationPattern.CONSENSUS,
            difficulty=BenchmarkDifficulty.EXPERT,
            key_facts=[
                "EU AI Act",
                "Annex III",
                "Article 6",
                "Article 14",
                "Annex IV",
                "high-risk",
                "human oversight",
                "personal data",
                "autonomous decisions",
                "compliance readiness score",
                "observability",
                "regulatory audit",
            ],
            requires_tool_use=True,
            has_contradictions=True,
            min_workers=4,
        ),
        MultiAgentBenchmarkTask(
            id="ma-010",
            name="Global Platform Outage Root Cause Analysis",
            domain="infrastructure",
            task=(
                "A 47-minute global outage affected all CloudDeploy regions. "
                "Retrieve the incident timeline, deployment records, and our SLA "
                "commitments (99.95% uptime). Analyze infrastructure logs showing "
                "a cascading failure: primary database failover took 12 minutes (SLA "
                "target: 2 minutes), which triggered connection pool exhaustion, which "
                "caused API gateway circuit breakers to open. Compute the financial "
                "impact: 47 minutes of downtime at our SLA penalty rate of 5% monthly "
                "credits per 30 minutes of outage. Verify whether our disaster recovery "
                "plan was followed correctly — the verifier found that the failover was "
                "manual, not automatic, which violates our DR policy. The code analyst "
                "claims the auto-failover was disabled during a maintenance window "
                "3 days ago and never re-enabled. Determine the sequence of failures, "
                "the primary root cause, and the gap between actual and expected recovery."
            ),
            expected_worker_roles=[
                "retrieval_specialist", "tool_executor", "code_analyst", "verifier"
            ],
            expected_decomposition_count=5,
            coordination_pattern=CoordinationPattern.SEQUENTIAL,
            difficulty=BenchmarkDifficulty.EXPERT,
            key_facts=[
                "47-minute",
                "global outage",
                "99.95%",
                "12 minutes failover",
                "2 minutes SLA target",
                "connection pool exhaustion",
                "circuit breakers",
                "5% monthly credits",
                "manual failover",
                "auto-failover disabled",
                "maintenance window",
                "SLA penalty",
                "root cause",
            ],
            requires_tool_use=True,
            has_contradictions=True,
            min_workers=4,
        ),
    ],
)


# ── Factory function ──────────────────────────────────────────────────

def get_multi_agent_benchmark() -> MultiAgentBenchmark:
    """Return the standard multi-agent coordination benchmark."""
    return MULTI_AGENT_BENCHMARK


def get_tasks_by_difficulty(difficulty: BenchmarkDifficulty) -> List[MultiAgentBenchmarkTask]:
    """Get all benchmark tasks at a given difficulty level."""
    return MULTI_AGENT_BENCHMARK.by_difficulty(difficulty)


def get_tasks_by_pattern(pattern: CoordinationPattern) -> List[MultiAgentBenchmarkTask]:
    """Get all benchmark tasks using a given coordination pattern."""
    return [t for t in MULTI_AGENT_BENCHMARK.tasks if t.coordination_pattern == pattern]
