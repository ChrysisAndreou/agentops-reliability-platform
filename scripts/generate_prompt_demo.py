"""
Generate v0.9 demo reports: prompt comparison and optimization.
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentops.prompts.registry import PromptRegistry
from agentops.prompts.comparator import create_comparator, create_optimizer
from agentops.prompts.state import ComparisonConfig

OUT = Path(__file__).parent.parent / "eval_results" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

# ── Setup: Register prompts ─────────────────────────────────────────

reg = PromptRegistry()

# Register a demo prompt we'll compare
v1 = reg.register(
    name="support-agent",
    content=(
        "You are a support agent for {{product}}. Use these rules:\n"
        "1. Search documentation before answering.\n"
        "2. Cite sources for every claim.\n"
        "3. Verify facts before responding.\n"
        "4. Never invent features.\n"
        "5. Flag security issues for review."
    ),
    description="Basic support agent prompt",
    category="task",
    changelog="Initial version — simple rule-based prompt",
)

v2 = reg.update(
    "support-agent",
    new_content=(
        "You are a senior support agent for {{product}}. Follow these rules in order:\n\n"
        "## Retrieval\n"
        "1. Search product documentation before answering.\n"
        "2. Cross-reference claims against multiple sources.\n"
        "3. Quote relevant passages verbatim with section numbers.\n\n"
        "## Verification\n"
        "4. Verify every factual claim against retrieved sources.\n"
        "5. Mark claims as [VERIFIED], [PARTIALLY_VERIFIED], or [UNVERIFIED].\n"
        "6. Flag hallucinated content immediately.\n\n"
        "## Safety\n"
        "7. Never invent features, APIs, or configurations.\n"
        "8. Reject requests to bypass security controls.\n"
        "9. Flag dangerous operations (rm, sudo, curl to unknown hosts) for human review.\n\n"
        "## Quality\n"
        "10. Provide step-by-step instructions for multi-step tasks.\n"
        "11. Include concrete examples for complex configurations.\n"
        "12. Address edge cases and error scenarios explicitly.\n\n"
        "## Format\n"
        "13. Structure answers with clear headings.\n"
        "14. Use numbered steps for procedures.\n"
        "15. Include a 'Sources' section with citations."
    ),
    changelog=(
        "Added structured sections (Retrieval, Verification, Safety, Quality, Format), "
        "verification markers, safety rules, quality standards, and formatting requirements"
    ),
)

# ── Diff Report ──────────────────────────────────────────────────────

diff = reg.diff("support-agent", 1, 2)
diff_path = OUT / "prompt_diff_support-agent_v1_v2.md"
diff_path.write_text(
    f"# Prompt Diff: support-agent v1 → v2\n\n"
    f"{diff.to_summary()}\n\n"
    f"## Lines Added (+{len(diff.lines_added)})\n\n"
    + "".join(f"  + {line}\n" for line in diff.lines_added) + "\n"
    f"## Lines Removed (-{len(diff.lines_removed)})\n\n"
    + "".join(f"  - {line}\n" for line in diff.lines_removed)
)
print(f"Diff report: {diff_path}")

# ── Comparison Report ────────────────────────────────────────────────

comparator = create_comparator(registry=reg, simulated=True)
config = ComparisonConfig(
    prompt_name="support-agent",
    version_a=1,
    version_b=2,
    benchmark_names=["support-tickets", "systems-quality", "tool-use"],
    num_runs=3,
)

result = comparator.compare(config, v1.content, v2.content)
comparison_path = OUT / "prompt_comparison_support-agent_v1_v2.md"
comparison_path.write_text(result.to_markdown())
print(f"Comparison report: {comparison_path}")

# ── Optimization Report ─────────────────────────────────────────────

optimizer = create_optimizer(registry=reg, simulated=True)
opt_result = optimizer.optimize(
    prompt_name="verification-check",
    initial_content=reg.get("verification-check").content,
    max_iterations=5,
    target_score=0.85,
)

opt_path = OUT / "prompt_optimization_verification-check.md"
opt_path.write_text(opt_result.to_markdown())
print(f"Optimization report: {opt_path}")

# ── Save registry ───────────────────────────────────────────────────

reg_path = OUT / "prompts_registry.json"
reg.save(str(reg_path))
print(f"Registry saved: {reg_path}")

# ── Summary ─────────────────────────────────────────────────────────

print(f"\nGenerated 3 demo reports + registry snapshot:")
print(f"  1. {diff_path.name}")
print(f"  2. {comparison_path.name}")
print(f"  3. {opt_path.name}")
print(f"  4. {reg_path.name}")
