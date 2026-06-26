"""
CLI for the AgentOps Reliability Platform.

Commands:
  agentops run <task>               — run agent on a single task
  agentops eval [--benchmark NAME]  — run evaluation benchmarks
  agentops serve [--port PORT]      — start FastAPI server
  agentops traces                   — list recent traces
  agentops trace <run_id>           — inspect a specific trace
  agentops stats                    — aggregate statistics
  agentops index <docs_dir>         — index documents for retrieval
"""

from .main import app

__all__ = ["app"]
