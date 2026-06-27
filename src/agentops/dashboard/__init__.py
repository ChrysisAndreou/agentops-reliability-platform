"""
Live observability dashboard for the AgentOps Reliability Platform.

Provides:
- Real-time WebSocket trace streaming
- Interactive HTML dashboard with charts
- Aggregate statistics and failure analysis
- Eval result visualization
"""

from .server import DashboardServer, create_dashboard_app

__all__ = ["DashboardServer", "create_dashboard_app"]
