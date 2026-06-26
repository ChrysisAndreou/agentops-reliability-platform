"""
FastAPI application for the AgentOps Reliability Platform.

Provides endpoints for:
- Running agents on tasks
- Inspecting trace history
- Viewing evaluation reports
- Checking system health
"""

from .app import create_app

__all__ = ["create_app"]
