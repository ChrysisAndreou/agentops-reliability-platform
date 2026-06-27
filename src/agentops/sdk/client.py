"""
AgentOps HTTP client — typed wrapper around the AgentOps API server.

Handles connection management, retries, batching, and typed response parsing.
Uses only stdlib + requests (already a transitive dep) to keep the SDK lightweight.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any

from .state import SDKConfig, RunContext, TraceSpan, TraceStatus, SpanStatus


class AgentOpsHTTPClient:
    """HTTP client for communicating with an AgentOps API server.

    Usage:
        client = AgentOpsHTTPClient(SDKConfig(endpoint="http://localhost:8000"))
        if client.health_check():
            traces = client.list_traces(limit=10)
    """

    def __init__(self, config: SDKConfig | None = None):
        self.config = config or SDKConfig()
        self._session_ready = False

    def _url(self, path: str) -> str:
        base = self.config.endpoint.rstrip("/")
        return f"{base}{path}"

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the AgentOps server.

        Retries on transient errors up to config.max_retries.
        Returns parsed JSON response.
        """
        url = self._url(path)
        timeout = timeout or self.config.timeout_seconds
        body = json.dumps(data).encode("utf-8") if data else None

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method=method
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.code < 500:
                    raise ConnectionError(
                        f"AgentOps server returned {e.code}: {e.reason}"
                    ) from e
                last_error = e
            except (urllib.error.URLError, OSError) as e:
                last_error = e

            if attempt < self.config.max_retries:
                time.sleep(0.5 * (2 ** attempt))

        raise ConnectionError(
            f"Failed to connect to AgentOps at {url} after "
            f"{self.config.max_retries + 1} attempts"
        ) from last_error

    def health_check(self) -> bool:
        """Check if the AgentOps server is reachable and healthy.

        Returns True if the server responds with status=ok.
        """
        try:
            resp = self._request("GET", "/health", timeout=5.0)
            self._session_ready = resp.get("status") == "ok"
            return self._session_ready
        except (ConnectionError, OSError):
            self._session_ready = False
            return False

    def submit_trace(self, ctx: RunContext) -> dict[str, Any]:
        """Submit a completed agent run trace to the server.

        This wraps the POST /api/run endpoint with the run context data.
        """
        payload = {
            "task": ctx.task,
            "task_id": ctx.run_id,
            "context": json.dumps({
                "model": ctx.model,
                "final_answer": ctx.final_answer,
                "verification_passed": ctx.verification_passed,
                "verification_notes": ctx.verification_notes,
                "grounded_claims": ctx.grounded_claims,
                "ungrounded_claims": ctx.ungrounded_claims,
                "citations_used": ctx.citations_used,
                "plan_steps": ctx.plan_steps,
                "tool_calls": [tc.to_dict() for tc in ctx.tool_calls],
                "retrievals": [r.to_dict() for r in ctx.retrievals],
                "error": ctx.error,
                "metadata": ctx.metadata,
                "trace": ctx.root_span.to_dict() if ctx.root_span else None,
            }),
        }
        return self._request("POST", "/api/run", data=payload)

    def list_traces(
        self,
        verification_passed: bool | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List traces from the server, optionally filtered."""
        params = []
        if verification_passed is not None:
            params.append(f"verification_passed={str(verification_passed).lower()}")
        params.append(f"limit={limit}")
        path = f"/api/traces?{'&'.join(params)}"
        resp = self._request("GET", path)
        return resp.get("traces", [])

    def get_trace(self, run_id: str) -> dict[str, Any] | None:
        """Get full trace detail for a given run ID."""
        try:
            return self._request("GET", f"/api/traces/{run_id}")
        except ConnectionError as e:
            if "404" in str(e):
                return None
            raise

    def get_replay(self, run_id: str) -> dict[str, Any] | None:
        """Get replay data for a trace."""
        try:
            return self._request("GET", f"/api/traces/{run_id}/replay")
        except ConnectionError as e:
            if "404" in str(e):
                return None
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics from the server."""
        return self._request("GET", "/api/stats")

    def list_evals(self) -> dict[str, Any]:
        """List evaluation runs on the server."""
        return self._request("GET", "/api/evals")

    def get_eval(self, eval_id: str) -> dict[str, Any] | None:
        """Get a specific evaluation report."""
        try:
            return self._request("GET", f"/api/evals/{eval_id}")
        except ConnectionError as e:
            if "404" in str(e):
                return None
            raise

    def run_eval(
        self, benchmark: str, profile: str = "production"
    ) -> dict[str, Any]:
        """Trigger an evaluation run via the server.

        This uses the server's /api/run endpoint iteratively for each
        benchmark task. For production use, call server-side eval endpoints.
        """
        # The server exposes /api/evals for listing, not triggering.
        # For SDK eval triggering, we'd need the CLI or a dedicated endpoint.
        # Return a helpful message instead.
        return {
            "status": "not_implemented",
            "message": (
                "Eval triggering via SDK requires the CLI. "
                "Use the server's agentops CLI directly or POST "
                "individual tasks via /api/run."
            ),
        }
