"""
Dashboard server with WebSocket live streaming and interactive HTML UI.

Usage:
    server = DashboardServer(trace_store, eval_results)
    app = server.create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    _TEMPLATES_AVAILABLE = True
except ImportError:
    _TEMPLATES_AVAILABLE = False


_TEMPLATE_DIR = Path(__file__).parent / "templates"


class DashboardServer:
    """Live observability dashboard with WebSocket streaming.

    Features:
    - Real-time trace feed via WebSocket
    - Interactive HTML dashboard with Chart.js visualizations
    - Summary statistics (pass rate, latency, failure breakdown)
    - Eval result viewer
    - Failure pattern analysis
    """

    def __init__(
        self,
        trace_store=None,
        eval_results: dict[str, Any] | None = None,
        agent=None,
    ):
        self.trace_store = trace_store
        self.eval_results = eval_results or {}
        self.agent = agent
        self._ws_clients: set[WebSocket] = set()

    def create_app(self) -> FastAPI:
        app = FastAPI(
            title="AgentOps Dashboard",
            description="Live observability dashboard for AI agent reliability",
            version="0.10.0",
        )

        app.get("/health")(self._health)
        app.get("/api/dashboard/stats")(self._dashboard_stats)
        app.get("/api/dashboard/traces")(self._dashboard_traces)
        app.get("/api/dashboard/evals")(self._dashboard_evals)
        app.get("/api/dashboard/failures")(self._failure_analysis)
        app.websocket("/ws")(self._websocket_endpoint)

        if _TEMPLATES_AVAILABLE:
            app.get("/", response_class=HTMLResponse)(self._dashboard_html)
            app.get("/dashboard", response_class=HTMLResponse)(self._dashboard_html)

        return app

    async def _health(self):
        return {"status": "ok", "dashboard": "v0.10.0", "timestamp": time.time()}

    async def _dashboard_stats(self):
        """Aggregate statistics for the dashboard."""
        if not self.trace_store:
            return {"error": "No trace store configured"}

        stats = self.trace_store.stats()
        total = stats.get("total_runs", 0)

        if total == 0:
            return {
                "total_runs": 0,
                "pass_rate": 0,
                "failure_rate": 0,
                "avg_latency_ms": 0,
                "recent_runs": 0,
            }

        # Count recent runs (last hour)
        recent = 0
        conn = self.trace_store._get_conn()
        time.time() - 3600
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE total_latency_ms > 0"
            ).fetchone()
            if row:
                recent = row[0]
        except Exception:
            pass

        # Failure breakdown
        failures = self._get_failure_breakdown()

        return {
            "total_runs": total,
            "pass_rate": stats.get("verification_pass_rate", 0),
            "failure_rate": stats.get("failure_rate", 0),
            "avg_latency_ms": stats.get("avg_latency_ms", 0),
            "recent_runs": recent,
            "failure_breakdown": failures,
        }

    async def _dashboard_traces(self, limit: int = 50, failed_only: bool = False):
        """Get trace summaries for the dashboard."""
        if not self.trace_store:
            return {"traces": [], "count": 0}

        traces = self.trace_store.query(
            verification_passed=None if not failed_only else False,
            limit=limit,
        )

        return {
            "count": len(traces),
            "traces": [
                {
                    "run_id": t.run_id,
                    "task": t.task[:80],
                    "verification_passed": t.verification_passed,
                    "success": t.success,
                    "tool_calls": t.tool_calls_count,
                    "latency_ms": t.total_latency_ms,
                    "error": t.error[:100] if t.error else None,
                }
                for t in traces
            ],
        }

    async def _dashboard_evals(self):
        """Get eval run summaries for the dashboard."""
        return {
            "eval_runs": list(self.eval_results.keys()),
            "count": len(self.eval_results),
            "details": self.eval_results,
        }

    async def _failure_analysis(self):
        """Analyze failure patterns from trace data."""
        if not self.trace_store:
            return {"patterns": [], "total_failures": 0}

        conn = self.trace_store._get_conn()
        failed = conn.execute(
            "SELECT task, error, verification_notes, total_latency_ms "
            "FROM traces WHERE verification_passed = 0 OR success = 0 "
            "ORDER BY created_at DESC LIMIT 100"
        ).fetchall()

        # Categorize failures
        categories: dict[str, int] = {}
        for row in failed:
            error = row["error"] or ""
            notes = row["verification_notes"] or ""
            combined = (error + " " + notes).lower()

            if "hallucination" in combined or "ungrounded" in combined:
                cat = "hallucination"
            elif "tool" in combined or "tool_call" in combined:
                cat = "tool_error"
            elif "timeout" in combined or "latency" in combined:
                cat = "timeout"
            elif "retrieval" in combined or "no results" in combined:
                cat = "retrieval_failure"
            elif "verification" in combined:
                cat = "verification_failure"
            else:
                cat = "other"

            categories[cat] = categories.get(cat, 0) + 1

        patterns = [
            {"category": cat, "count": count}
            for cat, count in sorted(categories.items(), key=lambda x: -x[1])
        ]

        return {
            "total_failures": len(failed),
            "patterns": patterns,
        }

    async def _websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for live trace streaming."""
        await websocket.accept()
        self._ws_clients.add(websocket)

        try:
            # Send initial stats
            stats = await self._dashboard_stats()
            await websocket.send_json({"type": "stats", "data": stats})

            # Keep connection alive and push updates
            time.time()
            while True:
                await asyncio.sleep(5)

                # Check if client is still connected
                try:
                    # Send heartbeat
                    await websocket.send_json({"type": "heartbeat", "ts": time.time()})

                    # Push updated stats if trace store exists
                    if self.trace_store:
                        new_stats = await self._dashboard_stats()
                        await websocket.send_json({"type": "stats", "data": new_stats})

                        # Push recent traces
                        traces = await self._dashboard_traces(limit=5)
                        await websocket.send_json({"type": "traces", "data": traces})

                except Exception:
                    break

        except WebSocketDisconnect:
            pass
        finally:
            self._ws_clients.discard(websocket)

    async def broadcast(self, event_type: str, data: dict):
        """Broadcast an event to all connected WebSocket clients."""
        dead: set[WebSocket] = set()
        for ws in self._ws_clients:
            try:
                await ws.send_json({"type": event_type, "data": data})
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    def _get_failure_breakdown(self) -> dict[str, int]:
        """Get failure category breakdown."""
        if not self.trace_store:
            return {}

        conn = self.trace_store._get_conn()
        # Count by error pattern
        total_failed = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE verification_passed = 0"
        ).fetchone()[0]

        return {"verification_failed": total_failed or 0}

    async def _dashboard_html(self, request: Request):
        """Serve the interactive dashboard HTML page."""
        if not _TEMPLATES_AVAILABLE:
            return HTMLResponse(content=_FALLBACK_HTML, status_code=200)

        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

        try:
            template = env.get_template("dashboard.html")
            stats = await self._dashboard_stats()
            return HTMLResponse(
                template.render(
                    title="AgentOps Dashboard",
                    stats=stats,
                    ws_url=f"ws://{request.url.hostname}:{request.url.port}/ws",
                )
            )
        except Exception:
            return HTMLResponse(content=_FALLBACK_HTML, status_code=200)


def create_dashboard_app(
    trace_store=None,
    eval_results: dict[str, Any] | None = None,
    agent=None,
) -> FastAPI:
    """Factory function to create a dashboard-enabled FastAPI app."""
    server = DashboardServer(
        trace_store=trace_store,
        eval_results=eval_results,
        agent=agent,
    )
    return server.create_app()


# Fallback HTML when Jinja2 is not installed
_FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentOps Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem; }
        .header { border-bottom: 1px solid #21262d; padding-bottom: 1rem; margin-bottom: 2rem; }
        .header h1 { font-size: 1.5rem; color: #58a6ff; }
        .header p { color: #8b949e; font-size: 0.875rem; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .card { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 1rem; }
        .card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
        .card .value { font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }
        .card .value.pass { color: #3fb950; }
        .card .value.fail { color: #f85149; }
        .card .value.info { color: #58a6ff; }
        .section { margin-top: 2rem; }
        .section h2 { font-size: 1.1rem; margin-bottom: 1rem; color: #f0f6fc; }
        table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
        th { text-align: left; padding: 0.5rem; border-bottom: 1px solid #21262d; color: #8b949e; font-weight: 500; }
        td { padding: 0.5rem; border-bottom: 1px solid #21262d; }
        tr:hover { background: #161b22; }
        .badge { display: inline-block; padding: 0.125rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 500; }
        .badge.pass { background: #033a16; color: #3fb950; }
        .badge.fail { background: #3a0316; color: #f85149; }
        .error-msg { color: #f85149; font-size: 0.8rem; }
        .ws-status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.5rem; }
        .ws-status.connected { background: #3fb950; }
        .ws-status.disconnected { background: #f85149; }
        .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .chart-container { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 1rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AgentOps Reliability Dashboard <span class="ws-status disconnected" id="ws-status"></span></h1>
        <p>Live observability for AI agent reliability — v0.10.0</p>
    </div>

    <div class="cards" id="stats-cards">
        <div class="card"><div class="label">Total Runs</div><div class="value info" id="stat-total">-</div></div>
        <div class="card"><div class="label">Verification Pass Rate</div><div class="value pass" id="stat-pass">-</div></div>
        <div class="card"><div class="label">Failure Rate</div><div class="value fail" id="stat-fail">-</div></div>
        <div class="card"><div class="label">Avg Latency</div><div class="value info" id="stat-latency">-</div></div>
    </div>

    <div class="charts">
        <div class="chart-container"><canvas id="failureChart"></canvas></div>
        <div class="chart-container"><canvas id="latencyChart"></canvas></div>
    </div>

    <div class="section">
        <h2>Recent Traces</h2>
        <table>
            <thead><tr><th>Run ID</th><th>Task</th><th>Verified</th><th>Latency</th><th>Error</th></tr></thead>
            <tbody id="traces-body"><tr><td colspan="5">Loading...</td></tr></tbody>
        </table>
    </div>

    <script>
        const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = wsProtocol + '//' + location.host + '/ws';
        let ws;
        let failureChart, latencyChart;

        function connect() {
            ws = new WebSocket(wsUrl);
            ws.onopen = () => {
                document.getElementById('ws-status').className = 'ws-status connected';
            };
            ws.onclose = () => {
                document.getElementById('ws-status').className = 'ws-status disconnected';
                setTimeout(connect, 3000);
            };
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'stats') updateStats(msg.data);
                if (msg.type === 'traces') updateTraces(msg.data);
            };
        }

        function updateStats(data) {
            document.getElementById('stat-total').textContent = data.total_runs || 0;
            document.getElementById('stat-pass').textContent = ((data.pass_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('stat-fail').textContent = ((data.failure_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('stat-latency').textContent = (data.avg_latency_ms || 0).toFixed(0) + 'ms';

            const fb = data.failure_breakdown || {};
            updateCharts(fb);
        }

        function updateTraces(data) {
            const tbody = document.getElementById('traces-body');
            if (!data.traces || data.traces.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5">No traces yet</td></tr>';
                return;
            }
            tbody.innerHTML = data.traces.map(t => `
                <tr>
                    <td><code>${t.run_id}</code></td>
                    <td>${t.task}</td>
                    <td><span class="badge ${t.verification_passed ? 'pass' : 'fail'}">${t.verification_passed ? 'PASS' : 'FAIL'}</span></td>
                    <td>${t.latency_ms.toFixed(0)}ms</td>
                    <td class="error-msg">${t.error || ''}</td>
                </tr>
            `).join('');
        }

        function updateCharts(failureBreakdown) {
            const ctx1 = document.getElementById('failureChart').getContext('2d');
            const ctx2 = document.getElementById('latencyChart').getContext('2d');

            if (failureChart) failureChart.destroy();
            if (latencyChart) latencyChart.destroy();

            failureChart = new Chart(ctx1, {
                type: 'doughnut',
                data: {
                    labels: ['Verification Failed', 'Verification Passed'],
                    datasets: [{
                        data: [failureBreakdown.verification_failed || 0, Math.max(0, (document.getElementById('stat-total').textContent || 0) - (failureBreakdown.verification_failed || 0))],
                        backgroundColor: ['#f85149', '#3fb950'],
                        borderColor: '#0d1117',
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Verification Results', color: '#c9d1d9' },
                        legend: { labels: { color: '#8b949e' } }
                    }
                }
            });

            latencyChart = new Chart(ctx2, {
                type: 'bar',
                data: {
                    labels: ['Avg Latency'],
                    datasets: [{
                        label: 'ms',
                        data: [parseFloat(document.getElementById('stat-latency').textContent) || 0],
                        backgroundColor: '#58a6ff',
                        borderColor: '#0d1117',
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: { display: true, text: 'Latency', color: '#c9d1d9' },
                        legend: { labels: { color: '#8b949e' } }
                    },
                    scales: {
                        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                        x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
                    }
                }
            });
        }

        // Initial fetch
        fetch('/api/dashboard/stats').then(r => r.json()).then(updateStats);
        fetch('/api/dashboard/traces').then(r => r.json()).then(updateTraces);

        connect();
    </script>
</body>
</html>"""
