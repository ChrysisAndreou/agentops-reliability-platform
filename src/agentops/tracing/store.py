"""
SQLite-backed trace store for persisting and querying agent execution traces.

Each trace records the complete agent run: plan, retrieval steps, tool calls,
verification results, final answer, latency, and error information.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StoredTrace:
    """A trace record as stored in the database."""
    id: int
    run_id: str
    task_id: str
    task: str
    agent_type: str
    model: str
    final_answer: str
    success: bool
    error: str | None
    verification_passed: bool
    verification_notes: str
    grounded_claims: list[str]
    ungrounded_claims: list[str]
    plan: list[str]
    tool_calls_count: int
    retrieved_chunks_count: int
    total_latency_ms: float
    reliability_trace: list[dict[str, Any]]
    created_at: str


class TraceStore:
    """SQLite trace store with query and replay capabilities.

    Usage:
        store = TraceStore("traces.db")
        store.save(result)  # AgentRunResult
        traces = store.query(verification_passed=False)
        replay_data = store.get_replay(run_id)
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                task_id TEXT NOT NULL,
                task TEXT NOT NULL,
                agent_type TEXT DEFAULT 'reliability',
                model TEXT DEFAULT '',
                final_answer TEXT DEFAULT '',
                success INTEGER DEFAULT 0,
                error TEXT,
                verification_passed INTEGER DEFAULT 0,
                verification_notes TEXT DEFAULT '',
                grounded_claims TEXT DEFAULT '[]',
                ungrounded_claims TEXT DEFAULT '[]',
                plan TEXT DEFAULT '[]',
                tool_calls_count INTEGER DEFAULT 0,
                retrieved_chunks_count INTEGER DEFAULT 0,
                total_latency_ms REAL DEFAULT 0,
                reliability_trace TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_traces_run_id ON traces(run_id);
            CREATE INDEX IF NOT EXISTS idx_traces_task_id ON traces(task_id);
            CREATE INDEX IF NOT EXISTS idx_traces_verification ON traces(verification_passed);
            CREATE INDEX IF NOT EXISTS idx_traces_success ON traces(success);
            CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);

            CREATE TABLE IF NOT EXISTS eval_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_id TEXT UNIQUE NOT NULL,
                benchmark_name TEXT NOT NULL,
                total_tasks INTEGER DEFAULT 0,
                passed_verification INTEGER DEFAULT 0,
                avg_latency_ms REAL DEFAULT 0,
                avg_grounded_ratio REAL DEFAULT 0,
                failure_patterns TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_eval_runs_id ON eval_runs(eval_id);
        """)
        conn.commit()

    def save(self, result: Any) -> str:
        """Save an AgentRunResult to the trace store. Returns run_id."""
        import uuid
        run_id = result.task_id if hasattr(result, "task_id") else str(uuid.uuid4())[:8]

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO traces
               (run_id, task_id, task, agent_type, model, final_answer, success,
                error, verification_passed, verification_notes,
                grounded_claims, ungrounded_claims, plan,
                tool_calls_count, retrieved_chunks_count, total_latency_ms,
                reliability_trace)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                getattr(result, "task_id", ""),
                getattr(result, "task", ""),
                getattr(result, "agent_type", "reliability") if hasattr(result, "agent_type") else "reliability",
                "",
                getattr(result, "final_answer", ""),
                1 if getattr(result, "success", False) else 0,
                getattr(result, "error", None),
                1 if getattr(result, "verification_passed", False) else 0,
                getattr(result, "verification_notes", ""),
                json.dumps(getattr(result, "grounded_claims", [])),
                json.dumps(getattr(result, "ungrounded_claims", [])),
                json.dumps(getattr(result, "plan", [])),
                getattr(result, "tool_calls_count", 0),
                getattr(result, "retrieved_chunks_count", 0),
                getattr(result, "total_latency_ms", 0),
                json.dumps(getattr(result, "reliability_trace", [])),
            ),
        )
        conn.commit()
        return run_id

    def save_eval_run(self, eval_id: str, benchmark_name: str,
                      results: list[Any], failure_patterns: list[dict]) -> None:
        """Save a complete evaluation run summary."""
        if not results:
            return

        total = len(results)
        passed = sum(1 for r in results if getattr(r, "verification_passed", False))
        avg_latency = sum(getattr(r, "total_latency_ms", 0) for r in results) / max(total, 1)
        grounded_counts = []
        for r in results:
            g = len(getattr(r, "grounded_claims", []))
            u = len(getattr(r, "ungrounded_claims", []))
            grounded_counts.append(g / max(g + u, 1) if (g + u) > 0 else 0)
        avg_grounded = sum(grounded_counts) / max(len(grounded_counts), 1)

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO eval_runs
               (eval_id, benchmark_name, total_tasks, passed_verification,
                avg_latency_ms, avg_grounded_ratio, failure_patterns)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eval_id, benchmark_name, total, passed, avg_latency,
             round(avg_grounded, 3), json.dumps(failure_patterns)),
        )
        conn.commit()

    def query(
        self,
        verification_passed: bool | None = None,
        success: bool | None = None,
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[StoredTrace]:
        """Query traces with optional filters."""
        conn = self._get_conn()
        conditions = []
        params: list[Any] = []

        if verification_passed is not None:
            conditions.append("verification_passed = ?")
            params.append(1 if verification_passed else 0)
        if success is not None:
            conditions.append("success = ?")
            params.append(1 if success else 0)
        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM traces {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_trace(r) for r in rows]

    def get_replay(self, run_id: str) -> dict[str, Any] | None:
        """Get replay data for a specific run."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM traces WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None

        return {
            "run_id": row["run_id"],
            "task": row["task"],
            "plan": json.loads(row["plan"]),
            "reliability_trace": json.loads(row["reliability_trace"]),
            "final_answer": row["final_answer"],
            "verification_passed": bool(row["verification_passed"]),
            "tool_calls_count": row["tool_calls_count"],
        }

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics for all traces."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        if total == 0:
            return {"total_runs": 0}

        passed = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE verification_passed = 1"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE success = 0"
        ).fetchone()[0]
        avg_latency = conn.execute(
            "SELECT AVG(total_latency_ms) FROM traces"
        ).fetchone()[0] or 0

        return {
            "total_runs": total,
            "verification_pass_rate": round(passed / total, 3),
            "failure_rate": round(failed / total, 3),
            "avg_latency_ms": round(avg_latency, 1),
        }

    def _row_to_trace(self, row: sqlite3.Row) -> StoredTrace:
        return StoredTrace(
            id=row["id"],
            run_id=row["run_id"],
            task_id=row["task_id"],
            task=row["task"],
            agent_type=row["agent_type"],
            model=row["model"],
            final_answer=row["final_answer"],
            success=bool(row["success"]),
            error=row["error"],
            verification_passed=bool(row["verification_passed"]),
            verification_notes=row["verification_notes"],
            grounded_claims=json.loads(row["grounded_claims"]),
            ungrounded_claims=json.loads(row["ungrounded_claims"]),
            plan=json.loads(row["plan"]),
            tool_calls_count=row["tool_calls_count"],
            retrieved_chunks_count=row["retrieved_chunks_count"],
            total_latency_ms=row["total_latency_ms"],
            reliability_trace=json.loads(row["reliability_trace"]),
            created_at=row["created_at"],
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
