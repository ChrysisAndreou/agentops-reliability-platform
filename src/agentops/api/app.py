"""
FastAPI application factory for the AgentOps API.

Endpoints:
  GET  /health                 — health check
  GET  /api/traces             — list trace summaries
  GET  /api/traces/{run_id}    — get full trace detail
  GET  /api/traces/{run_id}/replay — get replay data
  POST /api/run                — run an agent on a task
  GET  /api/evals              — list evaluation runs
  GET  /api/evals/{eval_id}    — get evaluation report
  GET  /api/stats              — aggregate statistics
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def create_app(
    agent=None,
    trace_store=None,
    eval_results: dict[str, Any] | None = None,
) -> FastAPI:
    """Create a FastAPI app wired to a reliability agent and trace store.

    Args:
        agent: ReliabilityAgent instance (optional — POST /api/run requires it).
        trace_store: TraceStore instance for querying traces.
        eval_results: Dict of cached evaluation results.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="AgentOps Reliability Platform",
        description="API for running, tracing, and evaluating tool-using AI agents",
        version="0.1.0",
    )

    _agent = agent
    _trace_store = trace_store
    _eval_results = eval_results or {}

    class RunRequest(BaseModel):
        task: str = Field(..., description="The task/question for the agent")
        task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
        context: str = Field(default="")
        model_config = {"json_schema_extra": {"example": {"task": "How do I enable 2FA?", "context": "CloudDeploy platform"}}}

    class RunResponse(BaseModel):
        run_id: str
        task: str
        success: bool
        verification_passed: bool
        final_answer: str
        grounded_claims: int
        ungrounded_claims: int
        citations_used: int
        latency_ms: float
        error: str | None = None
        model_config = {"json_schema_extra": {"example": {
            "run_id": "abc123", "task": "How do I enable 2FA?", "success": True,
            "verification_passed": True, "final_answer": "To enable 2FA...",
            "grounded_claims": 3, "ungrounded_claims": 0, "citations_used": 2,
            "latency_ms": 4521.0
        }}}

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": time.time()}

    @app.get("/api/stats")
    async def stats():
        if _trace_store:
            return _trace_store.stats()
        return {"error": "No trace store configured"}

    @app.get("/api/traces")
    async def list_traces(verification_passed: bool | None = None, limit: int = 50):
        if not _trace_store:
            raise HTTPException(status_code=503, detail="No trace store configured")
        traces = _trace_store.query(verification_passed=verification_passed, limit=limit)
        return {
            "count": len(traces),
            "traces": [
                {
                    "run_id": t.run_id,
                    "task_id": t.task_id,
                    "task": t.task[:100],
                    "verification_passed": t.verification_passed,
                    "success": t.success,
                    "tool_calls": t.tool_calls_count,
                    "latency_ms": t.total_latency_ms,
                }
                for t in traces
            ],
        }

    @app.get("/api/traces/{run_id}")
    async def get_trace(run_id: str):
        if not _trace_store:
            raise HTTPException(status_code=503, detail="No trace store configured")
        traces = _trace_store.query(limit=1)
        matching = [t for t in traces if t.run_id == run_id]
        if not matching:
            # Try query by task_id
            traces = _trace_store.query(task_id=run_id, limit=1)
            if not traces:
                raise HTTPException(status_code=404, detail=f"Trace '{run_id}' not found")
            t = traces[0]
        else:
            t = matching[0]

        return {
            "run_id": t.run_id,
            "task_id": t.task_id,
            "task": t.task,
            "final_answer": t.final_answer,
            "success": t.success,
            "error": t.error,
            "verification_passed": t.verification_passed,
            "verification_notes": t.verification_notes,
            "grounded_claims": t.grounded_claims,
            "ungrounded_claims": t.ungrounded_claims,
            "plan": t.plan,
            "tool_calls_count": t.tool_calls_count,
            "retrieved_chunks_count": t.retrieved_chunks_count,
            "total_latency_ms": t.total_latency_ms,
            "reliability_trace": t.reliability_trace,
        }

    @app.get("/api/traces/{run_id}/replay")
    async def replay_trace(run_id: str):
        if not _trace_store:
            raise HTTPException(status_code=503, detail="No trace store configured")
        replay_data = _trace_store.get_replay(run_id)
        if replay_data is None:
            raise HTTPException(status_code=404, detail=f"Trace '{run_id}' not found")
        return replay_data

    @app.post("/api/run", response_model=RunResponse)
    async def run_agent(req: RunRequest):
        if not _agent:
            raise HTTPException(status_code=503, detail="No agent configured")
        result = await _agent.run(task=req.task, task_id=req.task_id, context=req.context)
        if _trace_store:
            _trace_store.save(result)

        return RunResponse(
            run_id=result.task_id,
            task=result.task,
            success=result.success,
            verification_passed=result.verification_passed,
            final_answer=result.final_answer,
            grounded_claims=len(result.grounded_claims),
            ungrounded_claims=len(result.ungrounded_claims),
            citations_used=len(result.citations_used),
            latency_ms=result.total_latency_ms,
            error=result.error,
        )

    @app.get("/api/evals")
    async def list_evals():
        return {"eval_runs": list(_eval_results.keys()), "count": len(_eval_results)}

    @app.get("/api/evals/{eval_id}")
    async def get_eval(eval_id: str):
        if eval_id not in _eval_results:
            raise HTTPException(status_code=404, detail=f"Eval '{eval_id}' not found")
        return _eval_results[eval_id]

    return app
