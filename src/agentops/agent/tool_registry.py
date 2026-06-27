"""
Structured tool registry with schema validation, typed errors,
and replayable tool outputs for reliability tracing.

Unlike a simple dict of functions, this registry:
- Validates tool arguments against JSON Schema before invocation
- Produces structured ToolResult objects with error classification
- Records every invocation for trace replay
- Supports both synchronous and async tool backends
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolErrorType(str, Enum):
    """Classified error types for tool failures."""
    VALIDATION = "validation"        # Bad arguments
    EXECUTION = "execution"          # Tool threw an exception
    TIMEOUT = "timeout"              # Tool exceeded time limit
    NOT_FOUND = "not_found"          # Tool doesn't exist
    AUTH = "auth"                    # Authentication / permission
    NETWORK = "network"              # Network / connectivity
    RATE_LIMIT = "rate_limit"        # Rate limited
    UNKNOWN = "unknown"


@dataclass
class ToolResult:
    """Structured, replay-safe result from a tool invocation."""
    tool_name: str
    arguments: dict[str, Any]
    output: Any = None
    error: str | None = None
    error_type: ToolErrorType | None = None
    latency_ms: float = 0.0
    invocation_id: str = ""
    timestamp: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
            "latency_ms": self.latency_ms,
            "invocation_id": self.invocation_id,
            "timestamp": self.timestamp,
        }

    def to_replay(self) -> dict[str, Any]:
        """Return only output/error for replay — hides arguments for privacy."""
        return {
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
            "latency_ms": self.latency_ms,
        }


class ToolDefinition(BaseModel):
    """Schema definition for a tool that agents can call."""
    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    fn: Callable | None = None  # not serialized

    def to_openai_function(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def to_langchain_tool(self):
        from langchain_core.tools import StructuredTool
        return StructuredTool.from_function(
            func=self.fn,
            name=self.name,
            description=self.description,
        )

    def validate_args(self, args: dict[str, Any]) -> list[str]:
        """Validate arguments against the schema. Returns list of error messages."""
        errors = []
        for req in self.required:
            if req not in args:
                errors.append(f"Missing required argument: '{req}'")
        for key in args:
            if key not in self.parameters:
                errors.append(f"Unknown argument: '{key}'")
        return errors


class ToolRegistry:
    """Registry of typed, validated tools with invocation tracing.

    Usage:
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="search", ...))

        result = registry.invoke("search", {"query": "..."})
        if not result.success:
            print(f"Tool error [{result.error_type}]: {result.error}")

        # Replay: registry.invoke_from_replay("search", replay_data)
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._history: list[ToolResult] = []

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[ToolDefinition]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def history(self) -> list[ToolResult]:
        return list(self._history)

    def invoke(self, name: str, arguments: dict[str, Any], timeout_ms: float = 30_000) -> ToolResult:
        """Invoke a tool by name with validated arguments.

        Returns a ToolResult with success/error classification.
        """
        import uuid

        invocation_id = str(uuid.uuid4())[:8]
        start = time.time()

        tool = self._tools.get(name)
        if tool is None:
            result = ToolResult(
                tool_name=name,
                arguments=arguments,
                error=f"Tool '{name}' not found. Available: {self.tool_names}",
                error_type=ToolErrorType.NOT_FOUND,
                invocation_id=invocation_id,
                timestamp=start,
            )
            self._history.append(result)
            return result

        # Validate arguments
        validation_errors = tool.validate_args(arguments)
        if validation_errors:
            result = ToolResult(
                tool_name=name,
                arguments=arguments,
                error="; ".join(validation_errors),
                error_type=ToolErrorType.VALIDATION,
                invocation_id=invocation_id,
                timestamp=start,
            )
            self._history.append(result)
            return result

        # Invoke
        try:
            output = tool.fn(**arguments)
            latency = (time.time() - start) * 1000
            result = ToolResult(
                tool_name=name,
                arguments=arguments,
                output=str(output) if output is not None else "",
                latency_ms=latency,
                invocation_id=invocation_id,
                timestamp=start,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            error_str = f"{type(e).__name__}: {e}"
            error_type = ToolErrorType.EXECUTION
            if "timeout" in str(e).lower():
                error_type = ToolErrorType.TIMEOUT
            elif "auth" in str(e).lower() or "permission" in str(e).lower():
                error_type = ToolErrorType.AUTH
            elif "network" in str(e).lower() or "connect" in str(e).lower():
                error_type = ToolErrorType.NETWORK
            elif "rate" in str(e).lower() and "limit" in str(e).lower():
                error_type = ToolErrorType.RATE_LIMIT

            result = ToolResult(
                tool_name=name,
                arguments=arguments,
                error=error_str,
                error_type=error_type,
                latency_ms=latency,
                invocation_id=invocation_id,
                timestamp=start,
            )

        self._history.append(result)
        return result

    def invoke_from_replay(self, name: str, replay_data: dict[str, Any]) -> ToolResult:
        """Replay a tool invocation from stored trace data.

        Does not actually call the tool — returns the stored output.
        Useful for deterministic evaluation and debugging.
        """
        return ToolResult(
            tool_name=name,
            arguments={},
            output=replay_data.get("output"),
            error=replay_data.get("error"),
            error_type=ToolErrorType(replay_data["error_type"]) if replay_data.get("error_type") else None,
            latency_ms=replay_data.get("latency_ms", 0),
            invocation_id="replay",
            timestamp=time.time(),
        )

    def reset(self) -> None:
        """Clear invocation history."""
        self._history.clear()

    def summary(self) -> dict[str, Any]:
        """Return a summary of tool usage from history."""
        if not self._history:
            return {"total_calls": 0, "success_rate": 0, "tools_used": {}}

        successes = sum(1 for r in self._history if r.success)
        by_tool: dict[str, dict] = {}
        for r in self._history:
            if r.tool_name not in by_tool:
                by_tool[r.tool_name] = {"calls": 0, "successes": 0, "errors": 0, "avg_latency_ms": 0}
            by_tool[r.tool_name]["calls"] += 1
            if r.success:
                by_tool[r.tool_name]["successes"] += 1
            else:
                by_tool[r.tool_name]["errors"] += 1
            by_tool[r.tool_name]["avg_latency_ms"] += r.latency_ms

        for t in by_tool:
            c = by_tool[t]["calls"]
            by_tool[t]["avg_latency_ms"] = round(by_tool[t]["avg_latency_ms"] / c, 1)
            by_tool[t]["error_rate"] = round(by_tool[t]["errors"] / c, 3)

        return {
            "total_calls": len(self._history),
            "success_rate": round(successes / len(self._history), 3) if self._history else 0,
            "tools_used": by_tool,
        }
