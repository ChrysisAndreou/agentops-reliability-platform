"""
Type definitions for structured output evaluation.

Models JSON Schema validation results, function call quality scores,
and aggregate structured output reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SchemaValidationErrorType(str, Enum):
    """Classified error types for JSON schema validation failures."""

    MISSING_REQUIRED = "missing_required"
    WRONG_TYPE = "wrong_type"
    INVALID_ENUM = "invalid_enum"
    PATTERN_MISMATCH = "pattern_mismatch"
    OUT_OF_RANGE = "out_of_range"
    EXTRA_FIELD = "extra_field"
    MALFORMED_JSON = "malformed_json"
    NOT_JSON = "not_json"


class FunctionCallErrorType(str, Enum):
    """Classified error types for function/tool call failures."""

    WRONG_TOOL = "wrong_tool"           # Called a tool that doesn't exist or wrong one
    MISSING_PARAM = "missing_param"     # Required parameter not provided
    WRONG_PARAM_TYPE = "wrong_param_type"  # Parameter has wrong type
    INVALID_PARAM_VALUE = "invalid_param_value"  # Parameter value fails constraints
    EXTRA_PARAM = "extra_param"         # Unknown parameter provided
    MALFORMED_CALL = "malformed_call"   # Call doesn't parse as valid function call
    HALLUCINATED_TOOL = "hallucinated_tool"  # Made up a tool that doesn't exist


@dataclass
class JSONSchemaField:
    """A single field definition from a JSON Schema."""

    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    required: bool = False
    description: str = ""
    enum_values: list[Any] | None = None
    pattern: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    items_type: str | None = None  # For array fields


@dataclass
class JSONSchema:
    """A JSON Schema definition for structured output validation.

    Supports a practical subset of JSON Schema Draft-07:
    - type checking (string, number, integer, boolean, array, object)
    - required fields
    - enum values
    - string patterns (regex)
    - numeric min/max
    - string minLength/maxLength
    - nested object fields
    """

    name: str
    description: str = ""
    fields: list[JSONSchemaField] = field(default_factory=list)
    additional_properties: bool = True  # If False, extra fields are errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON Schema-like dict."""
        properties = {}
        required = []
        for f in self.fields:
            prop: dict[str, Any] = {"type": f.type}
            if f.description:
                prop["description"] = f.description
            if f.enum_values is not None:
                prop["enum"] = f.enum_values
            if f.pattern:
                prop["pattern"] = f.pattern
            if f.minimum is not None:
                prop["minimum"] = f.minimum
            if f.maximum is not None:
                prop["maximum"] = f.maximum
            if f.min_length is not None:
                prop["minLength"] = f.min_length
            if f.max_length is not None:
                prop["maxLength"] = f.max_length
            if f.type == "array" and f.items_type:
                prop["items"] = {"type": f.items_type}
            if f.required:
                required.append(f.name)
            properties[f.name] = prop

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        if not self.additional_properties:
            schema["additionalProperties"] = False
        if self.description:
            schema["description"] = self.description

        return schema

    @classmethod
    def from_dict(cls, name: str, schema_dict: dict[str, Any]) -> JSONSchema:
        """Parse a simplified JSON Schema dict into field definitions."""
        fields = []
        properties = schema_dict.get("properties", {})
        required_list = schema_dict.get("required", [])
        additional = schema_dict.get("additionalProperties", True)

        for field_name, field_def in properties.items():
            fields.append(JSONSchemaField(
                name=field_name,
                type=field_def.get("type", "string"),
                required=field_name in required_list,
                description=field_def.get("description", ""),
                enum_values=field_def.get("enum"),
                pattern=field_def.get("pattern"),
                minimum=field_def.get("minimum"),
                maximum=field_def.get("maximum"),
                min_length=field_def.get("minLength"),
                max_length=field_def.get("maxLength"),
                items_type=field_def.get("items", {}).get("type") if field_def.get("type") == "array" else None,
            ))

        return cls(
            name=name,
            description=schema_dict.get("description", ""),
            fields=fields,
            additional_properties=additional if isinstance(additional, bool) else True,
        )


@dataclass
class SchemaValidationError:
    """A single validation error found in a structured output."""

    field: str  # Which field failed ("<root>" for top-level errors)
    error_type: SchemaValidationErrorType
    message: str
    expected: Any | None = None
    actual: Any | None = None


@dataclass
class SchemaValidationResult:
    """Result of validating an agent output against a JSON Schema."""

    schema_name: str
    is_valid: bool
    errors: list[SchemaValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_output: str = ""
    parsed_output: dict[str, Any] | None = None
    fields_total: int = 0
    fields_valid: int = 0
    fields_invalid: int = 0
    fields_missing: int = 0

    @property
    def adherence_score(self) -> float:
        """Schema adherence: fraction of fields that are valid (0.0-1.0)."""
        if self.fields_total == 0:
            return 1.0 if self.is_valid else 0.0
        return self.fields_valid / self.fields_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "is_valid": self.is_valid,
            "adherence_score": round(self.adherence_score, 3),
            "fields_total": self.fields_total,
            "fields_valid": self.fields_valid,
            "fields_invalid": self.fields_invalid,
            "fields_missing": self.fields_missing,
            "errors": [
                {
                    "field": e.field,
                    "error_type": e.error_type.value,
                    "message": e.message,
                    "expected": e.expected,
                    "actual": e.actual,
                }
                for e in self.errors
            ],
            "warnings": self.warnings,
        }


@dataclass
class FunctionCallError:
    """A single error in a function/tool call."""

    error_type: FunctionCallErrorType
    message: str
    tool_name: str = ""
    param_name: str = ""
    expected: Any | None = None
    actual: Any | None = None


@dataclass
class FunctionCallResult:
    """Result of evaluating a single function/tool call."""

    call_id: str
    expected_tool: str
    actual_tool: str
    is_correct: bool
    errors: list[FunctionCallError] = field(default_factory=list)
    params_total: int = 0
    params_correct: int = 0
    params_incorrect: int = 0
    params_missing: int = 0
    raw_call: dict[str, Any] | None = None

    @property
    def correctness_score(self) -> float:
        """Function call correctness: fraction of params correct (0.0-1.0)."""
        if not self.is_correct and self.params_total == 0:
            return 1.0 if self.params_total == 0 and not self.errors else 0.0
        if self.params_total == 0:
            return 1.0
        return self.params_correct / max(self.params_total, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "expected_tool": self.expected_tool,
            "actual_tool": self.actual_tool,
            "is_correct": self.is_correct,
            "correctness_score": round(self.correctness_score, 3),
            "params_total": self.params_total,
            "params_correct": self.params_correct,
            "params_incorrect": self.params_incorrect,
            "params_missing": self.params_missing,
            "errors": [
                {
                    "error_type": e.error_type.value,
                    "message": e.message,
                    "tool_name": e.tool_name,
                    "param_name": e.param_name,
                }
                for e in self.errors
            ],
        }


@dataclass
class StructuredOutputMetrics:
    """Aggregate metrics for structured output quality across tasks."""

    # Schema adherence
    avg_schema_adherence: float = 0.0
    total_valid_outputs: int = 0
    total_invalid_outputs: int = 0

    # Function call quality
    avg_function_call_correctness: float = 0.0
    total_correct_calls: int = 0
    total_incorrect_calls: int = 0
    total_tool_selection_errors: int = 0
    total_param_errors: int = 0

    # Composite
    composite_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_schema_adherence": round(self.avg_schema_adherence, 3),
            "total_valid_outputs": self.total_valid_outputs,
            "total_invalid_outputs": self.total_invalid_outputs,
            "avg_function_call_correctness": round(self.avg_function_call_correctness, 3),
            "total_correct_calls": self.total_correct_calls,
            "total_incorrect_calls": self.total_incorrect_calls,
            "total_tool_selection_errors": self.total_tool_selection_errors,
            "total_param_errors": self.total_param_errors,
            "composite_score": round(self.composite_score, 3),
        }


@dataclass
class StructuredOutputReport:
    """Complete structured output evaluation report."""

    benchmark_name: str
    schema_results: list[SchemaValidationResult] = field(default_factory=list)
    function_call_results: list[FunctionCallResult] = field(default_factory=list)
    metrics: StructuredOutputMetrics = field(default_factory=StructuredOutputMetrics)
    generated_at: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# Structured Output Evaluation: {self.benchmark_name}",
            "",
            f"**Generated:** {self.generated_at}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Schema Adherence (avg) | {self.metrics.avg_schema_adherence:.3f} |",
            f"| Valid / Invalid Outputs | {self.metrics.total_valid_outputs} / {self.metrics.total_invalid_outputs} |",
            f"| Function Call Correctness (avg) | {self.metrics.avg_function_call_correctness:.3f} |",
            f"| Correct / Incorrect Calls | {self.metrics.total_correct_calls} / {self.metrics.total_incorrect_calls} |",
            f"| Tool Selection Errors | {self.metrics.total_tool_selection_errors} |",
            f"| Parameter Errors | {self.metrics.total_param_errors} |",
            f"| **Composite Score** | **{self.metrics.composite_score:.3f}** |",
            "",
        ]

        if self.schema_results:
            lines.append("## Schema Validation Results")
            lines.append("")
            lines.append("| Task | Schema | Adherence | Valid | Fields (OK/Total) | Errors |")
            lines.append("|------|--------|-----------|-------|-------------------|--------|")
            for r in self.schema_results:
                status = "✓" if r.is_valid else "✗"
                lines.append(
                    f"| {status} | {r.schema_name} | {r.adherence_score:.3f} | "
                    f"{'PASS' if r.is_valid else 'FAIL'} | "
                    f"{r.fields_valid}/{r.fields_total} | "
                    f"{len(r.errors)} |"
                )
            lines.append("")

            # Detail errors
            for r in self.schema_results:
                if r.errors:
                    lines.append(f"### Schema: {r.schema_name}")
                    for e in r.errors:
                        lines.append(f"- **{e.field}** [{e.error_type.value}]: {e.message}")

        if self.function_call_results:
            lines.append("## Function Call Results")
            lines.append("")
            lines.append("| Call ID | Expected → Actual | Correctness | Params (OK/Total) | Errors |")
            lines.append("|---------|-------------------|-------------|--------------------|--------|")
            for r in self.function_call_results:
                status = "✓" if r.is_correct else "✗"
                lines.append(
                    f"| {status} | {r.expected_tool} → {r.actual_tool} | "
                    f"{r.correctness_score:.3f} | "
                    f"{r.params_correct}/{r.params_total} | "
                    f"{len(r.errors)} |"
                )
            lines.append("")

            # Detail errors
            for r in self.function_call_results:
                if r.errors:
                    lines.append(f"### Call: {r.call_id}")
                    for e in r.errors:
                        lines.append(f"- **{e.error_type.value}**: {e.message}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "metrics": self.metrics.to_dict(),
            "schema_results": [r.to_dict() for r in self.schema_results],
            "function_call_results": [r.to_dict() for r in self.function_call_results],
            "generated_at": self.generated_at,
        }
