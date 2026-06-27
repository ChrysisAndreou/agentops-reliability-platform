"""
Structured output and function calling quality evaluation.

Evaluates agent outputs for:
- JSON Schema adherence (type correctness, required fields, constraints)
- Function/tool call quality (right tool? right params? right format?)
- Structured output consistency across runs

These metrics are critical for production agent systems where
agents must produce machine-readable outputs and call APIs correctly.
"""

from .metrics import (
    compute_structured_metrics,
    function_call_correctness,
    schema_adherence_score,
    structured_output_composite,
)
from .state import (
    FunctionCallError,
    FunctionCallResult,
    SchemaValidationError,
    SchemaValidationResult,
    StructuredOutputMetrics,
    StructuredOutputReport,
)
from .validator import (
    FunctionCallValidator,
    SchemaValidator,
    validate_function_call,
    validate_json_output,
)

__all__ = [
    # State
    "SchemaValidationResult",
    "SchemaValidationError",
    "FunctionCallResult",
    "FunctionCallError",
    "StructuredOutputReport",
    "StructuredOutputMetrics",
    # Validator
    "SchemaValidator",
    "FunctionCallValidator",
    "validate_json_output",
    "validate_function_call",
    # Metrics
    "schema_adherence_score",
    "function_call_correctness",
    "structured_output_composite",
    "compute_structured_metrics",
]
