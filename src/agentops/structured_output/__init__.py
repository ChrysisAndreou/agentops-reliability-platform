"""
Structured output and function calling quality evaluation.

Evaluates agent outputs for:
- JSON Schema adherence (type correctness, required fields, constraints)
- Function/tool call quality (right tool? right params? right format?)
- Structured output consistency across runs

These metrics are critical for production agent systems where
agents must produce machine-readable outputs and call APIs correctly.
"""

from .state import (
    SchemaValidationResult,
    SchemaValidationError,
    FunctionCallResult,
    FunctionCallError,
    StructuredOutputReport,
    StructuredOutputMetrics,
)
from .validator import (
    SchemaValidator,
    FunctionCallValidator,
    validate_json_output,
    validate_function_call,
)
from .metrics import (
    schema_adherence_score,
    function_call_correctness,
    structured_output_composite,
    compute_structured_metrics,
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
