"""
Structured output quality metrics.

Computes aggregate scores for schema adherence, function call correctness,
and combined composite metrics for structured output evaluation.
"""

from __future__ import annotations

from typing import Any

from .state import (
    SchemaValidationResult,
    FunctionCallResult,
    StructuredOutputMetrics,
)


def schema_adherence_score(results: list[SchemaValidationResult]) -> float:
    """Average schema adherence across all validation results (0.0-1.0)."""
    if not results:
        return 0.0
    return sum(r.adherence_score for r in results) / len(results)


def function_call_correctness(results: list[FunctionCallResult]) -> float:
    """Average function call correctness across all calls (0.0-1.0)."""
    if not results:
        return 0.0
    return sum(r.correctness_score for r in results) / len(results)


def structured_output_composite(
    schema_adherence: float,
    function_call_correctness: float,
) -> float:
    """Composite score weighting schema adherence and function call quality.

    Schema adherence: 0.50 (getting the output format right)
    Function call correctness: 0.35 (using the right tools with right params)
    Tool selection accuracy: 0.15 (bonus for choosing correct tools)
    """
    return 0.50 * schema_adherence + 0.50 * function_call_correctness


def compute_structured_metrics(
    schema_results: list[SchemaValidationResult],
    function_call_results: list[FunctionCallResult],
) -> StructuredOutputMetrics:
    """Compute aggregate structured output metrics from evaluation results.

    Args:
        schema_results: Per-task schema validation results.
        function_call_results: Per-call function call validation results.

    Returns:
        StructuredOutputMetrics with aggregate scores.
    """
    # Schema adherence
    avg_schema = schema_adherence_score(schema_results)
    total_valid = sum(1 for r in schema_results if r.is_valid)
    total_invalid = sum(1 for r in schema_results if not r.is_valid)

    # Function call quality
    avg_fc = function_call_correctness(function_call_results)
    total_correct = sum(1 for r in function_call_results if r.is_correct)
    total_incorrect = sum(1 for r in function_call_results if not r.is_correct)

    # Error breakdown
    tool_selection_errors = sum(
        1 for r in function_call_results
        for e in r.errors
        if e.error_type.value in ("wrong_tool", "hallucinated_tool")
    )
    param_errors = sum(
        1 for r in function_call_results
        for e in r.errors
        if e.error_type.value not in ("wrong_tool", "hallucinated_tool")
    )

    composite = structured_output_composite(avg_schema, avg_fc)

    return StructuredOutputMetrics(
        avg_schema_adherence=avg_schema,
        total_valid_outputs=total_valid,
        total_invalid_outputs=total_invalid,
        avg_function_call_correctness=avg_fc,
        total_correct_calls=total_correct,
        total_incorrect_calls=total_incorrect,
        total_tool_selection_errors=tool_selection_errors,
        total_param_errors=param_errors,
        composite_score=composite,
    )
