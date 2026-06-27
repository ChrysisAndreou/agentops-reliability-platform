"""
Schema and function call validators for structured agent outputs.

Validates agent outputs against JSON Schemas and function call specifications,
producing detailed error reports for each failure.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .state import (
    FunctionCallError,
    FunctionCallErrorType,
    FunctionCallResult,
    JSONSchema,
    JSONSchemaField,
    SchemaValidationError,
    SchemaValidationErrorType,
    SchemaValidationResult,
)


class SchemaValidator:
    """Validates agent outputs against a JSON Schema.

    Supports a practical subset of JSON Schema Draft-07:
    - Type checking (string, number, integer, boolean, array)
    - Required fields
    - Enum value constraints
    - Regex pattern matching
    - Numeric min/max bounds
    - String length bounds
    - Extra field detection

    Usage:
        schema = JSONSchema(name="ticket_response", fields=[...])
        validator = SchemaValidator(schema)
        result = validator.validate(agent_output_string)
        if not result.is_valid:
            for err in result.errors:
                print(f"  {err.field}: {err.message}")
    """

    def __init__(self, schema: JSONSchema):
        self.schema = schema

    def validate(self, raw_output: str) -> SchemaValidationResult:
        """Validate a raw string output against the schema.

        Attempts to parse JSON first, then validates each field.
        """
        errors: list[SchemaValidationError] = []
        warnings: list[str] = []
        parsed: dict[str, Any] | None = None

        # Step 1: Parse JSON
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            return SchemaValidationResult(
                schema_name=self.schema.name,
                is_valid=False,
                errors=[
                    SchemaValidationError(
                        field="<root>",
                        error_type=SchemaValidationErrorType.MALFORMED_JSON,
                        message=f"Failed to parse JSON: {e.msg} at position {e.pos}",
                        actual=raw_output[:200],
                    )
                ],
                raw_output=raw_output,
                fields_total=len(self.schema.fields),
                fields_valid=0,
                fields_invalid=len(self.schema.fields),
                fields_missing=0,
            )

        if not isinstance(parsed, dict):
            return SchemaValidationResult(
                schema_name=self.schema.name,
                is_valid=False,
                errors=[
                    SchemaValidationError(
                        field="<root>",
                        error_type=SchemaValidationErrorType.NOT_JSON,
                        message=f"Output is not a JSON object (got {type(parsed).__name__})",
                        actual=str(parsed)[:200],
                    )
                ],
                raw_output=raw_output,
                parsed_output=parsed,
                fields_total=len(self.schema.fields),
                fields_valid=0,
                fields_invalid=len(self.schema.fields),
                fields_missing=0,
            )

        fields_total = len(self.schema.fields)
        fields_valid = 0
        fields_invalid = 0
        fields_missing = 0

        # Step 2: Check required fields
        present_fields = set(parsed.keys())
        for field in self.schema.fields:
            if field.required and field.name not in present_fields:
                errors.append(SchemaValidationError(
                    field=field.name,
                    error_type=SchemaValidationErrorType.MISSING_REQUIRED,
                    message=f"Required field '{field.name}' is missing",
                ))
                fields_missing += 1

        # Step 3: Validate each present field
        for field in self.schema.fields:
            if field.name not in parsed:
                if not field.required:
                    fields_valid += 1  # Optional field, not present = OK
                continue

            value = parsed[field.name]
            field_errors = self._validate_field(field, value)
            if field_errors:
                errors.extend(field_errors)
                fields_invalid += 1
            else:
                fields_valid += 1

        # Step 4: Check for extra fields (if additionalProperties is false)
        if not self.schema.additional_properties:
            defined_fields = {f.name for f in self.schema.fields}
            for key in present_fields:
                if key not in defined_fields:
                    errors.append(SchemaValidationError(
                        field=key,
                        error_type=SchemaValidationErrorType.EXTRA_FIELD,
                        message=f"Unexpected field '{key}' (additional properties not allowed)",
                    ))
                    fields_invalid += 1

        return SchemaValidationResult(
            schema_name=self.schema.name,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            raw_output=raw_output,
            parsed_output=parsed,
            fields_total=fields_total,
            fields_valid=fields_valid,
            fields_invalid=fields_invalid,
            fields_missing=fields_missing,
        )

    def _validate_field(self, field: JSONSchemaField, value: Any) -> list[SchemaValidationError]:
        """Validate a single field value against its schema definition."""
        errors: list[SchemaValidationError] = []

        # Null handling — null is never valid for typed fields
        if value is None:
            errors.append(SchemaValidationError(
                field=field.name,
                error_type=SchemaValidationErrorType.WRONG_TYPE,
                message=f"Field '{field.name}' is null (expected {field.type})",
                expected=field.type,
                actual="null",
            ))
            return errors

        # Type checking
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        type_check = type_checks.get(field.type)
        if type_check and not type_check(value):
            errors.append(SchemaValidationError(
                field=field.name,
                error_type=SchemaValidationErrorType.WRONG_TYPE,
                message=f"Field '{field.name}' has wrong type (expected {field.type}, got {type(value).__name__})",
                expected=field.type,
                actual=type(value).__name__,
            ))
            return errors

        # Enum validation
        if field.enum_values is not None and value not in field.enum_values:
            errors.append(SchemaValidationError(
                field=field.name,
                error_type=SchemaValidationErrorType.INVALID_ENUM,
                message=f"Field '{field.name}' value '{value}' is not in allowed enum values: {field.enum_values}",
                expected=field.enum_values,
                actual=value,
            ))

        # String constraints
        if field.type == "string" and isinstance(value, str):
            if field.pattern is not None:
                try:
                    if not re.match(field.pattern, value):
                        errors.append(SchemaValidationError(
                            field=field.name,
                            error_type=SchemaValidationErrorType.PATTERN_MISMATCH,
                            message=f"Field '{field.name}' does not match pattern '{field.pattern}'",
                            expected=f"matches /{field.pattern}/",
                            actual=value[:100],
                        ))
                except re.error:
                    pass  # Invalid regex in schema — skip

            if field.min_length is not None and len(value) < field.min_length:
                errors.append(SchemaValidationError(
                    field=field.name,
                    error_type=SchemaValidationErrorType.OUT_OF_RANGE,
                    message=f"Field '{field.name}' is too short ({len(value)} < min {field.min_length})",
                    expected=f"length >= {field.min_length}",
                    actual=len(value),
                ))

            if field.max_length is not None and len(value) > field.max_length:
                errors.append(SchemaValidationError(
                    field=field.name,
                    error_type=SchemaValidationErrorType.OUT_OF_RANGE,
                    message=f"Field '{field.name}' is too long ({len(value)} > max {field.max_length})",
                    expected=f"length <= {field.max_length}",
                    actual=len(value),
                ))

        # Numeric constraints
        if field.type in ("number", "integer") and isinstance(value, (int, float)):
            if field.minimum is not None and value < field.minimum:
                errors.append(SchemaValidationError(
                    field=field.name,
                    error_type=SchemaValidationErrorType.OUT_OF_RANGE,
                    message=f"Field '{field.name}' value {value} is below minimum {field.minimum}",
                    expected=f">= {field.minimum}",
                    actual=value,
                ))

            if field.maximum is not None and value > field.maximum:
                errors.append(SchemaValidationError(
                    field=field.name,
                    error_type=SchemaValidationErrorType.OUT_OF_RANGE,
                    message=f"Field '{field.name}' value {value} exceeds maximum {field.maximum}",
                    expected=f"<= {field.maximum}",
                    actual=value,
                ))

        # Array item type checking
        if field.type == "array" and isinstance(value, list) and field.items_type:
            item_type_check = type_checks.get(field.items_type)
            if item_type_check:
                for i, item in enumerate(value):
                    if not item_type_check(item):
                        errors.append(SchemaValidationError(
                            field=f"{field.name}[{i}]",
                            error_type=SchemaValidationErrorType.WRONG_TYPE,
                            message=f"Array item at index {i} has wrong type (expected {field.items_type}, got {type(item).__name__})",
                            expected=field.items_type,
                            actual=type(item).__name__,
                        ))

        return errors


class FunctionCallValidator:
    """Validates agent function/tool calls for correctness.

    Checks whether the agent called the right tool with the right parameters
    in the right format. Produces detailed error categorization.

    Usage:
        tools = {
            "search_kb": {"query": "string", "limit": "integer"},
            "create_ticket": {"title": "string", "priority": "string", "assignee": "string"},
        }
        validator = FunctionCallValidator(tools)
        result = validator.validate(
            call_id="call-1",
            expected_tool="search_kb",
            expected_params={"query": "deployment failure"},
            actual_call={"tool": "search_kb", "params": {"query": "deployment failure"}},
        )
    """

    def __init__(self, tool_schemas: dict[str, dict[str, str]] | None = None):
        """Initialize with known tool schemas.

        Args:
            tool_schemas: Dict mapping tool_name -> {param_name: param_type}.
                         If None, only validates structural correctness.
        """
        self.tool_schemas = tool_schemas or {}

    def validate(
        self,
        call_id: str,
        expected_tool: str,
        expected_params: dict[str, Any],
        actual_call: dict[str, Any],
    ) -> FunctionCallResult:
        """Validate a function call against expectations.

        Args:
            call_id: Unique identifier for this call.
            expected_tool: The tool that SHOULD have been called.
            expected_params: The parameters that SHOULD have been passed.
            actual_call: The actual call made: {"tool": "...", "params": {...}}.

        Returns:
            FunctionCallResult with correctness assessment and detailed errors.
        """
        errors: list[FunctionCallError] = []

        # Extract actual tool and params
        actual_tool = actual_call.get("tool", actual_call.get("name", ""))
        actual_params = actual_call.get("params", actual_call.get("arguments", {}))

        if not isinstance(actual_params, dict):
            actual_params = {}

        params_total = len(expected_params)
        params_correct = 0
        params_incorrect = 0
        params_missing = 0

        # Check 1: Tool selection
        if actual_tool != expected_tool:
            if actual_tool and actual_tool not in self.tool_schemas:
                errors.append(FunctionCallError(
                    error_type=FunctionCallErrorType.HALLUCINATED_TOOL,
                    message=f"Agent called hallucinated tool '{actual_tool}' (not in available tools: {list(self.tool_schemas.keys())})",
                    tool_name=actual_tool,
                ))
            else:
                errors.append(FunctionCallError(
                    error_type=FunctionCallErrorType.WRONG_TOOL,
                    message=f"Agent called '{actual_tool}' but expected '{expected_tool}'",
                    tool_name=actual_tool,
                    expected=expected_tool,
                    actual=actual_tool,
                ))
            # All params are wrong since wrong tool
            params_incorrect = params_total
            return FunctionCallResult(
                call_id=call_id,
                expected_tool=expected_tool,
                actual_tool=actual_tool,
                is_correct=False,
                errors=errors,
                params_total=params_total,
                params_correct=0,
                params_incorrect=params_incorrect,
                params_missing=params_missing,
                raw_call=actual_call,
            )

        # Check 2: Parameter validation
        expected_schemas = self.tool_schemas.get(expected_tool, {})

        for param_name, expected_value in expected_params.items():
            if param_name not in actual_params:
                errors.append(FunctionCallError(
                    error_type=FunctionCallErrorType.MISSING_PARAM,
                    message=f"Missing required parameter '{param_name}' for tool '{expected_tool}'",
                    tool_name=expected_tool,
                    param_name=param_name,
                    expected=expected_value,
                ))
                params_missing += 1
                continue

            actual_value = actual_params[param_name]
            expected_type = expected_schemas.get(param_name, "string")

            # Type checking
            type_valid, type_msg = self._check_param_type(expected_type, actual_value, expected_value)
            if not type_valid:
                errors.append(FunctionCallError(
                    error_type=FunctionCallErrorType.WRONG_PARAM_TYPE,
                    message=f"Parameter '{param_name}': {type_msg}",
                    tool_name=expected_tool,
                    param_name=param_name,
                    expected=f"{expected_type}: {expected_value}",
                    actual=f"{type(actual_value).__name__}: {actual_value}",
                ))
                params_incorrect += 1
                continue

            # Value comparison (approximate for strings, exact for others)
            if not self._values_match(expected_value, actual_value, expected_type):
                errors.append(FunctionCallError(
                    error_type=FunctionCallErrorType.INVALID_PARAM_VALUE,
                    message=f"Parameter '{param_name}' has wrong value (expected '{expected_value}', got '{actual_value}')",
                    tool_name=expected_tool,
                    param_name=param_name,
                    expected=expected_value,
                    actual=actual_value,
                ))
                params_incorrect += 1
            else:
                params_correct += 1

        # Check 3: Extra parameters (not in expected set)
        for param_name in actual_params:
            if param_name not in expected_params:
                errors.append(FunctionCallError(
                    error_type=FunctionCallErrorType.EXTRA_PARAM,
                    message=f"Unexpected parameter '{param_name}' provided to tool '{expected_tool}'",
                    tool_name=expected_tool,
                    param_name=param_name,
                ))
                params_incorrect += 1

        return FunctionCallResult(
            call_id=call_id,
            expected_tool=expected_tool,
            actual_tool=actual_tool,
            is_correct=len(errors) == 0,
            errors=errors,
            params_total=params_total,
            params_correct=params_correct,
            params_incorrect=params_incorrect,
            params_missing=params_missing,
            raw_call=actual_call,
        )

    def _check_param_type(
        self, expected_type: str, actual_value: Any, expected_value: Any
    ) -> tuple[bool, str]:
        """Check if actual_value matches expected_type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_py_type = type_map.get(expected_type, str)

        if isinstance(expected_py_type, tuple):
            if not isinstance(actual_value, expected_py_type) or isinstance(actual_value, bool):
                return False, f"expected {expected_type}, got {type(actual_value).__name__}"
        else:
            if not isinstance(actual_value, expected_py_type):
                return False, f"expected {expected_type}, got {type(actual_value).__name__}"

        return True, ""

    def _values_match(self, expected: Any, actual: Any, param_type: str) -> bool:
        """Check if actual value matches expected value."""
        if param_type == "string":
            # Case-insensitive substring or exact match
            expected_str = str(expected).lower().strip()
            actual_str = str(actual).lower().strip()
            return expected_str == actual_str or expected_str in actual_str
        elif param_type in ("number", "integer"):
            try:
                return abs(float(expected) - float(actual)) < 1e-9
            except (ValueError, TypeError):
                return False
        elif param_type == "boolean":
            return bool(expected) == bool(actual)
        else:
            return expected == actual


def validate_json_output(raw_output: str, schema: JSONSchema) -> SchemaValidationResult:
    """Convenience: validate a raw JSON string against a schema."""
    validator = SchemaValidator(schema)
    return validator.validate(raw_output)


def validate_function_call(
    expected_tool: str,
    expected_params: dict[str, Any],
    actual_call: dict[str, Any],
    tool_schemas: dict[str, dict[str, str]] | None = None,
    call_id: str = "call",
) -> FunctionCallResult:
    """Convenience: validate a single function call."""
    validator = FunctionCallValidator(tool_schemas)
    return validator.validate(
        call_id=call_id,
        expected_tool=expected_tool,
        expected_params=expected_params,
        actual_call=actual_call,
    )
