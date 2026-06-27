"""
Tests for structured output evaluation — schema validation,
function call quality, metrics, and benchmark integration.

Covers the full structured_output module: state models,
SchemaValidator, FunctionCallValidator, metrics computation,
and CLI integration.
"""

from __future__ import annotations

import json
import pytest

from agentops.structured_output.state import (
    JSONSchema,
    JSONSchemaField,
    SchemaValidationResult,
    SchemaValidationError,
    SchemaValidationErrorType,
    FunctionCallResult,
    FunctionCallError,
    FunctionCallErrorType,
    StructuredOutputMetrics,
    StructuredOutputReport,
)
from agentops.structured_output.validator import (
    SchemaValidator,
    FunctionCallValidator,
    validate_json_output,
    validate_function_call,
)
from agentops.structured_output.metrics import (
    schema_adherence_score,
    function_call_correctness,
    structured_output_composite,
    compute_structured_metrics,
)


# ═══════════════════════════════════════════════════════════════════════
# JSONSchema & JSONSchemaField Tests
# ═══════════════════════════════════════════════════════════════════════

class TestJSONSchemaField:
    def test_basic_field(self):
        f = JSONSchemaField("name", "string", required=True, description="User name")
        assert f.name == "name"
        assert f.type == "string"
        assert f.required is True
        assert f.description == "User name"
        assert f.enum_values is None

    def test_optional_field(self):
        f = JSONSchemaField("age", "integer", required=False)
        assert f.required is False

    def test_field_with_constraints(self):
        f = JSONSchemaField(
            "score", "number",
            required=True,
            minimum=0.0,
            maximum=100.0,
            enum_values=[0, 50, 100],
        )
        assert f.minimum == 0.0
        assert f.maximum == 100.0
        assert f.enum_values == [0, 50, 100]

    def test_field_with_pattern(self):
        f = JSONSchemaField("email", "string", pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
        assert f.pattern is not None

    def test_field_with_string_length(self):
        f = JSONSchemaField("bio", "string", min_length=10, max_length=500)
        assert f.min_length == 10
        assert f.max_length == 500

    def test_array_field_with_items(self):
        f = JSONSchemaField("tags", "array", items_type="string")
        assert f.items_type == "string"


class TestJSONSchema:
    def test_basic_schema(self):
        schema = JSONSchema(
            name="user",
            description="A user object",
            fields=[
                JSONSchemaField("name", "string", required=True),
                JSONSchemaField("age", "integer", required=False),
            ],
        )
        assert schema.name == "user"
        assert len(schema.fields) == 2
        assert schema.additional_properties is True

    def test_to_dict(self):
        schema = JSONSchema(
            name="item",
            fields=[
                JSONSchemaField("id", "string", required=True),
                JSONSchemaField("price", "number", required=True, minimum=0),
            ],
            additional_properties=False,
        )
        d = schema.to_dict()
        assert d["type"] == "object"
        assert "id" in d["properties"]
        assert d["properties"]["id"]["type"] == "string"
        assert d["properties"]["price"]["minimum"] == 0
        assert d["required"] == ["id", "price"]
        assert d["additionalProperties"] is False

    def test_from_dict(self):
        d = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "minimum": 0},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        schema = JSONSchema.from_dict("test", d)
        assert schema.name == "test"
        assert len(schema.fields) == 2
        assert schema.fields[0].name == "name"
        assert schema.fields[0].required is True
        assert schema.fields[1].required is False
        assert schema.additional_properties is False

    def test_from_dict_with_pattern(self):
        d = {
            "properties": {
                "email": {"type": "string", "pattern": r"^[\w.+-]+@[\w-]+\.[\w.-]+$"},
            },
        }
        schema = JSONSchema.from_dict("test", d)
        assert schema.fields[0].pattern is not None

    def test_from_dict_with_enum(self):
        d = {
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
        }
        schema = JSONSchema.from_dict("test", d)
        assert schema.fields[0].enum_values == ["active", "inactive"]


# ═══════════════════════════════════════════════════════════════════════
# SchemaValidator Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSchemaValidator:
    def test_valid_output(self):
        schema = JSONSchema(
            name="user",
            fields=[
                JSONSchemaField("name", "string", required=True),
                JSONSchemaField("age", "integer", required=True),
                JSONSchemaField("active", "boolean", required=False),
            ],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"name": "Alice", "age": 30, "active": true}')
        assert result.is_valid is True
        assert result.adherence_score == 1.0
        assert result.fields_valid == 3
        assert result.errors == []

    def test_missing_required_field(self):
        schema = JSONSchema(
            name="user",
            fields=[
                JSONSchemaField("name", "string", required=True),
                JSONSchemaField("age", "integer", required=True),
            ],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"name": "Alice"}')
        assert result.is_valid is False
        assert result.fields_invalid >= 0
        assert result.fields_missing == 1
        assert any(e.error_type == SchemaValidationErrorType.MISSING_REQUIRED for e in result.errors)

    def test_wrong_type(self):
        schema = JSONSchema(
            name="item",
            fields=[JSONSchemaField("count", "integer", required=True)],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"count": "not_a_number"}')
        assert result.is_valid is False
        assert any(e.error_type == SchemaValidationErrorType.WRONG_TYPE for e in result.errors)

    def test_invalid_enum(self):
        schema = JSONSchema(
            name="status",
            fields=[JSONSchemaField("level", "string", required=True, enum_values=["low", "medium", "high"])],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"level": "extreme"}')
        assert result.is_valid is False
        assert any(e.error_type == SchemaValidationErrorType.INVALID_ENUM for e in result.errors)

    def test_pattern_mismatch(self):
        schema = JSONSchema(
            name="config",
            fields=[JSONSchemaField("pipeline_name", "string", required=True, pattern=r"^[a-z][a-z0-9-]*$")],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"pipeline_name": "INVALID_NAME"}')
        assert result.is_valid is False
        assert any(e.error_type == SchemaValidationErrorType.PATTERN_MISMATCH for e in result.errors)

    def test_numeric_out_of_range(self):
        schema = JSONSchema(
            name="config",
            fields=[JSONSchemaField("cpu", "number", required=True, minimum=0.5, maximum=64)],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"cpu": 100}')
        assert result.is_valid is False
        assert any(e.error_type == SchemaValidationErrorType.OUT_OF_RANGE for e in result.errors)

    def test_malformed_json(self):
        schema = JSONSchema(
            name="test",
            fields=[JSONSchemaField("x", "string", required=True)],
        )
        validator = SchemaValidator(schema)
        result = validator.validate("not json at all")
        assert result.is_valid is False
        assert any(e.error_type == SchemaValidationErrorType.MALFORMED_JSON for e in result.errors)

    def test_not_json_object(self):
        schema = JSONSchema(
            name="test",
            fields=[JSONSchemaField("x", "string", required=True)],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('"just a string"')
        assert result.is_valid is False
        assert any(e.error_type == SchemaValidationErrorType.NOT_JSON for e in result.errors)

    def test_extra_field_not_allowed(self):
        schema = JSONSchema(
            name="restricted",
            fields=[JSONSchemaField("name", "string", required=True)],
            additional_properties=False,
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"name": "Alice", "extra": "should not be here"}')
        assert result.is_valid is False
        assert any(e.error_type == SchemaValidationErrorType.EXTRA_FIELD for e in result.errors)

    def test_optional_field_missing_is_ok(self):
        schema = JSONSchema(
            name="profile",
            fields=[
                JSONSchemaField("name", "string", required=True),
                JSONSchemaField("bio", "string", required=False),
            ],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"name": "Alice"}')
        assert result.is_valid is True

    def test_adherence_score_partial(self):
        schema = JSONSchema(
            name="item",
            fields=[
                JSONSchemaField("name", "string", required=True),
                JSONSchemaField("price", "number", required=True),
            ],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"name": "Widget", "price": "free"}')
        assert result.is_valid is False
        assert result.adherence_score == 0.5  # 1 correct out of 2

    def test_null_value_in_typed_field(self):
        schema = JSONSchema(
            name="item",
            fields=[JSONSchemaField("count", "integer", required=True)],
        )
        validator = SchemaValidator(schema)
        result = validator.validate('{"count": null}')
        assert result.is_valid is False
        assert any(e.error_type == SchemaValidationErrorType.WRONG_TYPE for e in result.errors)

    def test_string_length_constraints(self):
        schema = JSONSchema(
            name="post",
            fields=[JSONSchemaField("title", "string", required=True, min_length=5, max_length=20)],
        )
        validator = SchemaValidator(schema)

        # Too short
        result = validator.validate('{"title": "Hi"}')
        assert result.is_valid is False

        # Too long
        result = validator.validate('{"title": "This title is way too long for this field"}')
        assert result.is_valid is False

        # Just right
        result = validator.validate('{"title": "Hello World"}')
        assert result.is_valid is True

    def test_array_item_type_checking(self):
        schema = JSONSchema(
            name="data",
            fields=[JSONSchemaField("scores", "array", required=True, items_type="number")],
        )
        validator = SchemaValidator(schema)

        # Valid
        result = validator.validate('{"scores": [1, 2, 3]}')
        assert result.is_valid is True

        # Invalid — string in number array
        result = validator.validate('{"scores": [1, "two", 3]}')
        assert result.is_valid is False

    def test_boolean_type_strictness(self):
        """Ensure booleans aren't confused with integers (1/0)."""
        schema = JSONSchema(
            name="flag",
            fields=[JSONSchemaField("enabled", "boolean", required=True)],
        )
        validator = SchemaValidator(schema)

        result = validator.validate('{"enabled": true}')
        assert result.is_valid is True

        result = validator.validate('{"enabled": 1}')
        assert result.is_valid is False  # 1 is not a boolean

    def test_complex_valid_output(self):
        schema = JSONSchema(
            name="incident-report",
            fields=[
                JSONSchemaField("severity", "string", required=True, enum_values=["critical", "high", "medium", "low"]),
                JSONSchemaField("service", "string", required=True),
                JSONSchemaField("description", "string", required=True, max_length=500),
                JSONSchemaField("affected_users", "integer", required=False, minimum=0),
                JSONSchemaField("is_resolved", "boolean", required=True),
                JSONSchemaField("tags", "array", required=False, items_type="string"),
            ],
        )
        validator = SchemaValidator(schema)
        output = json.dumps({
            "severity": "high",
            "service": "api-gateway",
            "description": "Service experiencing elevated latency in EU-West-1.",
            "affected_users": 1500,
            "is_resolved": False,
            "tags": ["latency", "api-gateway"],
        })
        result = validator.validate(output)
        assert result.is_valid is True
        assert result.adherence_score == 1.0


# ═══════════════════════════════════════════════════════════════════════
# FunctionCallValidator Tests
# ═══════════════════════════════════════════════════════════════════════

SAMPLE_TOOL_SCHEMAS = {
    "search": {"query": "string", "limit": "integer"},
    "deploy": {"service": "string", "version": "string", "dry_run": "boolean"},
    "alert": {"message": "string", "severity": "string"},
}


class TestFunctionCallValidator:
    def test_correct_call(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c1",
            expected_tool="search",
            expected_params={"query": "error 500", "limit": 10},
            actual_call={"tool": "search", "params": {"query": "error 500", "limit": 10}},
        )
        assert result.is_correct is True
        assert result.correctness_score == 1.0
        assert result.errors == []

    def test_wrong_tool_selection(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c2",
            expected_tool="search",
            expected_params={"query": "error"},
            actual_call={"tool": "deploy", "params": {"service": "x", "version": "1"}},
        )
        assert result.is_correct is False
        assert any(e.error_type == FunctionCallErrorType.WRONG_TOOL for e in result.errors)

    def test_hallucinated_tool(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c3",
            expected_tool="search",
            expected_params={"query": "test"},
            actual_call={"tool": "nonexistent_tool", "params": {"query": "test"}},
        )
        assert result.is_correct is False
        assert any(e.error_type == FunctionCallErrorType.HALLUCINATED_TOOL for e in result.errors)

    def test_missing_parameter(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c4",
            expected_tool="search",
            expected_params={"query": "test", "limit": 10},
            actual_call={"tool": "search", "params": {"query": "test"}},
        )
        assert result.is_correct is False
        assert any(e.error_type == FunctionCallErrorType.MISSING_PARAM for e in result.errors)
        assert result.params_missing == 1

    def test_wrong_parameter_type(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c5",
            expected_tool="search",
            expected_params={"query": "test", "limit": 10},
            actual_call={"tool": "search", "params": {"query": "test", "limit": "ten"}},
        )
        assert result.is_correct is False
        assert any(e.error_type == FunctionCallErrorType.WRONG_PARAM_TYPE for e in result.errors)

    def test_invalid_parameter_value(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c6",
            expected_tool="search",
            expected_params={"query": "deployment failure"},
            actual_call={"tool": "search", "params": {"query": "something completely different"}},
        )
        assert result.is_correct is False
        assert any(e.error_type == FunctionCallErrorType.INVALID_PARAM_VALUE for e in result.errors)

    def test_extra_parameter(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c7",
            expected_tool="search",
            expected_params={"query": "test"},
            actual_call={"tool": "search", "params": {"query": "test", "extra_param": "unexpected"}},
        )
        assert result.is_correct is False
        assert any(e.error_type == FunctionCallErrorType.EXTRA_PARAM for e in result.errors)

    def test_boolean_param_type_checking(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c8",
            expected_tool="deploy",
            expected_params={"service": "api", "version": "2.0", "dry_run": True},
            actual_call={"tool": "deploy", "params": {"service": "api", "version": "2.0", "dry_run": "yes"}},
        )
        assert result.is_correct is False
        assert any(e.error_type == FunctionCallErrorType.WRONG_PARAM_TYPE for e in result.errors)

    def test_string_approximate_match(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c9",
            expected_tool="alert",
            expected_params={"message": "CPU usage critical", "severity": "high"},
            actual_call={"tool": "alert", "params": {"message": "cpu usage critical alert", "severity": "high"}},
        )
        # "cpu usage critical" should be found inside "cpu usage critical alert"
        assert result.is_correct is True

    def test_multiple_errors_in_one_call(self):
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c10",
            expected_tool="search",
            expected_params={"query": "test", "limit": 10},
            actual_call={"tool": "search", "params": {"query": 12345, "limit": "ten", "bonus": True}},
        )
        assert result.is_correct is False
        assert len(result.errors) >= 2  # wrong type + extra param

    def test_convenience_function(self):
        result = validate_function_call(
            expected_tool="search",
            expected_params={"query": "test"},
            actual_call={"tool": "search", "params": {"query": "test"}},
            tool_schemas=SAMPLE_TOOL_SCHEMAS,
        )
        assert result.is_correct is True

    def test_call_with_name_field(self):
        """Test that 'name' field is recognized as tool name."""
        validator = FunctionCallValidator(SAMPLE_TOOL_SCHEMAS)
        result = validator.validate(
            call_id="c11",
            expected_tool="search",
            expected_params={"query": "test"},
            actual_call={"name": "search", "arguments": {"query": "test"}},
        )
        assert result.is_correct is True


# ═══════════════════════════════════════════════════════════════════════
# StructuredOutput Metrics Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStructuredOutputMetrics:
    def test_schema_adherence_perfect(self):
        results = [
            SchemaValidationResult("s1", True, fields_total=3, fields_valid=3),
            SchemaValidationResult("s2", True, fields_total=5, fields_valid=5),
        ]
        assert schema_adherence_score(results) == 1.0

    def test_schema_adherence_partial(self):
        results = [
            SchemaValidationResult("s1", True, fields_total=4, fields_valid=4),
            SchemaValidationResult("s2", False, fields_total=4, fields_valid=2),
        ]
        score = schema_adherence_score(results)
        assert score == 0.75  # (1.0 + 0.5) / 2

    def test_schema_adherence_empty(self):
        assert schema_adherence_score([]) == 0.0

    def test_function_call_correctness_perfect(self):
        results = [
            FunctionCallResult("c1", "search", "search", True, params_total=2, params_correct=2),
            FunctionCallResult("c2", "deploy", "deploy", True, params_total=3, params_correct=3),
        ]
        assert function_call_correctness(results) == 1.0

    def test_function_call_correctness_mixed(self):
        results = [
            FunctionCallResult("c1", "s", "s", True, params_total=2, params_correct=2),
            FunctionCallResult("c2", "d", "x", False, params_total=2, params_correct=0),
        ]
        score = function_call_correctness(results)
        assert score == 0.5

    def test_function_call_correctness_empty(self):
        assert function_call_correctness([]) == 0.0

    def test_composite_score(self):
        score = structured_output_composite(0.8, 0.9)
        assert abs(score - 0.85) < 1e-9  # 0.5*0.8 + 0.5*0.9

    def test_compute_structured_metrics(self):
        schema_results = [
            SchemaValidationResult("s1", True, fields_total=2, fields_valid=2),
        ]
        fc_results = [
            FunctionCallResult("c1", "s", "s", True, params_total=2, params_correct=2),
            FunctionCallResult("c2", "d", "d", True, params_total=1, params_correct=1),
        ]
        metrics = compute_structured_metrics(schema_results, fc_results)
        assert metrics.avg_schema_adherence == 1.0
        assert metrics.avg_function_call_correctness == 1.0
        assert metrics.total_valid_outputs == 1
        assert metrics.total_correct_calls == 2
        assert metrics.composite_score == 1.0

    def test_compute_structured_metrics_with_errors(self):
        schema_results = [
            SchemaValidationResult("s1", False, fields_total=4, fields_valid=2),
        ]
        fc_results = [
            FunctionCallResult("c1", "s", "w", False, params_total=2, params_correct=0,
                              errors=[FunctionCallError(FunctionCallErrorType.WRONG_TOOL, "wrong tool", tool_name="w")]),
        ]
        metrics = compute_structured_metrics(schema_results, fc_results)
        assert metrics.total_valid_outputs == 0
        assert metrics.total_invalid_outputs == 1
        assert metrics.total_correct_calls == 0
        assert metrics.total_tool_selection_errors == 1


# ═══════════════════════════════════════════════════════════════════════
# StructuredOutputReport Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStructuredOutputReport:
    def test_empty_report(self):
        report = StructuredOutputReport(benchmark_name="test")
        assert report.benchmark_name == "test"
        assert report.schema_results == []
        assert report.function_call_results == []

    def test_to_markdown(self):
        report = StructuredOutputReport(
            benchmark_name="test-bench",
            schema_results=[
                SchemaValidationResult("incident", True, fields_total=3, fields_valid=3),
            ],
            function_call_results=[
                FunctionCallResult("c1", "search", "search", True, params_total=2, params_correct=2),
            ],
            metrics=StructuredOutputMetrics(
                avg_schema_adherence=1.0,
                total_valid_outputs=1,
                avg_function_call_correctness=1.0,
                total_correct_calls=1,
                composite_score=1.0,
            ),
        )
        md = report.to_markdown()
        assert "test-bench" in md
        assert "Schema Adherence" in md
        assert "Function Call Correctness" in md
        assert "Composite Score" in md

    def test_to_markdown_with_errors(self):
        report = StructuredOutputReport(
            benchmark_name="test",
            schema_results=[
                SchemaValidationResult(
                    "incident", False, fields_total=3, fields_valid=1,
                    errors=[SchemaValidationError("severity", SchemaValidationErrorType.WRONG_TYPE, "wrong")],
                ),
            ],
            function_call_results=[
                FunctionCallResult(
                    "c1", "search", "deploy", False, params_total=2, params_correct=0,
                    errors=[FunctionCallError(FunctionCallErrorType.WRONG_TOOL, "wrong")],
                ),
            ],
            metrics=StructuredOutputMetrics(
                avg_schema_adherence=0.33,
                total_invalid_outputs=1,
                avg_function_call_correctness=0.0,
                total_incorrect_calls=1,
                composite_score=0.165,
            ),
        )
        md = report.to_markdown()
        assert "FAIL" in md or "0.3" in md or "invalid" in md.lower()
        assert "wrong" in md.lower()

    def test_to_dict(self):
        report = StructuredOutputReport(
            benchmark_name="test",
            schema_results=[
                SchemaValidationResult("s1", True, fields_total=2, fields_valid=2),
            ],
            function_call_results=[
                FunctionCallResult("c1", "s", "s", True, params_total=1, params_correct=1),
            ],
            metrics=StructuredOutputMetrics(composite_score=1.0),
        )
        d = report.to_dict()
        assert d["benchmark_name"] == "test"
        assert len(d["schema_results"]) == 1
        assert len(d["function_call_results"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# Convenience function tests
# ═══════════════════════════════════════════════════════════════════════

class TestConvenienceFunctions:
    def test_validate_json_output(self):
        schema = JSONSchema(
            name="simple",
            fields=[JSONSchemaField("x", "integer", required=True)],
        )
        result = validate_json_output('{"x": 42}', schema)
        assert result.is_valid is True

    def test_validate_json_output_invalid(self):
        schema = JSONSchema(
            name="simple",
            fields=[JSONSchemaField("x", "integer", required=True)],
        )
        result = validate_json_output("not json", schema)
        assert result.is_valid is False


# ═══════════════════════════════════════════════════════════════════════
# Error type enums
# ═══════════════════════════════════════════════════════════════════════

class TestErrorTypes:
    def test_schema_error_types(self):
        assert SchemaValidationErrorType.MISSING_REQUIRED.value == "missing_required"
        assert SchemaValidationErrorType.WRONG_TYPE.value == "wrong_type"
        assert SchemaValidationErrorType.MALFORMED_JSON.value == "malformed_json"
        assert SchemaValidationErrorType.EXTRA_FIELD.value == "extra_field"

    def test_function_call_error_types(self):
        assert FunctionCallErrorType.WRONG_TOOL.value == "wrong_tool"
        assert FunctionCallErrorType.HALLUCINATED_TOOL.value == "hallucinated_tool"
        assert FunctionCallErrorType.MISSING_PARAM.value == "missing_param"
        assert FunctionCallErrorType.EXTRA_PARAM.value == "extra_param"
