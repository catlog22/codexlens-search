"""Data transformation pipeline with validation and filtering."""
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    records_in: int
    records_out: int
    errors: list[str]
    output: list[dict]


class DataValidator:
    """Validates records against a schema of required fields and types."""

    def __init__(self, schema: dict[str, type]):
        self._schema = schema

    def validate(self, record: dict) -> list[str]:
        errors = []
        for field_name, field_type in self._schema.items():
            if field_name not in record:
                errors.append(f"Missing required field: {field_name}")
            elif not isinstance(record[field_name], field_type):
                errors.append(
                    f"Field '{field_name}' expected {field_type.__name__}, "
                    f"got {type(record[field_name]).__name__}"
                )
        return errors


class TransformPipeline:
    """Chainable data transformation pipeline."""

    def __init__(self):
        self._steps: list[Callable[[dict], dict | None]] = []
        self._validator: DataValidator | None = None

    def add_step(self, fn: Callable[[dict], dict | None]) -> "TransformPipeline":
        self._steps.append(fn)
        return self

    def set_validator(self, validator: DataValidator) -> "TransformPipeline":
        self._validator = validator
        return self

    def execute(self, records: list[dict]) -> PipelineResult:
        errors: list[str] = []
        output: list[dict] = []

        for i, record in enumerate(records):
            if self._validator:
                validation_errors = self._validator.validate(record)
                if validation_errors:
                    errors.extend(f"Record {i}: {e}" for e in validation_errors)
                    continue

            current = record
            skip = False
            for step in self._steps:
                result = step(current)
                if result is None:
                    skip = True
                    break
                current = result

            if not skip:
                output.append(current)

        return PipelineResult(
            records_in=len(records),
            records_out=len(output),
            errors=errors,
            output=output,
        )
