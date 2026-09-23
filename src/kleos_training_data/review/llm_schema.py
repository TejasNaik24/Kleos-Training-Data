from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kleos_training_data.review.rubric import (
    DIMENSIONS,
    HARD_GATES,
    RUBRIC_VERSION,
    GateResults,
    RubricScores,
)

REVIEW_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "KLEOS training-data review",
    "type": "object",
    "additionalProperties": False,
    "required": ["scores", "gates", "gate_evidence", "rationale", "reviewer_confidence"],
    "properties": {
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(DIMENSIONS),
            "properties": {
                name: {"type": "integer", "minimum": 0, "maximum": 4, "description": question}
                for name, question in DIMENSIONS.items()
            },
        },
        "gates": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(HARD_GATES),
            "properties": {
                name: {"type": "string", "enum": ["PASS", "FAIL"], "description": question}
                for name, question in HARD_GATES.items()
            },
        },
        "gate_evidence": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": (
                "For every gate marked FAIL, the specific span or claim that "
                "failed it. A FAIL without evidence is unactionable."
            ),
        },
        "rationale": {"type": "string", "minLength": 20},
        "suggested_revision": {"type": ["string", "null"]},
        "reviewer_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


class LLMReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: RubricScores
    gates: GateResults
    gate_evidence: dict[str, str] = Field(default_factory=dict)
    rationale: str = Field(min_length=20)
    suggested_revision: str | None = None
    reviewer_confidence: float = Field(ge=0.0, le=1.0)
    rubric_version: str = RUBRIC_VERSION

    @model_validator(mode="after")
    def _failures_carry_evidence(self) -> LLMReview:
        missing = [gate for gate in self.gates.failed_ids() if not self.gate_evidence.get(gate)]
        if missing:
            raise ValueError(
                f"gate(s) {missing} are FAIL but carry no evidence; a failure "
                f"nobody can act on is worse than no review"
            )
        return self

    @model_validator(mode="after")
    def _evidence_names_real_gates(self) -> LLMReview:
        unknown = sorted(set(self.gate_evidence) - set(HARD_GATES))
        if unknown:
            raise ValueError(f"gate_evidence names unknown gate(s): {unknown}")
        return self

    def verdict(self, *, min_mean_score: float) -> Any:
        from kleos_training_data.review.rubric import ReviewVerdict

        return ReviewVerdict.decide(self.scores, self.gates, min_mean_score=min_mean_score)


def validate_against_schema(payload: dict[str, Any]) -> list[str]:
    import jsonschema

    validator = jsonschema.Draft202012Validator(REVIEW_RESPONSE_SCHEMA)
    return [
        f"{'.'.join(str(p) for p in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    ]


def parse_review(payload: dict[str, Any]) -> LLMReview:
    errors = validate_against_schema(payload)
    if errors:
        raise ValueError("review does not match the response schema: " + "; ".join(errors))
    return LLMReview.model_validate(payload)
