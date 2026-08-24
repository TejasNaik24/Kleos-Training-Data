"""The structured contract an AI reviewer must satisfy.

Validated twice, on purpose, because the two layers fail differently:

* **JSON Schema** catches a malformed generation with a field-level error good
  enough to retry on — "scores.actionability: 7 is greater than the maximum of
  4" tells a model exactly what to fix.
* **Pydantic** enforces the cross-field invariants that JSON Schema expresses
  badly: evidence required for every FAIL, confidence present, and the
  gates-dominate rule inherited from :class:`ReviewVerdict`.

``tests/test_review.py`` asserts the two accept and reject exactly the same
fixture payloads, so they cannot drift into disagreeing.

Nothing here lets prose flip a candidate into training data. An LLM produces a
structured opinion; deterministic gates and a human decide.
"""

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

#: JSON Schema handed to the model as its response format, and validated against
#: on receipt. Generated from the rubric so the two cannot diverge.
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
    """A machine reviewer's structured output, after validation."""

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
        """Every FAIL must name what failed it.

        A gate marked FAIL with no evidence is unactionable: a human cannot
        confirm it, cannot overrule it on the record, and cannot fix the
        content. In practice it is also the shape a model produces when it is
        guessing.
        """
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
        """The verdict this review implies. Gates dominate, as always."""
        from kleos_training_data.review.rubric import ReviewVerdict

        return ReviewVerdict.decide(self.scores, self.gates, min_mean_score=min_mean_score)


def validate_against_schema(payload: dict[str, Any]) -> list[str]:
    """Check a raw generation against the JSON Schema.

    Returns:
        Field-level error messages, empty when valid. Returned rather than
        raised because a malformed generation is usually worth one retry, and
        the messages are what the retry prompt needs.
    """
    import jsonschema

    validator = jsonschema.Draft202012Validator(REVIEW_RESPONSE_SCHEMA)
    return [
        f"{'.'.join(str(p) for p in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    ]


def parse_review(payload: dict[str, Any]) -> LLMReview:
    """Validate a generation through both layers.

    Raises:
        ValueError: If either layer rejects it. The JSON Schema runs first so
            the error names a field rather than a pydantic model.
    """
    errors = validate_against_schema(payload)
    if errors:
        raise ValueError("review does not match the response schema: " + "; ".join(errors))
    return LLMReview.model_validate(payload)
