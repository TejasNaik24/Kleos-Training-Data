"""Machine reviewers.

:class:`MockReviewer` is deterministic and rule-driven. It is what CI uses and
what the test suite exercises, so the entire review stage runs with no network
and no API key. That is not a convenience — a review stage testable only against
a live model is one whose behaviour nobody can pin down, and whose failures
arrive as flakes.

The real reviewer arrives in Phase J behind the ``review`` extra. Neither one
gets to approve anything: both produce a structured opinion that deterministic
gates and a human then act on.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kleos_training_data.privacy.detect import scan_payload
from kleos_training_data.privacy.facts import assess
from kleos_training_data.review.llm_schema import LLMReview, parse_review
from kleos_training_data.review.rubric import DIMENSIONS, HARD_GATES


@runtime_checkable
class LLMReviewer(Protocol):
    """What every machine reviewer must provide."""

    name: str

    def review(self, packet_item: dict[str, Any]) -> LLMReview:
        """Score one candidate."""
        ...


REVIEW_INSTRUCTIONS = """\
You are reviewing a candidate training example for a research dataset.

The dataset teaches a **generalizable decision policy**, not facts about any
individual. Judge the example against that goal.

Score each dimension 0-4:
{dimensions}

Then decide each hard gate, PASS or FAIL:
{gates}

A hard gate is not a score. If any gate fails, the example is rejected however
good it reads. For every FAIL, name the specific span or claim in
`gate_evidence` — a failure nobody can act on is worse than no review.

The question behind `policy_not_facts`: if a model memorized this example and
surfaced it later, would it reveal something about a real person?

Respond with JSON matching the supplied schema. Nothing else.
"""


def render_instructions() -> str:
    """The instruction block sent to a machine reviewer."""
    return REVIEW_INSTRUCTIONS.format(
        dimensions="\n".join(f"  - {k}: {v}" for k, v in DIMENSIONS.items()),
        gates="\n".join(f"  - {k}: {v}" for k, v in HARD_GATES.items()),
    )


class MockReviewer:
    """A deterministic reviewer derived from the deterministic signals.

    It does not pretend to have an opinion. It reads the privacy scan, the
    private-fact assessment and the structural properties of the example, and
    scores from those. That makes it useful for exactly one thing — proving the
    review *stage* works end to end — and useless for judging quality, which is
    the honest division of labour.
    """

    name = "mock"

    def __init__(self, *, base_score: int = 4) -> None:
        self.base_score = base_score

    def review(self, packet_item: dict[str, Any]) -> LLMReview:
        payload = packet_item["payload"]
        detections = scan_payload(payload)
        fact_risk = assess(payload)

        # By the time review runs, the candidate is post-sanitization. So *any*
        # surviving detection means redaction missed something — a `redact`
        # severity hit here is not "handled", it is evidence that handling
        # failed. Grading only `block` and `review` would let a surviving email
        # address pass the gate that exists to catch it.
        blocking = [d for d in detections if d.severity == "block"]
        needs_review = [d for d in detections if d.severity in {"redact", "review"}]

        messages = payload.get("messages") or []
        assistant = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"), ""
        )
        prompt = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") != "assistant"
        )

        gates: dict[str, str] = {
            "no_private_data": "FAIL" if blocking or needs_review else "PASS",
            "policy_not_facts": "FAIL" if fact_risk.verdict == "fact_teaching" else "PASS",
            "no_unsupported_claims": (
                "FAIL"
                if any(s.rule_id == "fact.unsupported_entity" for s in fact_risk.signals)
                else "PASS"
            ),
            "schema_and_contract_valid": "PASS"
            if packet_item.get("contract_valid", True)
            else "FAIL",
        }

        evidence: dict[str, str] = {}
        if gates["no_private_data"] == "FAIL":
            kinds = sorted({d.kind for d in blocking + needs_review})
            evidence["no_private_data"] = f"privacy rules fired: {', '.join(kinds)}"
        if gates["policy_not_facts"] == "FAIL":
            evidence["policy_not_facts"] = (
                f"private-fact verdict {fact_risk.verdict}: "
                f"{', '.join(sorted({s.rule_id for s in fact_risk.signals}))}"
            )
        if gates["no_unsupported_claims"] == "FAIL":
            evidence["no_unsupported_claims"] = (
                "the assistant names an entity absent from the prompt"
            )
        if gates["schema_and_contract_valid"] == "FAIL":
            evidence["schema_and_contract_valid"] = "the payload failed contract validation"

        # Scores derived from structural properties, not from taste.
        base = self.base_score
        scores = {
            "decision_correctness": base,
            "evidence_grounding": base if not fact_risk.signals else 2,
            "reasoning_quality": base if len(assistant) > 60 else 2,
            "actionability": base if assistant.strip() else 0,
            "format_compliance": base,
            "generalizability": 2 if fact_risk.requires_human else base,
        }

        return parse_review(
            {
                "scores": scores,
                "gates": gates,
                "gate_evidence": evidence,
                "rationale": (
                    f"Deterministic review: {len(detections)} privacy detection(s), "
                    f"fact verdict {fact_risk.verdict}, prompt {len(prompt)} chars, "
                    f"answer {len(assistant)} chars."
                ),
                "suggested_revision": None,
                "reviewer_confidence": 1.0 if not fact_risk.signals else 0.6,
            }
        )


class AnthropicReviewer:
    """A real model reviewer, behind the ``review`` extra.

    It produces a structured opinion and nothing more. The output is validated
    twice — JSON Schema then pydantic — and a human decision is still required
    before anything can be promoted. There is no code path by which model prose
    becomes an approval.

    Refused when ``CI`` is set: an automated run has no human to own the
    decision afterwards, and CI uses the deterministic MockReviewer so the
    review stage stays testable without a key or a bill.
    """

    name = "anthropic"

    def __init__(self, *, model: str = "claude-sonnet-5", max_retries: int = 2) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        import os

        from kleos_training_data.errors import MissingDependencyError, ReviewError

        if os.environ.get("CI"):
            raise ReviewError(
                "The Anthropic reviewer is refused in CI.",
                suggestions=[
                    "CI uses --reviewer mock: deterministic, no key, no bill.",
                    "A machine review needs a human to own the decision afterwards, "
                    "and an automated run has none.",
                ],
            )
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ReviewError(
                "ANTHROPIC_API_KEY is not set.",
                suggestions=["Set it in .env, or use --reviewer mock."],
            )
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise MissingDependencyError(
                    "anthropic", extra="review", purpose="run a model reviewer"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def review(self, packet_item: dict[str, Any]) -> LLMReview:
        """Score one candidate, retrying only a malformed generation.

        A schema violation is worth one retry — the error names the field, which
        is exactly what a retry prompt needs. Anything else is not.
        """
        import json

        from kleos_training_data.errors import ReviewError
        from kleos_training_data.review.llm_schema import (
            REVIEW_RESPONSE_SCHEMA,
            validate_against_schema,
        )

        client = self._ensure_client()
        prompt = (
            render_instructions()
            + "\n\nResponse schema:\n"
            + json.dumps(REVIEW_RESPONSE_SCHEMA, indent=2)
            + "\n\nCandidate:\n"
            + json.dumps(packet_item, indent=2, ensure_ascii=False, sort_keys=True)
        )

        errors: list[str] = []
        for _attempt in range(self.max_retries + 1):
            message = client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                        + (
                            f"\n\nYour previous response was invalid: {'; '.join(errors)}"
                            if errors
                            else ""
                        ),
                    }
                ],
            )
            text = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            )
            try:
                payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
            except (ValueError, json.JSONDecodeError):
                errors = ["the response was not JSON"]
                continue

            errors = validate_against_schema(payload)
            if not errors:
                return parse_review(payload)

        raise ReviewError(
            f"The reviewer produced an invalid response {self.max_retries + 1} time(s).",
            details={"errors": "; ".join(errors)},
            suggestions=[
                "Arbitrary prose is never accepted as a review: the schema is what "
                "makes a machine opinion actionable.",
            ],
        )


#: Registry. The real reviewer registers here once the `review` extra exists.
REVIEWERS: dict[str, type] = {"mock": MockReviewer, "anthropic": AnthropicReviewer}


def resolve_reviewer(name: str) -> LLMReviewer:
    """Instantiate a reviewer by name."""
    from kleos_training_data.errors import ReviewError

    factory = REVIEWERS.get(name)
    if factory is None:
        raise ReviewError(
            f"Unknown reviewer {name!r}.",
            details={"available": ", ".join(sorted(REVIEWERS))},
            suggestions=[
                "The Anthropic reviewer arrives with the `review` extra. CI uses "
                "the mock, so the review stage stays testable without a key.",
            ],
        )
    return factory()
