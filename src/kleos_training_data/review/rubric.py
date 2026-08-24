"""The review rubric, and the invariant that hard gates dominate scores.

Six scored dimensions, 0–4, higher is better. Four PASS/FAIL hard gates.

The rule is simple to state and easy to erode: **a failing hard gate overrides
every score.** An example can be well-written, correctly reasoned, perfectly
formatted, and still contain someone's phone number.

Stating that in a docstring is not enough, and neither is checking it in one
function — the check gets bypassed the first time somebody constructs a verdict
by another path, deserializes one from JSON, or refactors the decision logic. So
it is a **model invariant**: :class:`ReviewVerdict` refuses to exist with a
failing gate and any decision other than ``rejected``. Not by convention, not by
a call to a validator someone must remember — by construction, on every path
including ``model_validate``.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Bumped when a dimension, gate or threshold changes. Recorded on every review,
#: so two reviews scored under different rubrics are never silently averaged.
RUBRIC_VERSION: Final[str] = "review-rubric-v1"

#: Scored dimensions, with the question each one actually asks.
DIMENSIONS: Final[dict[str, str]] = {
    "decision_correctness": "Is the decision the one the stated policy implies?",
    "evidence_grounding": "Is every claim traceable to something in the prompt?",
    "reasoning_quality": (
        "Is the justification the *right* reason — not a right answer reached "
        "for a wrong or unstated reason?"
    ),
    "actionability": "Could the reader act on this without a follow-up question?",
    "format_compliance": "Does it match the format the task and axes call for?",
    "generalizability": (
        "Would this same reasoning transfer to a different person with a different set of entities?"
    ),
}

#: Hard gates. Any FAIL rejects, whatever the scores say.
HARD_GATES: Final[dict[str, str]] = {
    "no_private_data": "No PII, secret or identifying detail survives in the text.",
    "policy_not_facts": (
        "The example teaches a transferable policy, not a fact about a real "
        "individual. If a model memorized it and surfaced it later, nothing "
        "about a real person would be revealed."
    ),
    "no_unsupported_claims": "The answer asserts nothing the prompt does not support.",
    "schema_and_contract_valid": "The example satisfies the public dataset contract.",
}

#: Gates a human may never overrule. Fixing the content is the only way forward,
#: and that changes its hash, its id, and therefore requires a fresh review.
UNOVERRIDABLE_GATES: Final[frozenset[str]] = frozenset({"no_private_data", "policy_not_facts"})

#: Minimum score any single dimension may have.
MIN_DIMENSION_SCORE: Final[int] = 2

#: Default minimum mean across dimensions. A scenario may raise it.
DEFAULT_MIN_MEAN_SCORE: Final[float] = 3.0

GateStatus = Literal["PASS", "FAIL"]
Decision = Literal["approved", "needs_revision", "rejected"]


class RubricScores(BaseModel):
    """One score per dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_correctness: int = Field(ge=0, le=4)
    evidence_grounding: int = Field(ge=0, le=4)
    reasoning_quality: int = Field(ge=0, le=4)
    actionability: int = Field(ge=0, le=4)
    format_compliance: int = Field(ge=0, le=4)
    generalizability: int = Field(ge=0, le=4)

    @property
    def values(self) -> list[int]:
        return [getattr(self, name) for name in DIMENSIONS]

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    @property
    def minimum(self) -> int:
        return min(self.values)

    def below_floor(self) -> list[str]:
        """Dimensions under the per-dimension floor.

        Checked separately from the mean, because a 0 in one dimension is not
        redeemed by 4s elsewhere. An example that is beautifully written and
        reaches the wrong decision is not a 3.2-quality example.
        """
        return sorted(name for name in DIMENSIONS if getattr(self, name) < MIN_DIMENSION_SCORE)


class GateResults(BaseModel):
    """PASS/FAIL for each hard gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    no_private_data: GateStatus
    policy_not_facts: GateStatus
    no_unsupported_claims: GateStatus
    schema_and_contract_valid: GateStatus

    def failed_ids(self) -> list[str]:
        return sorted(name for name in HARD_GATES if getattr(self, name) == "FAIL")

    def any_failed(self) -> bool:
        return bool(self.failed_ids())

    def failed_unoverridable(self) -> list[str]:
        """Failing gates a human is not permitted to overrule."""
        return sorted(set(self.failed_ids()) & UNOVERRIDABLE_GATES)

    @classmethod
    def all_pass(cls) -> GateResults:
        return cls(
            no_private_data="PASS",
            policy_not_facts="PASS",
            no_unsupported_claims="PASS",
            schema_and_contract_valid="PASS",
        )


class ReviewVerdict(BaseModel):
    """Scores, gates, and the decision they imply.

    The invariant lives here. A verdict carrying a failing gate and a decision
    other than ``rejected`` **cannot be constructed** — not by :meth:`decide`,
    not by a human record, not by ``model_validate`` on a JSON payload, and not
    by a future refactor that forgets the rule.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: RubricScores
    gates: GateResults
    decision: Decision
    rubric_version: str = RUBRIC_VERSION

    @model_validator(mode="after")
    def _gates_dominate(self) -> ReviewVerdict:
        if self.gates.any_failed() and self.decision != "rejected":
            raise ValueError(
                f"gate(s) {self.gates.failed_ids()} FAILED, so decision must be "
                f"'rejected'; got {self.decision!r}. Hard gates are not scoreable — "
                f"an example can be well-written, correctly reasoned and still "
                f"contain someone's phone number."
            )
        return self

    @classmethod
    def decide(
        cls,
        scores: RubricScores,
        gates: GateResults,
        *,
        min_mean_score: float = DEFAULT_MIN_MEAN_SCORE,
    ) -> ReviewVerdict:
        """Derive the decision from scores and gates.

        Order matters: gates first, unconditionally. Only once every gate passes
        do the numbers get a say.
        """
        if gates.any_failed():
            return cls(scores=scores, gates=gates, decision="rejected")
        if scores.below_floor() or scores.mean < min_mean_score:
            return cls(scores=scores, gates=gates, decision="needs_revision")
        return cls(scores=scores, gates=gates, decision="approved")

    @property
    def approved(self) -> bool:
        return self.decision == "approved"

    def explain(self) -> str:
        """One line saying what drove the decision."""
        if self.gates.any_failed():
            return f"rejected: hard gate(s) failed — {', '.join(self.gates.failed_ids())}"
        if self.decision == "needs_revision":
            low = self.scores.below_floor()
            if low:
                return f"needs revision: {', '.join(low)} below the floor of {MIN_DIMENSION_SCORE}"
            return f"needs revision: mean {self.scores.mean:.2f} below threshold"
        return f"approved: all gates pass, mean {self.scores.mean:.2f}"
