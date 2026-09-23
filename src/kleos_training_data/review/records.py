from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from kleos_training_data.hashing import canonical_hash
from kleos_training_data.review.llm_schema import LLMReview
from kleos_training_data.review.rubric import (
    DIMENSIONS,
    RUBRIC_VERSION,
    GateResults,
    ReviewVerdict,
    RubricScores,
)
from kleos_training_data.staging.reasons import RejectionReason
from kleos_training_data.staging.records import StagingRecord


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MachineReviewRecord(StagingRecord):
    record_type: str = "llm_review"
    candidate_id: str
    content_hash: str
    packet_id: str
    reviewer_backend: str
    reviewer_model: str | None = None
    review: LLMReview
    reviewed_at: str = Field(default_factory=_now)

    @property
    def verdict(self) -> ReviewVerdict:
        return self.review.verdict(min_mean_score=0.0)


class HumanDecision(StagingRecord):
    record_type: str = "human_decision"
    candidate_id: str
    content_hash: str
    reviewer_role: str = "operator"
    decision: Literal["approve", "reject", "revise"]
    gates: GateResults
    scores: RubricScores | None = None
    score_overrides: dict[str, int] = Field(default_factory=dict)
    override_justification: str | None = None
    reason_codes: list[RejectionReason] = Field(default_factory=list)
    notes: str | None = None
    rubric_version: str = RUBRIC_VERSION
    decided_at: str = Field(default_factory=_now)
    signature: str | None = None

    @model_validator(mode="after")
    def _cannot_approve_over_privacy(self) -> HumanDecision:
        if self.decision != "approve":
            return self
        blocked = self.gates.failed_unoverridable()
        if blocked:
            raise ValueError(
                f"cannot approve over failing gate(s) {blocked}. These are not "
                f"overridable: fix the content instead, which changes its "
                f"content_hash and its id and requires a fresh review."
            )
        return self

    @model_validator(mode="after")
    def _approval_requires_every_gate(self) -> HumanDecision:
        if self.decision == "approve" and self.gates.any_failed():
            raise ValueError(f"cannot approve with gate(s) {self.gates.failed_ids()} failing")
        return self

    @model_validator(mode="after")
    def _overrides_are_justified(self) -> HumanDecision:
        if self.score_overrides and not (self.override_justification or "").strip():
            raise ValueError(
                "score_overrides require override_justification; an unexplained "
                "override cannot be audited later"
            )
        unknown = sorted(set(self.score_overrides) - set(DIMENSIONS))
        if unknown:
            raise ValueError(f"score_overrides names unknown dimension(s): {unknown}")
        for name, value in self.score_overrides.items():
            if not 0 <= value <= 4:
                raise ValueError(f"score override {name}={value} is outside 0-4")
        return self

    @model_validator(mode="after")
    def _rejection_carries_a_reason(self) -> HumanDecision:
        if self.decision == "reject" and not self.reason_codes:
            raise ValueError("a rejection needs at least one reason code")
        return self

    def compute_signature(self) -> str:
        payload = self.model_dump(mode="json", exclude={"signature", "record_hash"})
        return canonical_hash(payload)

    def signed(self) -> HumanDecision:
        return self.model_copy(update={"signature": self.compute_signature()})

    def signature_valid(self) -> bool:
        return self.signature is not None and self.signature == self.compute_signature()

    def applies_to(self, content_hash: str) -> bool:
        return self.signature_valid() and self.content_hash == content_hash


def combine(
    machine: MachineReviewRecord | None,
    human: HumanDecision,
    *,
    min_mean_score: float,
) -> ReviewVerdict:
    scores = human.scores
    if scores is None and machine is not None:
        scores = machine.review.scores
    if scores is None:
        scores = RubricScores(**dict.fromkeys(DIMENSIONS, 4))

    if human.score_overrides:
        scores = RubricScores(**{**scores.model_dump(), **human.score_overrides})

    if human.decision == "reject":
        return ReviewVerdict(
            scores=scores,
            gates=human.gates
            if human.gates.any_failed()
            else GateResults(
                no_private_data="PASS",
                policy_not_facts="PASS",
                no_unsupported_claims="PASS",
                schema_and_contract_valid="FAIL",
            ),
            decision="rejected",
        )

    if human.decision == "revise":
        return ReviewVerdict(scores=scores, gates=human.gates, decision="needs_revision")

    return ReviewVerdict.decide(scores, human.gates, min_mean_score=min_mean_score)
