"""Review: rubric, packets, machine review, and the human decision.

Deterministic gates are authoritative. A machine reviewer produces a structured
opinion. A human owns the final call — and cannot approve over a failing privacy
or private-fact gate, because that is not a judgement about one example.
"""

from __future__ import annotations

from kleos_training_data.review.llm_schema import (
    REVIEW_RESPONSE_SCHEMA,
    LLMReview,
    parse_review,
    validate_against_schema,
)
from kleos_training_data.review.packets import (
    PacketItem,
    ReviewPacket,
    machine_prompt,
    nearest_neighbours,
)
from kleos_training_data.review.records import (
    HumanDecision,
    MachineReviewRecord,
    combine,
)
from kleos_training_data.review.reviewers import (
    REVIEWERS,
    LLMReviewer,
    MockReviewer,
    render_instructions,
    resolve_reviewer,
)
from kleos_training_data.review.rubric import (
    DEFAULT_MIN_MEAN_SCORE,
    DIMENSIONS,
    HARD_GATES,
    MIN_DIMENSION_SCORE,
    RUBRIC_VERSION,
    UNOVERRIDABLE_GATES,
    GateResults,
    ReviewVerdict,
    RubricScores,
)

__all__ = [
    "DEFAULT_MIN_MEAN_SCORE",
    "DIMENSIONS",
    "HARD_GATES",
    "MIN_DIMENSION_SCORE",
    "REVIEWERS",
    "REVIEW_RESPONSE_SCHEMA",
    "RUBRIC_VERSION",
    "UNOVERRIDABLE_GATES",
    "GateResults",
    "HumanDecision",
    "LLMReview",
    "LLMReviewer",
    "MachineReviewRecord",
    "MockReviewer",
    "PacketItem",
    "ReviewPacket",
    "ReviewVerdict",
    "RubricScores",
    "combine",
    "machine_prompt",
    "nearest_neighbours",
    "parse_review",
    "render_instructions",
    "resolve_reviewer",
    "validate_against_schema",
]
