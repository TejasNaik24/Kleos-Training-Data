"""The closed vocabulary of rejection reasons.

Closed on purpose. Free-text rejection reasons cannot be aggregated, and the
question this repository has to be able to answer is *which failure dominates* —
"we rejected 40% of candidates for private-fact risk" is actionable, "we rejected
a lot of things for various reasons" is not.

Adding a code is a deliberate act: it changes what the corpus report can say.
"""

from __future__ import annotations

from enum import Enum


class RejectionReason(str, Enum):
    """Why a candidate did not become training data.

    ``str`` subclass so the values stay JSON-serializable and comparable to
    plain strings, which the record models rely on.
    """

    # --- contract ---
    SCHEMA_INVALID = "SCHEMA_INVALID"
    ID_MISMATCH = "ID_MISMATCH"
    RENDER_ROUNDTRIP_MISMATCH = "RENDER_ROUNDTRIP_MISMATCH"
    METADATA_EXTRA_NOT_ALLOWED = "METADATA_EXTRA_NOT_ALLOWED"
    TASK_MISMATCH = "TASK_MISMATCH"
    AXES_INCOMPLETE = "AXES_INCOMPLETE"

    # --- privacy ---
    SECRET_DETECTED = "SECRET_DETECTED"
    PII_UNRESOLVED = "PII_UNRESOLVED"
    SURROGATE_RESIDUE = "SURROGATE_RESIDUE"
    PRIVATE_FACT = "PRIVATE_FACT"
    CONSENT_UNCLEAR = "CONSENT_UNCLEAR"

    # --- review ---
    REVIEW_MISSING = "REVIEW_MISSING"
    REVIEW_SIGNATURE_MISMATCH = "REVIEW_SIGNATURE_MISMATCH"
    REVIEW_GATE_FAILED = "REVIEW_GATE_FAILED"
    REVIEW_SCORE_BELOW_THRESHOLD = "REVIEW_SCORE_BELOW_THRESHOLD"
    OPERATOR_REJECTED = "OPERATOR_REJECTED"
    INCORRECT_BEHAVIOR = "INCORRECT_BEHAVIOR"
    BAD_POLICY = "BAD_POLICY"

    # --- quality ---
    CORPUS_DUPLICATE = "CORPUS_DUPLICATE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    EVAL_LEAKAGE = "EVAL_LEAKAGE"
    LOW_INFORMATION = "LOW_INFORMATION"
    INSUFFICIENT_VARIATION = "INSUFFICIENT_VARIATION"
    UNRESOLVED_AMBIGUITY = "UNRESOLVED_AMBIGUITY"

    # --- provenance ---
    PROVENANCE_INVALID = "PROVENANCE_INVALID"
    LANE_NOT_PROMOTABLE = "LANE_NOT_PROMOTABLE"
    STAGING_INTEGRITY = "STAGING_INTEGRITY"


#: Reasons that indicate a privacy failure rather than a quality one. These
#: drive exit code 4 and, when they fire on a production capture, an incident
#: rather than a re-run.
PRIVACY_REASONS: frozenset[RejectionReason] = frozenset(
    {
        RejectionReason.SECRET_DETECTED,
        RejectionReason.PII_UNRESOLVED,
        RejectionReason.SURROGATE_RESIDUE,
        RejectionReason.PRIVATE_FACT,
        RejectionReason.CONSENT_UNCLEAR,
    }
)

#: Reasons where fixing the content and re-submitting is the right next step.
#: The rest mean the candidate should not exist — a duplicate does not become
#: promotable by being edited.
RETRYABLE_REASONS: frozenset[RejectionReason] = frozenset(
    {
        RejectionReason.SCHEMA_INVALID,
        RejectionReason.AXES_INCOMPLETE,
        RejectionReason.PII_UNRESOLVED,
        RejectionReason.PRIVATE_FACT,
        RejectionReason.REVIEW_SCORE_BELOW_THRESHOLD,
        RejectionReason.INCORRECT_BEHAVIOR,
        RejectionReason.UNRESOLVED_AMBIGUITY,
        RejectionReason.METADATA_EXTRA_NOT_ALLOWED,
    }
)
