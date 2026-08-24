"""On-disk record formats for the staging zone.

Every record is ``extra="forbid"`` and carries a ``record_hash`` over its own
canonical content. The store recomputes that hash on read, so a record edited
outside the pipeline is a *detectable* event rather than an invisible one.

That is not paranoia about tampering. It is the ordinary case of someone fixing
a typo by hand in a staged candidate after it was reviewed — which silently
invalidates the review signature bound to the old bytes, and would otherwise
promote an example nobody actually approved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kleos_training_data.hashing import canonical_hash
from kleos_training_data.staging.reasons import RejectionReason

#: Bumped when a record's shape changes in a way that makes old files
#: unreadable. Recorded on every record so a staging directory written by an
#: older version is diagnosable rather than merely broken.
RECORD_VERSION: str = "1"


class CaptureLane(str, Enum):
    """Where a candidate's content came from, and what it may become.

    This is the most consequential field in the repository. ``PRODUCTION_OBSERVATION``
    can never be promoted — promotion gate G10 rejects it outright — because the
    KLEOS backend answers from the authenticated user's own stored projects and
    memories, so every capture from it is one person's private data whatever the
    prompt was.

    Such a capture is *seed material*: a human reads it, learns what situation
    genuinely arises, and writes a new generalized scenario. The resulting
    example is ``synthetic_seeded`` and has no textual descent from the capture.
    """

    SYNTHETIC = "synthetic"
    MOCK_BACKEND = "mock_backend"
    PRODUCTION_OBSERVATION = "production_observation"

    @property
    def promotable(self) -> bool:
        return self is not CaptureLane.PRODUCTION_OBSERVATION

    @property
    def contract_source(self) -> str:
        """The ``metadata.source`` value a candidate from this lane may claim."""
        return {
            CaptureLane.SYNTHETIC: "synthetic",
            CaptureLane.MOCK_BACKEND: "synthetic",
            CaptureLane.PRODUCTION_OBSERVATION: "real_sanitized",
        }[self]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StagingRecord(BaseModel):
    """Base for everything written to the staging zone."""

    model_config = ConfigDict(extra="forbid")

    record_type: str
    record_version: str = RECORD_VERSION
    record_hash: str | None = None

    def compute_hash(self) -> str:
        """Digest over the record's content, excluding the hash field itself."""
        payload = self.model_dump(mode="json", exclude={"record_hash"})
        return canonical_hash(payload)

    def sealed(self) -> StagingRecord:
        """Return a copy carrying its own hash."""
        return self.model_copy(update={"record_hash": self.compute_hash()})

    def hash_matches(self) -> bool:
        """Whether the stored hash still describes the content."""
        return self.record_hash is not None and self.record_hash == self.compute_hash()


class ScenarioRef(StagingRecord):
    """Which scenario point a record came from."""

    record_type: str = "scenario_ref"
    family: str
    task: str
    point_index: int
    catalog_version: str
    policy_claim: str
    policy: str
    scenario_fingerprint: str


class TransportInfo(BaseModel):
    """What the transport did. Counts and timings only — never bodies."""

    model_config = ConfigDict(extra="forbid")

    attempts: int = 1
    retried_on: list[int] = Field(default_factory=list)
    status: int | None = None
    latency_ms: int | None = None
    request_ids: list[str] = Field(default_factory=list)
    frame_type_counts: dict[str, int] = Field(default_factory=dict)
    response_bytes: int | None = None


class RawCapture(StagingRecord):
    """Adapter output, untrusted.

    ``answer_text`` is the only field holding content, and it is unsanitized by
    definition. A raw capture is never promoted, never copied, and never
    committed — ``staging/`` is deny-by-default in ``.gitignore`` and guarded by
    the scanner's ``FORBIDDEN_PATHS``.
    """

    record_type: str = "raw_capture"
    capture_id: str
    batch_id: str
    lane: CaptureLane
    adapter: str
    adapter_version: str = "1"
    scenario: ScenarioRef

    endpoint: str
    request_field_names: list[str] = Field(default_factory=list)
    request_hash: str | None = None
    integrations_disabled: bool = True

    answer_text: str
    answer_sha256: str
    truncated: bool = False
    transport: TransportInfo = Field(default_factory=TransportInfo)
    captured_at: str = Field(default_factory=_now)


class NormalizedCandidate(StagingRecord):
    """Contract-shaped, pre-sanitization.

    ``provisional`` is true because the id is derived from content that
    sanitization may still change. ``sanitize`` mints the final id and records
    the one it superseded.
    """

    record_type: str = "normalized_candidate"
    candidate_id: str
    provisional: bool = True
    source_capture_id: str | None = None
    batch_id: str
    lane: CaptureLane
    scenario: ScenarioRef

    payload: dict[str, Any]

    scenario_family: str
    group_id: str
    perturbation_of: str | None = None
    perturbation_kind: str | None = None

    content_hash: str
    normalized_at: str = Field(default_factory=_now)
    #: What normalization changed, so a transformation is never invisible.
    transformations: list[str] = Field(default_factory=list)


class SanitizedCandidate(StagingRecord):
    """Post-redaction, ready for review."""

    record_type: str = "sanitized_candidate"
    candidate_id: str
    superseded_candidate_id: str | None = None
    batch_id: str
    lane: CaptureLane
    scenario: ScenarioRef

    sanitization_status: str
    ruleset_version: str
    surrogate_map_id: str | None = None

    payload: dict[str, Any]
    scenario_family: str
    group_id: str
    perturbation_of: str | None = None
    perturbation_kind: str | None = None

    content_hash: str
    sanitized_at: str = Field(default_factory=_now)


class RejectionRecord(StagingRecord):
    """Why a candidate did not proceed.

    Keeps the reason and the hash, not the offending content: "which failure
    dominates?" must stay answerable without retaining the text that failed.
    """

    record_type: str = "rejection"
    candidate_id: str
    content_hash: str
    stage: str
    reason_codes: list[RejectionReason]
    detail: str = ""
    remediation: str | None = None
    retryable: bool = False
    rejected_at: str = Field(default_factory=_now)


class PromotedExample(StagingRecord):
    """A candidate that passed every gate.

    ``example`` is the exact dict that will be written to ``train.jsonl``.
    ``audit`` stays in ``staging/`` and is **never** written into a release — it
    would be a pointer back into the staging zone surviving into whatever the
    release is shared with.
    """

    record_type: str = "promoted_example"
    example: dict[str, Any]
    audit: dict[str, Any]
    promoted_at: str = Field(default_factory=_now)
