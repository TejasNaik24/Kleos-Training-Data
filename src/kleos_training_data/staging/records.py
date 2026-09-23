from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kleos_training_data.hashing import canonical_hash
from kleos_training_data.staging.reasons import RejectionReason

RECORD_VERSION: str = "1"


class CaptureLane(str, Enum):
    SYNTHETIC = "synthetic"
    MOCK_BACKEND = "mock_backend"
    PRODUCTION_OBSERVATION = "production_observation"

    @property
    def promotable(self) -> bool:
        return self is not CaptureLane.PRODUCTION_OBSERVATION

    @property
    def contract_source(self) -> str:
        return {
            CaptureLane.SYNTHETIC: "synthetic",
            CaptureLane.MOCK_BACKEND: "synthetic",
            CaptureLane.PRODUCTION_OBSERVATION: "real_sanitized",
        }[self]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StagingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_type: str
    record_version: str = RECORD_VERSION
    record_hash: str | None = None

    def compute_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"record_hash"})
        return canonical_hash(payload)

    def sealed(self) -> StagingRecord:
        return self.model_copy(update={"record_hash": self.compute_hash()})

    def hash_matches(self) -> bool:
        return self.record_hash is not None and self.record_hash == self.compute_hash()


class ScenarioRef(StagingRecord):
    record_type: str = "scenario_ref"
    family: str
    task: str
    point_index: int
    catalog_version: str
    policy_claim: str
    policy: str
    scenario_fingerprint: str


class TransportInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = 1
    retried_on: list[int] = Field(default_factory=list)
    status: int | None = None
    latency_ms: int | None = None
    request_ids: list[str] = Field(default_factory=list)
    frame_type_counts: dict[str, int] = Field(default_factory=dict)
    response_bytes: int | None = None


class RawCapture(StagingRecord):
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
    transformations: list[str] = Field(default_factory=list)


class SanitizedCandidate(StagingRecord):
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
    record_type: str = "promoted_example"
    example: dict[str, Any]
    audit: dict[str, Any]
    promoted_at: str = Field(default_factory=_now)
