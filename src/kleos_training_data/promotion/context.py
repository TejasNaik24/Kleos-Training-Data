from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kleos_training_data.contract.constants import ALLOWED_METADATA_EXTRAS, VARIATION_AXES
from kleos_training_data.contract.dedup import (
    CorpusEntry,
    CorpusIndex,
    DuplicateFinding,
    conversation_text,
)
from kleos_training_data.contract.schemas import ExampleMetadata, TrainingExample
from kleos_training_data.contract.writer import round_trips
from kleos_training_data.ids import CollisionLedger
from kleos_training_data.privacy.detect import Detection, scan_payload
from kleos_training_data.privacy.entities import EntityVault
from kleos_training_data.privacy.facts import assess
from kleos_training_data.privacy.sanitize import verify_sanitized
from kleos_training_data.review.records import HumanDecision, MachineReviewRecord, combine
from kleos_training_data.review.rubric import ReviewVerdict
from kleos_training_data.staging.records import SanitizedCandidate


@dataclass
class PromotionContext:
    candidate: SanitizedCandidate
    privacy: dict[str, Any] | None
    human: HumanDecision | None
    machine: MachineReviewRecord | None
    ledger: CollisionLedger
    corpus: CorpusIndex
    eval_corpus: CorpusIndex
    vault: EntityVault | None = None
    min_mean_score: float = 3.0
    near_duplicate_threshold: float = 0.85
    allowed_sources: frozenset[str] = field(default_factory=lambda: frozenset({"synthetic"}))

    _residual: list[Detection] | None = field(default=None, init=False, repr=False)

    @property
    def payload(self) -> dict[str, Any]:
        return self.candidate.payload

    @property
    def privacy_status(self) -> str:
        return (self.privacy or {}).get("status", "unknown")

    @property
    def source(self) -> str:
        return (self.payload.get("metadata") or {}).get(
            "source", self.candidate.lane.contract_source
        )

    @property
    def residual_detections(self) -> list[Detection]:
        if self._residual is None:
            self._residual = scan_payload(self.payload)
        return self._residual

    @property
    def fact_verdict(self) -> str:
        return assess(self.payload).verdict

    def fact_signal_ids(self) -> str:
        return ", ".join(sorted({s.rule_id for s in assess(self.payload).signals})) or "(none)"

    def schema_error(self) -> str | None:
        try:
            TrainingExample.model_validate(self.as_example())
        except Exception as exc:
            return str(exc).split("\n")[0]
        return None

    def as_example(self) -> dict[str, Any]:
        metadata = {
            "source": self.source,
            "quality_status": "reviewed",
            "scenario_family": self.candidate.scenario_family,
            "group_id": self.candidate.group_id,
            "perturbation_of": self.candidate.perturbation_of,
            "perturbation_kind": self.candidate.perturbation_kind,
            "author": "kleos-training-data",
            **{
                k: v
                for k, v in (self.payload.get("metadata") or {}).items()
                if k
                not in {
                    "source",
                    "quality_status",
                    "scenario_family",
                    "group_id",
                    "perturbation_of",
                    "perturbation_kind",
                    "author",
                }
            },
        }
        return {
            "id": self.candidate.candidate_id,
            "task": self.payload["task"],
            "messages": self.payload["messages"],
            "variation_axes": self.payload["variation_axes"],
            "metadata": metadata,
        }

    def surrogate_problems(self) -> list[str]:
        return verify_sanitized(self.payload, vault=self.vault)

    def unregistered_axes(self) -> list[str]:
        axes = self.payload.get("variation_axes") or {}
        return sorted(set(axes) - set(VARIATION_AXES))

    def disallowed_metadata_extras(self) -> list[str]:
        known = set(ExampleMetadata.model_fields)
        extras = set(self.as_example()["metadata"]) - known
        return sorted(extras - ALLOWED_METADATA_EXTRAS)

    def round_trips(self) -> bool:
        try:
            return round_trips(TrainingExample.model_validate(self.as_example()))
        except Exception:
            return False

    def _entry(self) -> CorpusEntry:
        return CorpusEntry(
            example_id=self.candidate.candidate_id,
            text=conversation_text(self.payload),
            scenario_family=self.candidate.scenario_family,
            entities=(self.payload.get("variation_axes") or {}).get("entities"),
        )

    def duplicate_findings(self) -> list[DuplicateFinding]:
        return self.corpus.find_duplicates(self._entry(), threshold=self.near_duplicate_threshold)

    def leakage_findings(self) -> list[DuplicateFinding]:
        full = self.corpus_scan(self.eval_corpus, include_assistant=True)
        prompt_only = self.corpus_scan(self.eval_corpus, include_assistant=False)

        seen: set[tuple[str, str]] = set()
        merged: list[DuplicateFinding] = []
        for finding in full + prompt_only:
            if finding.kind.value == "scenario_repeat":
                continue
            key = (finding.kind.value, finding.existing_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(finding)
        return merged

    def corpus_scan(self, index: CorpusIndex, *, include_assistant: bool) -> list[DuplicateFinding]:
        entry = CorpusEntry(
            example_id=self.candidate.candidate_id,
            text=conversation_text(self.payload, include_assistant=include_assistant),
            scenario_family=self.candidate.scenario_family,
        )
        return index.find_duplicates(entry, threshold=self.near_duplicate_threshold)

    def verdict(self) -> ReviewVerdict:
        if self.human is None:
            raise ValueError("no human decision")
        return combine(self.machine, self.human, min_mean_score=self.min_mean_score)
