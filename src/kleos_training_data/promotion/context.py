"""Everything a gate needs to judge one candidate.

Assembled once and passed to every gate, so no gate does its own I/O. That keeps
gates pure functions of a value, which is what makes each one testable against a
hand-built context rather than against a populated staging directory.
"""

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
from kleos_training_data.contract.schemas import TrainingExample
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
    """One candidate, plus everything needed to decide about it."""

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

    # --- cached derivations -------------------------------------------------

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
        """Detections in the *sanitized* payload.

        Re-scanned here rather than trusted from the sanitization record. The
        record says what sanitization believed; this says what is actually in the
        bytes about to be promoted, which is the only thing that matters at this
        point.
        """
        if self._residual is None:
            self._residual = scan_payload(self.payload)
        return self._residual

    @property
    def fact_verdict(self) -> str:
        return assess(self.payload).verdict

    def fact_signal_ids(self) -> str:
        return ", ".join(sorted({s.rule_id for s in assess(self.payload).signals})) or "(none)"

    def schema_error(self) -> str | None:
        """The contract error, or ``None`` when it validates."""
        try:
            TrainingExample.model_validate(self.as_example())
        except Exception as exc:
            return str(exc).split("\n")[0]
        return None

    def as_example(self) -> dict[str, Any]:
        """The payload in final contract shape, with id and metadata attached."""
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
        """Metadata keys that are neither contract fields nor allowlisted.

        ``ExampleMetadata`` is ``extra="allow"`` upstream, so anything here ships
        inside ``train.jsonl``. A stray ``capture_id`` or ``input_hash`` would be
        a live pointer back into the staging zone, surviving into whatever the
        release is shared with.
        """
        known = set(TrainingExample.model_fields["metadata"].annotation.model_fields)  # type: ignore[union-attr]
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
        """Overlap with held-out evaluation material.

        Run twice, deliberately. The public ``conversation_text`` includes the
        assistant turn for a training example but is prompt-only for an
        evaluation example, so a single comparison is asymmetric exactly where it
        matters most — a training candidate whose *prompt* matches an eval prompt
        is leakage regardless of what the answers say.

        The first pass matches public semantics so our verdict is never weaker
        than theirs; the second compares prompt to prompt.
        """
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
        if self.human is None:  # pragma: no cover - guarded by G08/G09
            raise ValueError("no human decision")
        return combine(self.machine, self.human, min_mean_score=self.min_mean_score)
