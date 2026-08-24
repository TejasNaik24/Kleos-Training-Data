"""Duplicate and leakage detection, mirroring the public implementation.

Ported from ``kleos_models.data.leakage``. Four kinds, in increasing looseness:

``exact``
    Identical conversation text.
``normalized``
    Identical after :func:`normalize_text` — which collapses digit runs, so
    "Deadline: Mar 3" and "deadline mar 7" are the same scenario. That is
    deliberate and worth restating: **changing only a number does not make a
    scenario new.**
``near``
    Jaccard over character 5-grams above a threshold.
``scenario_repeat``
    Same ``scenario_family`` on both sides. Not a duplicate, but it is what
    makes a train/test split meaningless if it straddles the boundary.

The first two are fatal. Near-duplicates are reviewed rather than blindly
deleted: two examples at 0.87 similarity are sometimes a redundant pair and
sometimes a deliberate perturbation, and only a human can tell which.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kleos_training_data.hashing import jaccard, normalize_text, sha256_hex, shingles

#: Matches the public repo's default. Changing it changes which pairs a release
#: reports, so it is a research decision rather than a tuning knob.
DEFAULT_NEAR_DUPLICATE_THRESHOLD: float = 0.85

#: Shingle size for near-duplicate comparison. Character-level, so it survives
#: word reordering better than token shingles would.
SHINGLE_SIZE: int = 5


class DuplicateKind(str, Enum):
    """Why two examples are considered related."""

    EXACT = "exact_duplicate"
    NORMALIZED = "normalized_duplicate"
    NEAR = "near_duplicate"
    ID_COLLISION = "id_collision"
    SCENARIO_REPEAT = "scenario_repeat"


#: Kinds that block promotion outright. A near-duplicate is a judgement call; an
#: exact duplicate is not.
FATAL_KINDS: frozenset[DuplicateKind] = frozenset(
    {DuplicateKind.EXACT, DuplicateKind.NORMALIZED, DuplicateKind.ID_COLLISION}
)


@dataclass(frozen=True)
class DuplicateFinding:
    """One related pair."""

    kind: DuplicateKind
    candidate_id: str
    existing_id: str
    similarity: float
    detail: str = ""

    @property
    def fatal(self) -> bool:
        return self.kind in FATAL_KINDS

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "candidate_id": self.candidate_id,
            "existing_id": self.existing_id,
            "similarity": round(self.similarity, 4),
            "detail": self.detail,
        }


@dataclass
class CorpusEntry:
    """One already-promoted example, indexed for comparison."""

    example_id: str
    text: str
    scenario_family: str | None = None
    entities: str | None = None

    @property
    def exact_hash(self) -> str:
        return sha256_hex(self.text)

    @property
    def normalized_hash(self) -> str:
        return sha256_hex(normalize_text(self.text))


@dataclass
class CorpusIndex:
    """Everything already promoted, prepared for duplicate lookup."""

    entries: list[CorpusEntry] = field(default_factory=list)
    _by_exact: dict[str, str] = field(default_factory=dict, init=False)
    _by_normalized: dict[str, str] = field(default_factory=dict, init=False)
    _shingles: dict[str, set[str]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for entry in self.entries:
            self._register(entry)

    def _register(self, entry: CorpusEntry) -> None:
        self._by_exact.setdefault(entry.exact_hash, entry.example_id)
        self._by_normalized.setdefault(entry.normalized_hash, entry.example_id)
        self._shingles[entry.example_id] = shingles(normalize_text(entry.text), SHINGLE_SIZE)

    def add(self, entry: CorpusEntry) -> None:
        """Index one more example. Used as a promotion run proceeds."""
        self.entries.append(entry)
        self._register(entry)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def ids(self) -> set[str]:
        return {entry.example_id for entry in self.entries}

    def find_duplicates(
        self,
        candidate: CorpusEntry,
        *,
        threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
        check_scenario_repeat: bool = False,
    ) -> list[DuplicateFinding]:
        """Every relationship between ``candidate`` and the indexed corpus.

        Returns at most one duplicate finding per existing example, taking the
        strictest kind that applies — reporting an exact duplicate *and* a
        near-duplicate for the same pair would double-count one problem.
        """
        findings: list[DuplicateFinding] = []

        if candidate.example_id in self.ids:
            findings.append(
                DuplicateFinding(
                    kind=DuplicateKind.ID_COLLISION,
                    candidate_id=candidate.example_id,
                    existing_id=candidate.example_id,
                    similarity=1.0,
                    detail="an example with this id is already promoted",
                )
            )

        exact = self._by_exact.get(candidate.exact_hash)
        if exact is not None:
            return [
                *findings,
                DuplicateFinding(
                    kind=DuplicateKind.EXACT,
                    candidate_id=candidate.example_id,
                    existing_id=exact,
                    similarity=1.0,
                    detail="identical conversation text",
                ),
            ]

        normalized = self._by_normalized.get(candidate.normalized_hash)
        if normalized is not None:
            return [
                *findings,
                DuplicateFinding(
                    kind=DuplicateKind.NORMALIZED,
                    candidate_id=candidate.example_id,
                    existing_id=normalized,
                    similarity=1.0,
                    detail=(
                        "identical after normalization; changing only a number "
                        "does not make a scenario new"
                    ),
                ),
            ]

        target = shingles(normalize_text(candidate.text), SHINGLE_SIZE)
        for entry in self.entries:
            score = jaccard(target, self._shingles[entry.example_id])
            if score >= threshold:
                findings.append(
                    DuplicateFinding(
                        kind=DuplicateKind.NEAR,
                        candidate_id=candidate.example_id,
                        existing_id=entry.example_id,
                        similarity=score,
                        detail="review whether this is a redundant pair or a deliberate perturbation",
                    )
                )
            elif (
                check_scenario_repeat
                and candidate.scenario_family
                and candidate.scenario_family == entry.scenario_family
            ):
                findings.append(
                    DuplicateFinding(
                        kind=DuplicateKind.SCENARIO_REPEAT,
                        candidate_id=candidate.example_id,
                        existing_id=entry.example_id,
                        similarity=score,
                        detail=f"same scenario_family {candidate.scenario_family!r}",
                    )
                )

        return findings


def build_index(entries: Iterable[CorpusEntry]) -> CorpusIndex:
    """Index a collection of already-promoted examples."""
    return CorpusIndex(entries=list(entries))


def conversation_text(payload: Mapping[str, Any], *, include_assistant: bool = True) -> str:
    """Flat text view, matching ``TrainingExample.conversation_text``."""
    messages: list[Mapping[str, Any]] = payload.get("messages") or []
    return "\n".join(
        f"{m['role']}: {m['content']}"
        for m in messages
        if include_assistant or m.get("role") != "assistant"
    )
