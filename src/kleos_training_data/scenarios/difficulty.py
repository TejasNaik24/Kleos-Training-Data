from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Final

from kleos_training_data.scenarios.situations import EVIDENCE_WEIGHTS, Item, Situation

GATE_COMPONENTS: Final[frozenset[str]] = frozenset({"scope", "adequate", "explicit"})

ORDERING_KEYS: Final[dict[str, tuple[tuple[str, Callable[[Item], float]], ...]]] = {
    "rank_by_deadline_then_evidence": (("score", lambda i: _score(i)),),
    "rank_or_abstain_when_close": (("score", lambda i: _score(i)),),
    "verify_when_evidence_weak": (("score", lambda i: _score(i)),),
    "ask_when_request_ambiguous": (("score", lambda i: _score(i)),),
    "respect_workspace_scope": (
        ("scope", lambda i: 1.0 if i.in_scope else 0.0),
        ("score", lambda i: _score(i)),
    ),
    "ask_before_crossing_workspace": (
        ("scope", lambda i: 1.0 if i.in_scope else 0.0),
        ("score", lambda i: _score(i)),
    ),
    "rank_by_reliability_over_recency": (
        ("evidence", lambda i: i.evidence_weight),
        ("recency", lambda i: -float(i.deadline_days)),
    ),
    "resolve_or_abstain_on_support": (
        ("evidence", lambda i: i.evidence_weight),
        ("recency", lambda i: -float(i.deadline_days)),
    ),
    "defer_to_explicit_statement": (
        ("explicit", lambda i: 1.0 if i.evidence == "confirmed" else 0.0),
        ("evidence", lambda i: i.evidence_weight),
        ("recency", lambda i: -float(i.deadline_days)),
    ),
    "rank_by_relevance_over_recency": (
        ("relevance", lambda i: i.impact_weight),
        ("recency", lambda i: -float(i.deadline_days)),
    ),
    "prefer_least_privilege_source": (
        ("adequate", lambda i: 1.0 if i.evidence_weight >= EVIDENCE_WEIGHTS["reported"] else 0.0),
        ("narrowness", lambda i: -i.impact_weight),
        ("evidence", lambda i: i.evidence_weight),
    ),
    "select_by_evidence_need": (
        ("bearing", lambda i: i.evidence_weight * i.impact_weight),
        ("freshness", lambda i: -float(i.deadline_days)),
    ),
}

ONE_STEP: Final[float] = 0.20

DIFFICULTY_LEVELS: Final[tuple[str, ...]] = ("easy", "medium", "hard")


def _score(item: Item) -> float:
    return item.deadline_score * item.evidence_weight * item.impact_weight


def evidence_scale_min_step() -> float:
    grades = sorted(EVIDENCE_WEIGHTS.values(), reverse=True)
    return min((a - b) / a for a, b in pairwise(grades))


class Separation:
    __slots__ = ("component", "depth", "gap", "is_gate", "relative_gap")

    def __init__(
        self,
        component: str,
        depth: int,
        gap: float,
        relative_gap: float,
        *,
        is_gate: bool = False,
    ) -> None:
        self.component = component
        self.depth = depth
        self.gap = gap
        self.relative_gap = relative_gap
        self.is_gate = is_gate

    def __repr__(self) -> str:
        return (
            f"Separation(component={self.component!r}, depth={self.depth}, "
            f"relative_gap={self.relative_gap:.3f})"
        )


ARBITRARY_TIEBREAK: Final[int] = 99


def separation(situation: Situation, policy: str) -> Separation | None:
    components = ORDERING_KEYS.get(policy)
    if components is None or len(situation.items) < 2:
        return None

    def sort_key(item: Item) -> tuple:
        return (*(-extract(item) for _, extract in components), item.key)

    ordered = sorted(situation.items, key=sort_key)
    top, runner_up = ordered[0], ordered[1]

    rank_depth = 0
    for name, extract in components:
        first, second = extract(top), extract(runner_up)
        is_gate = name in GATE_COMPONENTS
        if first != second:
            scale = max(abs(first), abs(second), 1e-9)
            return Separation(
                component=name,
                depth=rank_depth,
                gap=abs(first - second),
                relative_gap=abs(first - second) / scale,
                is_gate=is_gate,
            )
        if not is_gate:
            rank_depth += 1

    return Separation("arbitrary_tiebreak", ARBITRARY_TIEBREAK, 0.0, 0.0, is_gate=False)


def derive_difficulty(situation: Situation, policy: str) -> str | None:
    parted = separation(situation, policy)
    if parted is None:
        return None
    if parted.is_gate:
        return "easy"
    if parted.depth == ARBITRARY_TIEBREAK:
        return "hard"
    clear = parted.relative_gap >= ONE_STEP
    if parted.depth == 0:
        return "easy" if clear else "medium"
    return "medium" if clear else "hard"
