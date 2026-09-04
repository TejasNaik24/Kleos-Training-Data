"""How hard a decision actually is, measured under the policy that resolves it.

``difficulty`` used to be an *authored* label. A scenario declared
``difficulty: [easy, medium, hard]``, the generator staggered candidate deadlines
by a per-level constant, and the label shipped as metadata. Nothing checked that
the label described the example.

It did not. Three things were wrong at once, and the third hid the first two:

1. **The knob moved the wrong thing.** Staggering deadlines is the only effect
   ``difficulty`` ever had. But of the ten registered policies, seven do not rank
   on deadline at all — they rank on evidence weight, impact weight, scope, or
   adequacy — and the remaining three fold deadline into a *product* with two
   other factors. For ``prefer_least_privilege_source`` the knob was completely
   inert: every difficulty level produced an identical median margin.

2. **The label was confounded with family composition.** Which families declare
   which levels varies with their evidence and impact axes, so the label tracked
   the family more than the situation.

3. **The audit metric was measured with one key for every policy** — the
   priority score — on the ``json`` subset only. That is the right key for three
   of ten policies and the wrong key for the rest.

Measured correctly, on the whole corpus and under each policy's own ordering,
v0.0.3's labels were **anti-correlated** with decision difficulty: 71% of
``easy`` points had nothing visible separating the top two candidates, against
36% of ``hard`` ones. The clean ``easy > medium > hard`` curve reported for
v0.0.2 was an artefact of measuring the wrong thing on a quarter of the data.

So difficulty is now **derived, not declared** — the same treatment the training
target already gets. A policy is a lexicographic sort; how hard its decision is
is how much of that sort the reader must consult before the top two separate:

Difficulty then rises with **how deep you must look** and falls with **how far
apart the candidates are** when you get there:

``easy``
    A stated categorical rule settles it — one candidate is out of scope, or
    inadequate, or inferred rather than stated — **or** they separate on the
    primary ranking criterion by at least one full step.
``medium``
    They separate on the primary criterion but by **less than one step**, or you
    had to consult a **secondary** criterion and it separated them clearly.
``hard``
    You had to consult a secondary criterion and it separated them only
    narrowly, **or** nothing stated separates them at all and the order rests on
    an arbitrary tiebreak nobody reading the prompt could reproduce.

**Gates are not ranking criteria.** Three policies begin with a boolean gate —
``scope``, ``adequate``, ``explicit``. A gate *excludes*; it does not order. Two
candidates on the same side of a gate have not "tied on the thing that matters",
the gate simply did not apply to them, so the comparison moves on. Treating a
shared gate value as a tie was the first version of this module, and it labelled
all four workspace families 100% ``hard`` — an artefact of the definition rather
than a property of the data. A gate that *does* differ is the clearest possible
separation: a stated rule decided it outright.

**One step** is not a tuned number. It is the smallest relative gap between two
adjacent grades on the evidence scale (0.20, between ``confirmed`` and
``corroborated``) — the finest distinction this dataset's own vocabulary can
express. Two candidates closer than that are closer than the data can describe.
It was not adjusted after seeing any distribution.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Final

from kleos_training_data.scenarios.situations import EVIDENCE_WEIGHTS, Item, Situation

#: Components that *gate* rather than rank. A gate is boolean: it removes
#: candidates from contention. Two candidates on the same side of one have not
#: tied on a ranking criterion, so the comparison moves past it; two candidates
#: on opposite sides were separated by a stated rule, which is the clearest
#: separation available.
GATE_COMPONENTS: Final[frozenset[str]] = frozenset({"scope", "adequate", "explicit"})

#: Ordered components of each policy's sort key, highest-is-best. Mirrors the
#: `sorted(...)` calls in policies.py. A policy missing here cannot have its
#: difficulty derived, which `validate_scenarios.py` reports rather than
#: silently labelling the examples anyway.
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

#: The finest distinction this dataset's vocabulary can draw, taken from the
#: evidence scale itself rather than chosen: the smallest relative gap between
#: two adjacent evidence grades. Candidates closer than this are closer than the
#: data can describe, so the decision between them is narrow by definition.
#:
#: Derived, not tuned. It was not adjusted after seeing the resulting
#: distribution, and a test asserts it still equals the scale's minimum step.
ONE_STEP: Final[float] = 0.20

DIFFICULTY_LEVELS: Final[tuple[str, ...]] = ("easy", "medium", "hard")


def _score(item: Item) -> float:
    """The priority score, mirrored from policies.py to avoid a circular import."""
    return item.deadline_score * item.evidence_weight * item.impact_weight


def evidence_scale_min_step() -> float:
    """Smallest relative gap between adjacent evidence grades.

    :data:`ONE_STEP` must equal this. Kept as a function so a test can assert the
    constant was derived from the scale rather than picked.
    """
    grades = sorted(EVIDENCE_WEIGHTS.values(), reverse=True)
    return min((a - b) / a for a, b in pairwise(grades))


class Separation:
    """Where, and by how much, the top two candidates part company."""

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
        #: How many *ranking* criteria were tied before this one separated them.
        #: Gates do not count — they filter rather than order.
        self.depth = depth
        self.gap = gap
        self.relative_gap = relative_gap
        #: True when a stated categorical rule, not a ranking criterion, decided it.
        self.is_gate = is_gate

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Separation(component={self.component!r}, depth={self.depth}, "
            f"relative_gap={self.relative_gap:.3f})"
        )


#: Marks a separation that never happened: the candidates were identical on every
#: stated criterion and the order came from the arbitrary key tiebreak.
ARBITRARY_TIEBREAK: Final[int] = 99


def separation(situation: Situation, policy: str) -> Separation | None:
    """How the top two candidates separate under ``policy``'s own ordering.

    Gate components are skipped when both candidates share a value and reported
    when they do not. Returns ``None`` when the policy has no registered ordering
    or the situation has fewer than two candidates — never a guess.
    """
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
            # Tied on a ranking criterion: the reader must look one level deeper.
            rank_depth += 1

    return Separation("arbitrary_tiebreak", ARBITRARY_TIEBREAK, 0.0, 0.0, is_gate=False)


def derive_difficulty(situation: Situation, policy: str) -> str | None:
    """The difficulty this situation actually has, or ``None`` if unmeasurable.

    Never consults the declared axis value. The label ships only if it was
    computed from the situation the model will see.
    """
    parted = separation(situation, policy)
    if parted is None:
        return None
    if parted.is_gate:
        # A stated rule settled it outright: this one is out of scope, or cannot
        # answer, or was inferred rather than stated. Nothing to weigh.
        return "easy"
    if parted.depth == ARBITRARY_TIEBREAK:
        return "hard"
    clear = parted.relative_gap >= ONE_STEP
    if parted.depth == 0:
        return "easy" if clear else "medium"
    return "medium" if clear else "hard"
