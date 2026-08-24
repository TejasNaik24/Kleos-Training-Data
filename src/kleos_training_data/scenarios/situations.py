"""The structured situation a scenario point describes.

A scenario is not a conversation with holes in it. It is a *situation*: a set of
competing items with deadlines, evidence and consequences, plus some irrelevant
context. The conversation is rendered from that structure, and the correct answer
is *computed* from it by a registered policy.

This indirection is the whole point. If the training target were authored by hand
alongside the prompt, nothing would stop the two drifting apart, and nothing
could check that a paraphrase of the prompt still deserves the same answer.
Because the answer is derived, ``validate_scenarios.py`` can assert that every
perturbation in an equivalence group genuinely preserves the decision — which is
what "equivalence" has to mean if consistency testing is to measure anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

#: Evidence strength, strongest first. The labels are what appears in the
#: rendered prompt; the weights are what the policy reasons over.
EVIDENCE_WEIGHTS: Final[dict[str, float]] = {
    "confirmed": 1.0,
    "corroborated": 0.8,
    "reported": 0.55,
    "single_source": 0.35,
    "unverified": 0.2,
    "contradicted": 0.1,
}

#: Consequence of getting it wrong or being late.
IMPACT_WEIGHTS: Final[dict[str, float]] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}

#: Human-readable evidence phrasing, used by the renderers.
EVIDENCE_PHRASES: Final[dict[str, str]] = {
    "confirmed": "confirmed directly by the owner",
    "corroborated": "corroborated by two independent sources",
    "reported": "reported in the weekly update",
    "single_source": "mentioned once, by a single source",
    "unverified": "unverified — nobody has checked it",
    "contradicted": "contradicted by a later message",
}


@dataclass(frozen=True)
class Item:
    """One competing thing the model has to reason about.

    ``key`` is a stable slot label (``item_a``) that survives surrogate
    substitution and reordering; ``name`` is the fictional display name that
    appears in the text. Keeping them separate is what lets a decision be
    compared across a paraphrase or a reorder — the ranking is over keys, and the
    names are free to change.
    """

    key: str
    name: str
    deadline_days: int
    evidence: str
    impact: str
    detail: str

    @property
    def deadline_score(self) -> float:
        """Nearer deadlines score higher, with diminishing urgency over time.

        ``1 / (1 + days)`` rather than a linear scale: the difference between
        "today" and "in three days" genuinely matters more than the difference
        between "in 30 days" and "in 33 days".
        """
        return 1.0 / (1.0 + max(self.deadline_days, 0))

    @property
    def evidence_weight(self) -> float:
        return EVIDENCE_WEIGHTS.get(self.evidence, 0.2)

    @property
    def impact_weight(self) -> float:
        return IMPACT_WEIGHTS.get(self.impact, 0.3)

    @property
    def deadline_phrase(self) -> str:
        """How the deadline reads in a prompt."""
        if self.deadline_days <= 0:
            return "due today"
        if self.deadline_days == 1:
            return "due tomorrow"
        if self.deadline_days <= 7:
            return f"due in {self.deadline_days} days"
        if self.deadline_days <= 30:
            return f"due in about {self.deadline_days // 7} weeks"
        return f"due in about {self.deadline_days // 30} months"

    @property
    def evidence_phrase(self) -> str:
        return EVIDENCE_PHRASES.get(self.evidence, self.evidence)


@dataclass(frozen=True)
class Situation:
    """One sampled point from a scenario's axis space.

    Everything needed to render a prompt and derive the correct answer, and
    nothing tied to a real person.
    """

    task: str
    family: str
    point_index: int
    axes: dict[str, str]
    items: tuple[Item, ...]
    distractors: tuple[str, ...] = ()
    question: str = "Which should I deal with first, and why?"
    metadata: dict[str, Any] = field(default_factory=dict)

    def item(self, key: str) -> Item:
        """Look up an item by its stable key."""
        for candidate in self.items:
            if candidate.key == key:
                return candidate
        raise KeyError(f"no item {key!r} in situation {self.family}:{self.point_index}")

    def presented(self) -> tuple[Item, ...]:
        """Items in the order the prompt shows them.

        Presentation order is a variation axis precisely so that a model which
        learned "pick the first one" can be caught. The decision must not depend
        on this.
        """
        order = self.axes.get("presentation_order", "as_given")
        if order == "reversed":
            return tuple(reversed(self.items))
        if order == "shuffled":
            # Deterministic, seeded by the point so it is reproducible.
            indices = sorted(
                range(len(self.items)),
                key=lambda i: ((i * 7 + self.point_index * 13) % max(len(self.items), 1), i),
            )
            # A shuffle that lands on the identity *or* the reversed permutation
            # renders text byte-identical to one of the other two orders. The
            # axis would say the order changed while the prompt did not, so a
            # `context_order` perturbation of a `reversed` base would be an exact
            # duplicate of it — and, sharing a request hash, one would silently
            # overwrite the other at normalization.
            #
            # Guarding only against the identity is not enough: that was the
            # first attempt, and the collision that survived it was a shuffle
            # equal to `reversed` on a base that was already reversed.
            forbidden = [list(range(len(self.items))), list(reversed(range(len(self.items))))]
            rotations = 0
            while indices in forbidden and rotations < len(indices):
                indices = indices[1:] + indices[:1]
                rotations += 1
            return tuple(self.items[i] for i in indices)
        return self.items

    @property
    def group_id(self) -> str:
        """Splitting key: a base example and its perturbations share this.

        Consistency evaluation compares members of one group, so a group
        straddling the train/test boundary would make the comparison meaningless.
        """
        return f"{self.family}:{self.point_index:04d}"
