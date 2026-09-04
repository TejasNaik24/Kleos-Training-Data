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

from kleos_training_data.errors import ScenarioError

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


#: What an item's ``deadline_days`` measures, per framing. The number is the
#: same; what it *means* is not, and rendering it as a due date inside a
#: memory-conflict scenario produced "recorded as of in 4 days" — a record
#: written in the future.
TIME_SENSES: Final[tuple[str, ...]] = ("due", "age", "staleness")


@dataclass(frozen=True)
class Framing:
    """What the items in a situation *are*, and how to talk about them.

    A framing is not decoration. The same structure — competing entries with a
    number, an evidence grade and a consequence — is a set of deadlines in one
    task and a set of candidate sources in another, and the two need different
    instructions, different nouns and a different reading of the number.

    Collapsing all seven tasks onto one framing is what produced tool_routing
    examples whose system prompt said "You help prioritize competing work" and
    whose candidate sources carried due dates. The task label said one thing and
    the content taught another.
    """

    name: str
    #: The system instruction. States the policy, never the answer.
    system: str
    #: Singular noun for one item, used by the renderers.
    noun: str
    #: What ``deadline_days`` measures here: "due", "age" or "staleness".
    time_sense: str
    #: Column label for the number in structured renderings.
    time_field: str
    #: How the evidence grade reads in this framing.
    evidence_label: str = "evidence"
    #: How the impact grade reads in this framing.
    impact_label: str = "impact"


FRAMINGS: Final[dict[str, Framing]] = {
    "priority": Framing(
        name="priority",
        system=(
            "You help prioritize competing work. Rank the items by where delay costs "
            "the most. That is a combination of three things, not a checklist in "
            "order: how near the deadline is, how well the claim is actually "
            "supported, and how much the outcome matters. A later deadline can still "
            "come first when it is better supported or costlier to miss. Name the "
            "factor that decided it. If the top options are too close to separate, "
            "say so and name what would resolve it instead of guessing."
        ),
        noun="item",
        time_sense="due",
        time_field="due_in_days",
    ),
    "briefing": Framing(
        name="briefing",
        system=(
            "You write a short briefing. Lead with what needs a decision soonest and "
            "is best supported; everything else is context and goes after it. Name the "
            "factor that decided the order. If the top items are too close to separate "
            "on the evidence available, say so rather than inventing a ranking."
        ),
        noun="update",
        time_sense="due",
        time_field="due_in_days",
    ),
    "routing": Framing(
        name="routing",
        system=(
            "You decide which source to consult to answer a question. Choose by how "
            "directly a source bears on what was asked, not by how powerful or how "
            "recently updated it is — a stale but directly relevant source beats a "
            "fresh irrelevant one. Name the factor that decided it. If nothing "
            "available can answer the question, say what is missing instead of "
            "guessing."
        ),
        noun="source",
        time_sense="staleness",
        time_field="last_synced_days_ago",
        evidence_label="bearing_on_question",
        impact_label="coverage",
    ),
    "memory": Framing(
        name="memory",
        system=(
            "You resolve conflicts between stored records. Prefer the better-supported "
            "record over the merely newer one; recency decides only when two records "
            "are supported equally well, and how much rides on the record breaks a "
            "remaining tie. An explicit statement from the user outranks anything "
            "inferred, unless it is clearly stale and newer corroborated evidence "
            "contradicts it — then surface the conflict rather than overwriting what "
            "was said. Name the factor that decided it, and say plainly when the "
            "records cannot be reconciled from what is stored."
        ),
        noun="record",
        time_sense="age",
        time_field="recorded_days_ago",
        evidence_label="support",
        impact_label="stakes",
    ),
    "context": Framing(
        name="context",
        system=(
            "You choose which context to keep when not all of it fits. Keep what bears "
            "on the question; recency only breaks ties between fragments of equal "
            "relevance, and a fresh irrelevant note displaces a useful one. Name the "
            "factor that decided it."
        ),
        noun="fragment",
        time_sense="age",
        time_field="last_touched_days_ago",
        evidence_label="support",
        impact_label="relevance",
    ),
    "workspace": Framing(
        name="workspace",
        system=(
            "You work inside the workspace the user named. Workspaces are context "
            "boundaries: answer from the active one, and when something relevant sits "
            "in another workspace, say so explicitly rather than acting on it or "
            "silently ignoring it. Do not carry content across a boundary the user "
            "did not open. Name the factor that decided it."
        ),
        noun="item",
        time_sense="due",
        time_field="due_in_days",
    ),
}


def resolve_framing(name: str) -> Framing:
    """Look up a framing, with an actionable error when it is unknown."""
    try:
        return FRAMINGS[name]
    except KeyError:
        raise ScenarioError(
            f"Unknown framing {name!r}.",
            details={"registered": ", ".join(sorted(FRAMINGS))},
            suggestions=[
                "Fix the scenario's `prompt.framing` field.",
                "A framing decides what the items are and what the model is told "
                "to do with them, so adding one is a deliberate act.",
            ],
        ) from None


def framing_of(situation: Situation) -> Framing:
    """The framing a situation renders under."""
    return resolve_framing(situation.framing)


@dataclass(frozen=True)
class Item:
    """One competing thing the model has to reason about.

    ``key`` is a stable slot label (``item_a``) that survives surrogate
    substitution and reordering; ``name`` is the fictional display name that
    appears in the text. Keeping them separate is what lets a decision be
    compared across a paraphrase or a reorder — the ranking is over keys, and the
    names are free to change.

    ``scope`` says whether the item sits inside the workspace the user named.
    It is a field rather than a phrase sniffed out of ``detail`` because
    :func:`respect_workspace_scope` used to look for the substring "out of
    scope", no generated detail ever contained it, and the policy's entire
    scope branch was therefore unreachable — every workspace_reasoning example
    silently fell through to plain deadline ranking under a workspace label.
    """

    key: str
    name: str
    deadline_days: int
    evidence: str
    impact: str
    detail: str
    #: "in" when the item belongs to the workspace under discussion, "out"
    #: when it belongs to another one.
    scope: str = "in"

    @property
    def in_scope(self) -> bool:
        return self.scope != "out"

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

    @staticmethod
    def _plural(count: int, noun: str) -> str:
        return f"{count} {noun}" if count == 1 else f"{count} {noun}s"

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
            return f"due in about {self._plural(self.deadline_days // 7, 'week')}"
        return f"due in about {self._plural(self.deadline_days // 30, 'month')}"

    @property
    def age_phrase(self) -> str:
        """How the same number reads when it measures how old a record is.

        Memory-conflict and context families reason over record age, not over a
        due date. Rendering age with the deadline phrasing produced "recorded as
        of in 4 days", which describes a record written in the future.
        """
        if self.deadline_days <= 0:
            return "recorded today"
        if self.deadline_days == 1:
            return "recorded yesterday"
        if self.deadline_days <= 7:
            return f"recorded {self._plural(self.deadline_days, 'day')} ago"
        if self.deadline_days <= 30:
            return f"recorded about {self._plural(self.deadline_days // 7, 'week')} ago"
        return f"recorded about {self._plural(self.deadline_days // 30, 'month')} ago"

    @property
    def staleness_phrase(self) -> str:
        """How the same number reads when it measures how stale a source is."""
        if self.deadline_days <= 0:
            return "synced just now"
        if self.deadline_days == 1:
            return "last synced yesterday"
        if self.deadline_days <= 7:
            return f"last synced {self._plural(self.deadline_days, 'day')} ago"
        if self.deadline_days <= 30:
            return f"last synced about {self._plural(self.deadline_days // 7, 'week')} ago"
        return f"last synced about {self._plural(self.deadline_days // 30, 'month')} ago"

    def time_phrase(self, sense: str) -> str:
        """The time phrase appropriate to a framing's sense of ``deadline_days``."""
        if sense == "age":
            return self.age_phrase
        if sense == "staleness":
            return self.staleness_phrase
        return self.deadline_phrase

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
    #: Which registered framing renders this situation. A framing decides what
    #: the items *are* — competing work, candidate sources, stored records,
    #: context fragments — and therefore what the system prompt instructs and
    #: how time reads. One framing for all seven tasks meant a tool_routing
    #: example was told "You help prioritize competing work" and listed its
    #: candidate sources with due dates.
    framing: str = "priority"
    #: The information need, for framings where the question needs an
    #: antecedent. "Which of these should I check to answer that?" has no
    #: referent unless something states what "that" is.
    need: str = ""
    #: The workspace under discussion, for the workspace framing.
    workspace_name: str = ""
    #: True when the *request itself* does not determine an answer — the user
    #: asked for "the latest numbers" without saying which. This is a property of
    #: the question, not of the evidence, and it is why referential
    #: underspecification needs its own policy: no amount of verifying the
    #: candidates fixes a referent nobody has pinned down.
    request_ambiguous: bool = False
    #: See PromptSpec.stale_explicit_conflict.
    stale_explicit_conflict: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_sense(self) -> str:
        """What ``deadline_days`` measures here, from this situation's framing."""
        return resolve_framing(self.framing).time_sense

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
