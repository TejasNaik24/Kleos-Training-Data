from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from kleos_training_data.errors import ScenarioError

EVIDENCE_WEIGHTS: Final[dict[str, float]] = {
    "confirmed": 1.0,
    "corroborated": 0.8,
    "reported": 0.55,
    "single_source": 0.35,
    "unverified": 0.2,
    "contradicted": 0.1,
}

IMPACT_WEIGHTS: Final[dict[str, float]] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}

EVIDENCE_PHRASES: Final[dict[str, str]] = {
    "confirmed": "confirmed directly by the owner",
    "corroborated": "corroborated by two independent sources",
    "reported": "reported in the weekly update",
    "single_source": "mentioned once, by a single source",
    "unverified": "unverified — nobody has checked it",
    "contradicted": "contradicted by a later message",
}


TIME_SENSES: Final[tuple[str, ...]] = ("due", "age", "staleness")


@dataclass(frozen=True)
class Framing:
    name: str
    system: str
    noun: str
    time_sense: str
    time_field: str
    evidence_label: str = "evidence"
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
    return resolve_framing(situation.framing)


@dataclass(frozen=True)
class Item:
    key: str
    name: str
    deadline_days: int
    evidence: str
    impact: str
    detail: str
    scope: str = "in"

    @property
    def in_scope(self) -> bool:
        return self.scope != "out"

    @property
    def deadline_score(self) -> float:
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
    task: str
    family: str
    point_index: int
    axes: dict[str, str]
    items: tuple[Item, ...]
    distractors: tuple[str, ...] = ()
    question: str = "Which should I deal with first, and why?"
    framing: str = "priority"
    need: str = ""
    workspace_name: str = ""
    request_ambiguous: bool = False
    stale_explicit_conflict: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_sense(self) -> str:
        return resolve_framing(self.framing).time_sense

    def item(self, key: str) -> Item:
        for candidate in self.items:
            if candidate.key == key:
                return candidate
        raise KeyError(f"no item {key!r} in situation {self.family}:{self.point_index}")

    def presented(self) -> tuple[Item, ...]:
        order = self.axes.get("presentation_order", "as_given")
        if order == "reversed":
            return tuple(reversed(self.items))
        if order == "shuffled":
            indices = sorted(
                range(len(self.items)),
                key=lambda i: ((i * 7 + self.point_index * 13) % max(len(self.items), 1), i),
            )
            forbidden = [list(range(len(self.items))), list(reversed(range(len(self.items))))]
            rotations = 0
            while indices in forbidden and rotations < len(indices):
                indices = indices[1:] + indices[:1]
                rotations += 1
            return tuple(self.items[i] for i in indices)
        return self.items

    @property
    def group_id(self) -> str:
        return f"{self.family}:{self.point_index:04d}"
