from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.situations import EVIDENCE_WEIGHTS, Item, Situation

CLOSE_CALL_RELATIVE_MARGIN: Final[float] = 0.15

STALE_AFTER_DAYS: Final[int] = 30

STRONG_CONFLICT_MIN_EVIDENCE: Final[float] = 0.8

CLOSE_CALL_MARGIN: None = None


def relative_separation(top: Item, runner_up: Item) -> float:
    a, b = _score(top), _score(runner_up)
    leader = max(a, b)
    if leader <= 0.0:
        return 0.0
    return abs(a - b) / leader


@dataclass(frozen=True)
class Decision:
    ranking: tuple[str, ...]
    deciding_factor: str
    rationale: tuple[str, ...]
    abstained: bool = False
    resolver: str | None = None

    def comparable(self) -> tuple[tuple[str, ...], str, bool]:
        return (self.ranking, self.deciding_factor, self.abstained)


def _score(item: Item) -> float:
    return item.deadline_score * item.evidence_weight * item.impact_weight


def _deciding_factor(top: Item, runner_up: Item) -> str:
    gaps = {
        "deadline": abs(top.deadline_score - runner_up.deadline_score),
        "evidence": abs(top.evidence_weight - runner_up.evidence_weight),
        "impact": abs(top.impact_weight - runner_up.impact_weight),
    }
    return max(gaps, key=lambda factor: gaps[factor])


def _justify(item: Item, factor: str, sense: str = "due") -> str:
    when = item.time_phrase(sense)
    if factor == "deadline":
        return f"{when}, and the evidence is {item.evidence}"
    if factor == "evidence":
        return f"evidence is {item.evidence} ({item.evidence_phrase}), {when}"
    return f"{item.impact} consequence if it slips, and it is {when}"


def rank_by_deadline_then_evidence(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: ranking needs at least two items.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(situation.items, key=lambda item: (-_score(item), item.key))
    factor = _deciding_factor(ordered[0], ordered[1])
    sense = situation.time_sense
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(_justify(item, factor, sense) for item in ordered),
    )


def rank_or_abstain_when_close(situation: Situation) -> Decision:
    decision = rank_by_deadline_then_evidence(situation)
    top, runner_up = situation.item(decision.ranking[0]), situation.item(decision.ranking[1])

    if relative_separation(top, runner_up) >= CLOSE_CALL_RELATIVE_MARGIN:
        return decision

    weakest = min((top, runner_up), key=lambda item: item.evidence_weight)
    if weakest.evidence_weight >= STRONG_CONFLICT_MIN_EVIDENCE:
        differences = []
        if top.deadline_days != runner_up.deadline_days:
            differences.append("when they are due")
        if top.impact != runner_up.impact:
            differences.append("how much each one costs if it slips")
        detail = (
            f"they differ only on {' and '.join(differences)}"
            if differences
            else "nothing stated separates them at all"
        )
        resolver = (
            f"both are already well established, so there is nothing left to verify — "
            f"{detail}. Tell me which of those matters more to you and this resolves "
            f"immediately"
        )
    else:
        resolver = (
            f"confirm the status of {weakest.name} — it is {weakest.evidence_phrase}, "
            f"and that is the only thing separating the two"
        )

    return Decision(
        ranking=decision.ranking,
        deciding_factor="insufficient_separation",
        rationale=decision.rationale,
        abstained=True,
        resolver=resolver,
    )


def resolve_or_abstain_on_support(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: conflict resolution needs at least two records.",
            details={"item_count": len(situation.items)},
        )

    decision = rank_by_reliability_over_recency(situation)
    top = situation.item(decision.ranking[0])
    runner_up = situation.item(decision.ranking[1])

    scale = max(top.evidence_weight, runner_up.evidence_weight, 1e-9)
    if abs(top.evidence_weight - runner_up.evidence_weight) / scale >= CLOSE_CALL_RELATIVE_MARGIN:
        return decision

    if top.deadline_days != runner_up.deadline_days:
        return decision

    return Decision(
        ranking=decision.ranking,
        deciding_factor="insufficient_separation",
        rationale=decision.rationale,
        abstained=True,
        resolver=(
            f"{top.name} and {runner_up.name} are supported equally well and were "
            f"recorded at the same time, so nothing stored separates them. Tell me "
            f"which one reflects what you actually decided and I will drop the other"
        ),
    )


def rank_by_reliability_over_recency(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: conflict resolution needs at least two records.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(
        situation.items,
        key=lambda item: (-item.evidence_weight, item.deadline_days, item.key),
    )
    top, runner_up = ordered[0], ordered[1]
    factor = "evidence" if top.evidence_weight != runner_up.evidence_weight else "deadline"
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(f"{item.evidence_phrase}; {item.age_phrase}" for item in ordered),
    )


def select_by_evidence_need(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: routing needs at least two candidate sources.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(
        situation.items,
        key=lambda item: (
            -(item.evidence_weight * item.impact_weight),
            item.deadline_days,
            item.key,
        ),
    )
    top, runner_up = ordered[0], ordered[1]
    factor = "evidence" if top.evidence_weight != runner_up.evidence_weight else "impact"
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(
            f"{item.evidence_phrase}; {item.impact} bearing on the question, "
            f"{item.staleness_phrase}"
            for item in ordered
        ),
    )


def rank_by_relevance_over_recency(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: prioritization needs at least two fragments.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(
        situation.items,
        key=lambda item: (-item.impact_weight, item.deadline_days, item.key),
    )
    top, runner_up = ordered[0], ordered[1]
    factor = "impact" if top.impact_weight != runner_up.impact_weight else "deadline"
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(
            f"{item.impact} relevance to the question; {item.age_phrase}" for item in ordered
        ),
    )


def respect_workspace_scope(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: scope reasoning needs at least two items.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(
        situation.items,
        key=lambda item: (not item.in_scope, -_score(item), item.key),
    )
    out_of_scope = [item for item in ordered if not item.in_scope]

    if out_of_scope and len(out_of_scope) < len(ordered):
        return Decision(
            ranking=tuple(item.key for item in ordered),
            deciding_factor="scope",
            rationale=tuple(
                (
                    f"in the active workspace; {item.deadline_phrase}"
                    if item.in_scope
                    else "in another workspace — flagged, not acted on"
                )
                for item in ordered
            ),
        )

    return rank_by_deadline_then_evidence(situation)


def defer_to_explicit_statement(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: conflict resolution needs at least two records.",
            details={"item_count": len(situation.items)},
        )

    def explicit(item: Item) -> bool:
        return item.evidence == "confirmed"

    ordered = sorted(
        situation.items,
        key=lambda item: (not explicit(item), -item.evidence_weight, item.deadline_days, item.key),
    )
    stated = [item for item in ordered if explicit(item)]
    factor = "explicit_statement" if stated and len(stated) < len(ordered) else "evidence"

    if stated and len(stated) < len(ordered):
        leader = ordered[0]
        challengers = [
            i
            for i in situation.items
            if not explicit(i)
            and i.deadline_days < leader.deadline_days
            and i.evidence_weight >= STRONG_CONFLICT_MIN_EVIDENCE
        ]
        if leader.deadline_days >= STALE_AFTER_DAYS and challengers:
            newest = min(challengers, key=lambda i: i.deadline_days)
            return Decision(
                ranking=tuple(item.key for item in ordered),
                deciding_factor="stale_explicit_conflict",
                rationale=tuple(
                    (
                        f"stated outright by you, but {item.age_phrase} — old enough that "
                        f"it may no longer hold"
                        if explicit(item)
                        else f"inferred — {item.evidence_phrase}; {item.age_phrase}"
                    )
                    for item in ordered
                ),
                abstained=True,
                resolver=(
                    f"you told me {leader.name} outright, but that was {leader.age_phrase} "
                    f"and {newest.name} now says otherwise — {newest.evidence_phrase}, "
                    f"{newest.age_phrase}. I am not going to overwrite what you said on "
                    f"my own: is {leader.name} still right?"
                ),
            )

    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(
            (
                f"stated outright by you; {item.age_phrase}"
                if explicit(item)
                else f"inferred — {item.evidence_phrase}; {item.age_phrase}"
            )
            for item in ordered
        ),
    )


def verify_when_evidence_weak(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: this policy needs at least two options.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(situation.items, key=lambda item: (-_score(item), item.key))
    usable = [item for item in ordered if item.evidence_weight >= EVIDENCE_WEIGHTS["reported"]]

    if not usable:
        weakest = min(ordered, key=lambda item: item.evidence_weight)
        return Decision(
            ranking=tuple(item.key for item in ordered),
            deciding_factor="missing_input",
            rationale=tuple(_justify(item, "evidence", situation.time_sense) for item in ordered),
            abstained=True,
            resolver=(
                f"none of these is established well enough to act on — the weakest, "
                f"{weakest.name}, is {weakest.evidence_phrase}. Tell me which of them "
                f"you have actually confirmed, and this becomes answerable"
            ),
        )

    factor = _deciding_factor(ordered[0], ordered[1])
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(_justify(item, factor, situation.time_sense) for item in ordered),
    )


def ask_before_crossing_workspace(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: this policy needs at least two items.",
            details={"item_count": len(situation.items)},
        )

    inside = [i for i in situation.items if i.in_scope]
    outside = [i for i in situation.items if not i.in_scope]

    def adequate(item: Item) -> bool:
        return item.evidence_weight >= EVIDENCE_WEIGHTS["reported"]

    ordered = sorted(
        situation.items,
        key=lambda item: (not item.in_scope, -_score(item), item.key),
    )

    if inside and outside and not any(adequate(i) for i in inside):
        strongest_outside = max(outside, key=lambda i: i.evidence_weight)
        if adequate(strongest_outside):
            return Decision(
                ranking=tuple(item.key for item in ordered),
                deciding_factor="ask_before_crossing",
                rationale=tuple(
                    (
                        f"in {situation.workspace_name}, but {item.evidence_phrase} — "
                        f"not enough to act on"
                        if item.in_scope
                        else f"in another workspace; {item.evidence_phrase}"
                    )
                    for item in ordered
                ),
                abstained=True,
                resolver=(
                    f"nothing in {situation.workspace_name} is established well enough "
                    f"to answer this. {strongest_outside.name} would settle it, but it "
                    f"sits in another workspace — say the word and I will look there"
                ),
            )

    return respect_workspace_scope(situation)


def prefer_least_privilege_source(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: routing needs at least two candidate sources.",
            details={"item_count": len(situation.items)},
        )

    def adequate(item: Item) -> bool:
        return item.evidence_weight >= EVIDENCE_WEIGHTS["reported"]

    ordered = sorted(
        situation.items,
        key=lambda item: (not adequate(item), item.impact_weight, -item.evidence_weight, item.key),
    )
    usable = [item for item in ordered if adequate(item)]
    factor = "least_privilege" if len(usable) > 1 else "evidence"

    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(
            (
                f"{item.impact} reach; {item.evidence_phrase}, {item.staleness_phrase}"
                if adequate(item)
                else f"cannot answer this — {item.evidence_phrase}"
            )
            for item in ordered
        ),
    )


def ask_when_request_ambiguous(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: this policy needs at least two options.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(situation.items, key=lambda item: (-_score(item), item.key))

    if situation.request_ambiguous:
        return Decision(
            ranking=tuple(item.key for item in ordered),
            deciding_factor="request_ambiguous",
            rationale=tuple(_justify(item, "evidence", situation.time_sense) for item in ordered),
            abstained=True,
            resolver=(
                "before I pick, tell me what you actually need — the options here "
                "answer different questions, and choosing between them without "
                "knowing which one you meant would just be a guess"
            ),
        )

    factor = _deciding_factor(ordered[0], ordered[1])
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(_justify(item, factor, situation.time_sense) for item in ordered),
    )


POLICIES: Final[dict[str, Callable[[Situation], Decision]]] = {
    "rank_by_deadline_then_evidence": rank_by_deadline_then_evidence,
    "rank_or_abstain_when_close": rank_or_abstain_when_close,
    "rank_by_reliability_over_recency": rank_by_reliability_over_recency,
    "resolve_or_abstain_on_support": resolve_or_abstain_on_support,
    "select_by_evidence_need": select_by_evidence_need,
    "rank_by_relevance_over_recency": rank_by_relevance_over_recency,
    "respect_workspace_scope": respect_workspace_scope,
    "defer_to_explicit_statement": defer_to_explicit_statement,
    "verify_when_evidence_weak": verify_when_evidence_weak,
    "ask_when_request_ambiguous": ask_when_request_ambiguous,
    "prefer_least_privilege_source": prefer_least_privilege_source,
    "ask_before_crossing_workspace": ask_before_crossing_workspace,
}


def resolve_policy(name: str) -> Callable[[Situation], Decision]:
    try:
        return POLICIES[name]
    except KeyError:
        raise ScenarioError(
            f"Unknown policy {name!r}.",
            details={"registered": ", ".join(sorted(POLICIES))},
            suggestions=[
                "Fix the scenario's `expected.policy` field.",
                "Or register the policy in scenarios/policies.py — a policy is a "
                "research claim, so adding one is a deliberate act.",
            ],
        ) from None


def decide(situation: Situation, policy_name: str) -> Decision:
    return resolve_policy(policy_name)(situation)
