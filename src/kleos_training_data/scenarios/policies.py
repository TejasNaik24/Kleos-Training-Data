from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Final

from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.situations import EVIDENCE_WEIGHTS, Item, Situation

CLOSE_CALL_RELATIVE_MARGIN: Final[float] = 0.15

STALE_AFTER_DAYS: Final[int] = 30

STRONG_CONFLICT_MIN_EVIDENCE: Final[float] = 0.8

CLOSE_CALL_MARGIN: None = None

RULES: Final[tuple[str, ...]] = (
    "score",
    "reliability",
    "bearing",
    "relevance",
    "scope",
    "explicit",
    "least_privilege",
)

BASES: Final[tuple[str, ...]] = (
    "largest_gap",
    "first_difference",
    "scope",
    "explicit",
    "least_privilege",
    "decline",
)

CHECK_NAMES: Final[tuple[str, ...]] = (
    "close_call",
    "both_strong",
    "support_gap",
    "same_age",
    "usable",
    "ambiguous",
    "crossing",
    "mixed_scope",
    "stale_explicit",
    "strong_challenger",
    "usable_count",
)


def relative_separation(top: Item, runner_up: Item) -> float:
    a, b = _score(top), _score(runner_up)
    leader = max(a, b)
    if leader <= 0.0:
        return 0.0
    return abs(a - b) / leader


@dataclass(frozen=True)
class Check:
    name: str
    measured: float
    threshold: float
    triggered: bool
    subjects: tuple[str, ...] = ()


@dataclass(frozen=True)
class Trace:
    rule: str
    scored: bool
    basis: str
    checks: tuple[Check, ...] = ()
    gaps: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Decision:
    ranking: tuple[str, ...]
    deciding_factor: str
    rationale: tuple[str, ...]
    abstained: bool = False
    resolver: str | None = None
    trace: Trace | None = field(default=None, compare=False, repr=False)

    def comparable(self) -> tuple[tuple[str, ...], str, bool]:
        return (self.ranking, self.deciding_factor, self.abstained)


def _score(item: Item) -> float:
    return item.deadline_score * item.evidence_weight * item.impact_weight


def _gaps(top: Item, runner_up: Item) -> tuple[tuple[str, float], ...]:
    return (
        ("deadline", abs(top.deadline_score - runner_up.deadline_score)),
        ("evidence", abs(top.evidence_weight - runner_up.evidence_weight)),
        ("impact", abs(top.impact_weight - runner_up.impact_weight)),
    )


def _deciding_factor(top: Item, runner_up: Item) -> str:
    gaps = dict(_gaps(top, runner_up))
    return max(gaps, key=lambda factor: gaps[factor])


def _score_trace(ordered: list[Item], checks: tuple[Check, ...] = ()) -> Trace:
    return Trace(
        rule="score",
        scored=True,
        basis="largest_gap",
        checks=checks,
        gaps=_gaps(ordered[0], ordered[1]),
    )


def _extend(trace: Trace | None, checks: tuple[Check, ...] = (), **changes: Any) -> Trace:
    if trace is None:
        raise ScenarioError("A wrapped policy returned a decision without a trace.")
    return replace(trace, checks=trace.checks + checks, **changes)


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
        trace=_score_trace(ordered),
    )


def rank_or_abstain_when_close(situation: Situation) -> Decision:
    decision = rank_by_deadline_then_evidence(situation)
    top, runner_up = situation.item(decision.ranking[0]), situation.item(decision.ranking[1])

    separation = relative_separation(top, runner_up)
    close = Check(
        name="close_call",
        measured=separation,
        threshold=CLOSE_CALL_RELATIVE_MARGIN,
        triggered=separation < CLOSE_CALL_RELATIVE_MARGIN,
        subjects=(top.key, runner_up.key),
    )
    if separation >= CLOSE_CALL_RELATIVE_MARGIN:
        return replace(decision, trace=_extend(decision.trace, (close,)))

    weakest = min((top, runner_up), key=lambda item: item.evidence_weight)
    other = runner_up if weakest is top else top
    strong = Check(
        name="both_strong",
        measured=weakest.evidence_weight,
        threshold=STRONG_CONFLICT_MIN_EVIDENCE,
        triggered=weakest.evidence_weight >= STRONG_CONFLICT_MIN_EVIDENCE,
        subjects=(weakest.key, other.key),
    )
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
        trace=_extend(decision.trace, (close, strong), basis="decline"),
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
    gap = abs(top.evidence_weight - runner_up.evidence_weight) / scale
    support = Check(
        name="support_gap",
        measured=gap,
        threshold=CLOSE_CALL_RELATIVE_MARGIN,
        triggered=gap < CLOSE_CALL_RELATIVE_MARGIN,
        subjects=(top.key, runner_up.key),
    )
    if gap >= CLOSE_CALL_RELATIVE_MARGIN:
        return replace(decision, trace=_extend(decision.trace, (support,)))

    same_age = Check(
        name="same_age",
        measured=float(abs(top.deadline_days - runner_up.deadline_days)),
        threshold=0.0,
        triggered=top.deadline_days == runner_up.deadline_days,
        subjects=(top.key, runner_up.key),
    )
    if top.deadline_days != runner_up.deadline_days:
        return replace(decision, trace=_extend(decision.trace, (support, same_age)))

    return Decision(
        ranking=decision.ranking,
        deciding_factor="insufficient_separation",
        rationale=decision.rationale,
        abstained=True,
        trace=_extend(decision.trace, (support, same_age), basis="decline"),
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
        trace=Trace(rule="reliability", scored=False, basis="first_difference"),
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
        trace=Trace(rule="bearing", scored=True, basis="first_difference"),
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
        trace=Trace(rule="relevance", scored=False, basis="first_difference"),
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
    mixed = Check(
        name="mixed_scope",
        measured=float(len(out_of_scope)),
        threshold=1.0,
        triggered=bool(out_of_scope) and len(out_of_scope) < len(ordered),
        subjects=tuple(item.key for item in out_of_scope),
    )

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
            trace=Trace(rule="scope", scored=True, basis="scope", checks=(mixed,)),
        )

    fallback = rank_by_deadline_then_evidence(situation)
    return replace(fallback, trace=_extend(fallback.trace, (mixed,)))


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
    checks: tuple[Check, ...] = ()

    if stated and len(stated) < len(ordered):
        leader = ordered[0]
        challengers = [
            i
            for i in situation.items
            if not explicit(i)
            and i.deadline_days < leader.deadline_days
            and i.evidence_weight >= STRONG_CONFLICT_MIN_EVIDENCE
        ]
        strongest_newer = min(challengers, key=lambda i: i.deadline_days) if challengers else None
        checks = (
            Check(
                name="stale_explicit",
                measured=float(leader.deadline_days),
                threshold=float(STALE_AFTER_DAYS),
                triggered=leader.deadline_days >= STALE_AFTER_DAYS,
                subjects=(leader.key,),
            ),
            Check(
                name="strong_challenger",
                measured=strongest_newer.evidence_weight if strongest_newer else 0.0,
                threshold=STRONG_CONFLICT_MIN_EVIDENCE,
                triggered=bool(challengers),
                subjects=(strongest_newer.key,) if strongest_newer else (),
            ),
        )
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
                trace=Trace(rule="explicit", scored=False, basis="decline", checks=checks),
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
        trace=Trace(
            rule="explicit",
            scored=False,
            basis="explicit" if factor == "explicit_statement" else "first_difference",
            checks=checks,
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
    weakest_overall = min(ordered, key=lambda item: item.evidence_weight)
    usable_check = Check(
        name="usable",
        measured=max(item.evidence_weight for item in ordered),
        threshold=EVIDENCE_WEIGHTS["reported"],
        triggered=not usable,
        subjects=(weakest_overall.key,) if not usable else (),
    )

    if not usable:
        weakest = min(ordered, key=lambda item: item.evidence_weight)
        return Decision(
            ranking=tuple(item.key for item in ordered),
            deciding_factor="missing_input",
            rationale=tuple(_justify(item, "evidence", situation.time_sense) for item in ordered),
            abstained=True,
            trace=_extend(_score_trace(ordered), (usable_check,), basis="decline"),
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
        trace=_score_trace(ordered, (usable_check,)),
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
            crossing = Check(
                name="crossing",
                measured=strongest_outside.evidence_weight,
                threshold=EVIDENCE_WEIGHTS["reported"],
                triggered=True,
                subjects=(strongest_outside.key,),
            )
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
                trace=Trace(rule="scope", scored=True, basis="decline", checks=(crossing,)),
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
    count = Check(
        name="usable_count",
        measured=float(len(usable)),
        threshold=2.0,
        triggered=len(usable) > 1,
        subjects=tuple(item.key for item in usable),
    )

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
        trace=Trace(
            rule="least_privilege",
            scored=False,
            basis="least_privilege" if len(usable) > 1 else "first_difference",
            checks=(count,),
        ),
    )


def ask_when_request_ambiguous(situation: Situation) -> Decision:
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: this policy needs at least two options.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(situation.items, key=lambda item: (-_score(item), item.key))
    ambiguous = Check(
        name="ambiguous",
        measured=1.0 if situation.request_ambiguous else 0.0,
        threshold=1.0,
        triggered=situation.request_ambiguous,
    )

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
            trace=_extend(_score_trace(ordered), (ambiguous,), basis="decline"),
        )

    factor = _deciding_factor(ordered[0], ordered[1])
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(_justify(item, factor, situation.time_sense) for item in ordered),
        trace=_score_trace(ordered, (ambiguous,)),
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
