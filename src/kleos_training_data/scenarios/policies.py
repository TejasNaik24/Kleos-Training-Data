"""Registered decision policies — the thing a training example teaches.

Each policy is a pure function from a :class:`Situation` to a :class:`Decision`.
This is where the research claim lives: a policy is a *rule*, stated once, that
applies to any situation of its shape. The training targets are rendered from
its output, so the dataset teaches the rule rather than a collection of answers
that happen to be correct.

Three consequences follow, and all three are load-bearing:

1. **No self-reinforcement.** The target is never what KLEOS said. A model's own
   output becoming its next training target is how mistakes get amplified into
   policy (spec section 45).

2. **Perturbations are checkable.** Because the answer is computed, the scenario
   validator can assert that a paraphrase or a reorder yields the *same*
   decision. A perturbation that changes the answer is not a perturbation, and
   silently including one makes consistency testing measure noise.

3. **The rationale is derived too.** A rendered answer says which factor decided
   it, taken from the same computation — so the stated reason cannot drift from
   the actual reason.

A policy must be deterministic and must not read anything outside the situation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.situations import Item, Situation

#: Margin below which two items are treated as too close to separate confidently.
#: Not a tuning knob — it defines what "ambiguous" means for the abstain policy,
#: and changing it changes what the dataset teaches.
CLOSE_CALL_MARGIN: Final[float] = 0.08


@dataclass(frozen=True)
class Decision:
    """What a policy concluded, in a form that can be compared and rendered."""

    #: Item keys, best first. The unit of comparison across perturbations.
    ranking: tuple[str, ...]
    #: The factor that decided the top choice: "deadline", "evidence", "impact".
    deciding_factor: str
    #: One short justification per ranked item, in ranking order.
    rationale: tuple[str, ...]
    #: True when the policy declines to rank confidently.
    abstained: bool = False
    #: What would resolve the ambiguity, when abstaining.
    resolver: str | None = None

    def comparable(self) -> tuple[tuple[str, ...], str, bool]:
        """The part of a decision that a perturbation must preserve.

        Deliberately excludes the rationale text: a paraphrase is allowed to
        change how the reason reads, and required not to change what it is.
        """
        return (self.ranking, self.deciding_factor, self.abstained)


def _score(item: Item) -> float:
    """Combined priority score.

    Multiplicative, not additive: a confirmed-but-distant item and an
    urgent-but-unverified item should both be discounted, and a sum would let a
    single strong factor mask a weak one. Multiplication is what makes "urgent
    but unverified" rank below "slightly less urgent and confirmed", which is the
    policy this dataset exists to teach.
    """
    return item.deadline_score * item.evidence_weight * item.impact_weight


def _deciding_factor(top: Item, runner_up: Item) -> str:
    """Which factor actually separated the top two.

    Computed by asking what would have happened had each factor been equal, so
    the stated reason is the real one rather than a plausible-sounding guess.
    """
    gaps = {
        "deadline": abs(top.deadline_score - runner_up.deadline_score),
        "evidence": abs(top.evidence_weight - runner_up.evidence_weight),
        "impact": abs(top.impact_weight - runner_up.impact_weight),
    }
    return max(gaps, key=lambda factor: gaps[factor])


def _justify(item: Item, factor: str) -> str:
    """One clause explaining an item's position, leading with the deciding factor."""
    if factor == "deadline":
        return f"{item.deadline_phrase}, and the evidence is {item.evidence}"
    if factor == "evidence":
        return f"evidence is {item.evidence} ({item.evidence_phrase}), {item.deadline_phrase}"
    return f"{item.impact} consequence if it slips, and it is {item.deadline_phrase}"


def rank_by_deadline_then_evidence(situation: Situation) -> Decision:
    """Rank by deadline proximity weighted by evidence strength and consequence.

    The policy in one sentence: **prioritize the item where delay has the largest
    expected cost**, which is nearness of deadline discounted by how well the
    claim is actually supported and by how much the outcome matters.

    Ties break on the item key so the result is stable regardless of presentation
    order — a policy that depended on ordering would be teaching a position bias.
    """
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: ranking needs at least two items.",
            details={"item_count": len(situation.items)},
        )

    ordered = sorted(situation.items, key=lambda item: (-_score(item), item.key))
    factor = _deciding_factor(ordered[0], ordered[1])
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(_justify(item, factor) for item in ordered),
    )


def rank_or_abstain_when_close(situation: Situation) -> Decision:
    """Rank, but decline to commit when the top two are too close to separate.

    Teaches the harder half of the policy: when urgency is high but the evidence
    does not separate the options, the right move is to *get the missing
    evidence*, not to produce a confident ranking anyway. A dataset that only
    ever shows clean decisions teaches a model that a decision is always
    available.
    """
    decision = rank_by_deadline_then_evidence(situation)
    top, runner_up = situation.item(decision.ranking[0]), situation.item(decision.ranking[1])

    if abs(_score(top) - _score(runner_up)) >= CLOSE_CALL_MARGIN:
        return decision

    weakest = min((top, runner_up), key=lambda item: item.evidence_weight)
    return Decision(
        ranking=decision.ranking,
        deciding_factor="insufficient_separation",
        rationale=decision.rationale,
        abstained=True,
        resolver=(
            f"confirm the status of {weakest.name} — it is {weakest.evidence_phrase}, "
            f"and that is the only thing separating the two"
        ),
    )


def rank_by_reliability_over_recency(situation: Situation) -> Decision:
    """Prefer the better-supported record over the merely newer one.

    The deliberate inverse of "always trust the latest update". A model that
    learned recency as a surface rule gets this wrong, which is the point:
    ``metadata.deadline_days`` stands in for record age here, and the policy
    ignores it whenever evidence quality differs materially.
    """
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
        rationale=tuple(
            f"{item.evidence_phrase}; recorded {item.deadline_phrase.replace('due', 'as of')}"
            for item in ordered
        ),
    )


def select_by_evidence_need(situation: Situation) -> Decision:
    """Route to the source that can actually answer the question.

    The tool-routing policy: pick by *what kind of evidence the question needs*,
    not by which tool is most capable in general. A web search is not better than
    a calendar lookup — it is worse, when the question is "when is this due".

    Reuses the evidence weighting: an item's ``evidence`` stands for how directly
    that source bears on the question, and ``deadline_days`` for how stale its
    data is. A stale but directly-relevant source still beats a fresh irrelevant
    one, which is the whole lesson.
    """
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
            f"{item.evidence_phrase}; {item.impact} bearing on the question" for item in ordered
        ),
    )


def rank_by_relevance_over_recency(situation: Situation) -> Decision:
    """Keep the most relevant context, not the most recent.

    The inverse of the obvious heuristic, and the point of the family: a model
    that learned "keep the newest" fails whenever the newest thing is irrelevant.
    Recency only breaks ties between items of equal relevance.

    ``impact`` stands for relevance here and ``deadline_days`` for age.
    """
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
            f"{item.impact} relevance to the question; {item.deadline_phrase.replace('due', 'last touched')}"
            for item in ordered
        ),
    )


def respect_workspace_scope(situation: Situation) -> Decision:
    """Answer within the named workspace, and say so when something is outside it.

    ``entities`` on the axis names the workspace in scope; an item whose
    ``detail`` marks it out of scope is ranked last and flagged, rather than
    quietly answered anyway. Silently reaching outside the stated scope is how an
    assistant surfaces something the user did not ask it to look at.
    """
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: scope reasoning needs at least two items.",
            details={"item_count": len(situation.items)},
        )

    def in_scope(item: Item) -> bool:
        return "out of scope" not in item.detail.lower()

    ordered = sorted(
        situation.items,
        key=lambda item: (not in_scope(item), -_score(item), item.key),
    )
    out_of_scope = [item for item in ordered if not in_scope(item)]

    if out_of_scope and len(out_of_scope) < len(ordered):
        return Decision(
            ranking=tuple(item.key for item in ordered),
            deciding_factor="scope",
            rationale=tuple(
                (
                    f"in this workspace; {item.deadline_phrase}"
                    if in_scope(item)
                    else "outside the workspace you named — not acted on"
                )
                for item in ordered
            ),
        )

    decision = rank_by_deadline_then_evidence(situation)
    return decision


#: The registry. A scenario names a policy by key; an unregistered name is a
#: scenario-validation error rather than a runtime surprise during generation.
POLICIES: Final[dict[str, Callable[[Situation], Decision]]] = {
    "rank_by_deadline_then_evidence": rank_by_deadline_then_evidence,
    "rank_or_abstain_when_close": rank_or_abstain_when_close,
    "rank_by_reliability_over_recency": rank_by_reliability_over_recency,
    "select_by_evidence_need": select_by_evidence_need,
    "rank_by_relevance_over_recency": rank_by_relevance_over_recency,
    "respect_workspace_scope": respect_workspace_scope,
}


def resolve_policy(name: str) -> Callable[[Situation], Decision]:
    """Look up a policy, with an actionable error when it is unknown."""
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
    """Apply a named policy to a situation."""
    return resolve_policy(policy_name)(situation)
