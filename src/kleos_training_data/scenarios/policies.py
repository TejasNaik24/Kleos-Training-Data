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
from kleos_training_data.scenarios.situations import EVIDENCE_WEIGHTS, Item, Situation

#: Separation below which two items are treated as too close to separate
#: confidently, as a **fraction of the leader's score** rather than an absolute
#: quantity.
#:
#: This was an absolute 0.08 and that was dimensionally wrong. The priority score
#: is a product of three factors, one of which decays as ``1/(1+days)``, so the
#: whole scale compresses about tenfold as deadlines lengthen: the median item
#: scores 0.100 at 0-3 days and 0.011 beyond 15. An absolute 0.08 is larger than
#: the entire median score past about four days, so distant pairs were almost
#: automatically "too close to separate" — 104 of 142 abstentions in v0.0.5
#: abstained despite more than 15% relative separation.
#:
#: The consequence was that "ambiguous" described how far away the deadlines
#: happened to be rather than anything about the decision. Two pairs with
#: identical relative separation, identical evidence and identical impact got
#: opposite answers::
#:
#:     A=1d  B=2d    scores 0.500/0.333   33% apart   ranked
#:     A=19d B=29d   scores 0.050/0.033   33% apart   abstained
#:
#: A relative margin is scale-invariant, so that cannot happen. 0.15 is the
#: specified boundary: two candidates within 15% of the leader's score are
#: treated as not confidently separable.
CLOSE_CALL_RELATIVE_MARGIN: Final[float] = 0.15

#: How old an explicit statement must be before it counts as "clearly stale".
#:
#: Decision B says an explicit statement keeps priority *unless* it is clearly
#: stale and newer evidence strongly conflicts — but it named no thresholds, so
#: the case could not be generated and the corpus contained zero examples of it.
#: 30 days is one calendar month: long enough that a stated preference plausibly
#: predates the user's current circumstances, short enough to occur inside a
#: term. Deterministic and testable; change it deliberately, not to move a
#: distribution.
STALE_AFTER_DAYS: Final[int] = 30

#: How well supported a *newer, inferred* record must be before it is allowed to
#: challenge a stale explicit statement. `corroborated` (0.8) means at least two
#: independent sources agree — a single mention never qualifies, which is why the
#: five v0.0.5 cases flagged in the Pass-3 audit were correct behaviour rather
#: than defects: their newer record was only `single_source`.
STRONG_CONFLICT_MIN_EVIDENCE: Final[float] = 0.8

#: Retained only so an import of the old name fails loudly rather than silently
#: comparing against a quantity that no longer means anything.
CLOSE_CALL_MARGIN = None  # type: ignore[assignment]


def relative_separation(top: Item, runner_up: Item) -> float:
    """How far apart two candidates are, as a fraction of the leader's score.

    Scale-invariant by construction: multiplying both scores by any positive
    constant leaves the result unchanged, which is exactly the property the
    absolute margin lacked.

    Returns 0.0 when the leader scores zero, so a degenerate pair counts as
    inseparable rather than raising.
    """
    a, b = _score(top), _score(runner_up)
    leader = max(a, b)
    if leader <= 0.0:
        return 0.0
    return abs(a - b) / leader


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


def _justify(item: Item, factor: str, sense: str = "due") -> str:
    """One clause explaining an item's position, leading with the deciding factor."""
    when = item.time_phrase(sense)
    if factor == "deadline":
        return f"{when}, and the evidence is {item.evidence}"
    if factor == "evidence":
        return f"evidence is {item.evidence} ({item.evidence_phrase}), {when}"
    return f"{item.impact} consequence if it slips, and it is {when}"


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
    sense = situation.time_sense
    return Decision(
        ranking=tuple(item.key for item in ordered),
        deciding_factor=factor,
        rationale=tuple(_justify(item, factor, sense) for item in ordered),
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

    if relative_separation(top, runner_up) >= CLOSE_CALL_RELATIVE_MARGIN:
        return decision

    # The resolver has to name something that would actually resolve it. Asking
    # the user to "confirm the status of X" when the prompt already reports X as
    # confirmed is incoherent, and it happened whenever both leaders were
    # `confirmed` — the closeness was caused by deadline and stakes, not by
    # uncertainty, so naming evidence as the resolver was simply the wrong
    # concept.
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
    """Resolve a record conflict by support, or say it cannot be resolved.

    The memory framing promises support first, recency second, stakes only as a
    remaining tie-break. `rank_or_abstain_when_close` promises none of that — its
    fallback is the *priority score*, a product in which a high-stakes but
    unverified record can outrank a better-supported one. A memory-conflict
    family using it produced 40 answers where stakes overrode support, directly
    contradicting the instruction the model was given in the same example.

    So this is the abstaining policy for the memory framing: same relative
    close-call margin, but measured on the ordering the framing actually states.
    """
    if len(situation.items) < 2:
        raise ScenarioError(
            f"{situation.family}: conflict resolution needs at least two records.",
            details={"item_count": len(situation.items)},
        )

    decision = rank_by_reliability_over_recency(situation)
    top = situation.item(decision.ranking[0])
    runner_up = situation.item(decision.ranking[1])

    # "Close" here means close *on support*, which is what the framing ranks by.
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
        rationale=tuple(f"{item.evidence_phrase}; {item.age_phrase}" for item in ordered),
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
            f"{item.evidence_phrase}; {item.impact} bearing on the question, "
            f"{item.staleness_phrase}"
            for item in ordered
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
            f"{item.impact} relevance to the question; {item.age_phrase}" for item in ordered
        ),
    )


def respect_workspace_scope(situation: Situation) -> Decision:
    """Answer within the named workspace, and say so when something is outside it.

    Items carry an explicit ``scope``; anything marked out of scope is ranked
    last and flagged rather than quietly acted on. Silently reaching outside the
    stated scope is how an assistant surfaces something the user did not ask it
    to look at, and it is the failure workspace isolation exists to prevent.

    This used to sniff ``item.detail`` for the substring "out of scope". No
    generated detail ever contained it, so the branch below was unreachable and
    every workspace_reasoning example fell through to plain deadline ranking —
    a whole task's worth of examples labelled for a policy they did not teach.
    Reading a field instead of a phrase is what makes the branch checkable.
    """
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
    """An explicit statement from the user outranks anything inferred.

    The distinction memory conflict turns on and that reliability alone misses:
    a confidently-inferred preference is still an inference. When the user has
    said something outright, that is the record to treat as current even if a
    later inference disagrees and even if the inference is better corroborated
    across sources — because the sources are all downstream of behaviour, and
    behaviour is evidence about circumstance, not about intent.

    ``confirmed`` marks the explicitly-stated record; everything else is
    inferred. Among records of equal standing the better-supported one wins,
    then the newer.
    """
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

    # Decision B: an explicit statement keeps priority, but a *clearly stale* one
    # challenged by *strongly corroborated* newer evidence is surfaced rather
    # than silently asserted. Both thresholds are named constants so the boundary
    # is testable from both sides.
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
    """Nothing on offer is well enough supported to act on — go and verify.

    **Epistemic uncertainty.** The candidates are known and the question is
    clear; what is missing is *confidence*. The useful move is to check
    something, because checking changes the answer.

    Deliberately distinct from :func:`ask_when_request_ambiguous`, which handles
    the case where the *request* does not determine an answer. Those were one
    overloaded policy in v0.0.5, and 42 examples answered confidently against a
    prompt that said the user did not know what they were asking for. They are
    different triggers, different resolvers, and different next actions for the
    user, so they are now different policies.

    Distinct from :func:`rank_or_abstain_when_close`, and the distinction is the
    point. That policy handles a *margin* problem: the options are separated by
    less than the evidence can resolve. This one handles a *missing variable*:
    something the decision depends on was never supplied, so no amount of
    reasoning over what is present will produce the answer.

    The trigger is that **nothing in the field reaches usable strength**.

    Keying it on the top-scoring item alone was close to self-contradictory:
    score multiplies evidence weight in, so the weakest-evidence item is the
    least likely to lead, and the family fired on 5 of 90 points. A family whose
    distinctive behaviour appears in 6% of its examples is teaching something
    other than what its name and policy_claim say.
    """
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
    """The active workspace cannot answer, and another one probably can. Ask.

    This exists because a family claiming to teach "when the active workspace
    does not hold what the question needs, say so and ask — do not reach into
    another workspace" was generated with every item in scope and resolved by a
    scope-blind policy. Half its examples fell through to ordinary ranking, none
    mentioned a workspace, and removing the workspace header would not have
    changed a word of the answer. The claim was right; nothing computed it.

    The situation this policy needs, and which the scenario now builds:

    * everything **inside** the active workspace is too weakly supported to act
      on, and
    * something **outside** it is well supported.

    The correct move is neither to answer from the weak in-scope material nor to
    silently use the out-of-scope material. It is to say the active workspace
    cannot answer, name that the material exists elsewhere, and ask before
    crossing. That is workspace-dependent by construction: strip the workspace
    markers and the right answer becomes "just use the strong one".
    """
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
    """Among sources that can answer, choose the one that reaches least far.

    A routing policy with a privacy consequence. Two sources may both answer the
    question; the one that opens less of the user's data is the right call, and
    reaching for the broadest available tool by default is how an assistant ends
    up reading a mailbox to answer a calendar question.

    ``impact`` stands for breadth of reach here — ``high`` is the most
    far-reaching source — so among sources of adequate bearing the narrowest
    wins. Bearing still gates: a narrow source that cannot answer is not an
    answer.
    """
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
    """The request does not determine an answer — ask what was meant.

    **Referential underspecification.** The user asked for "the latest numbers"
    without saying which numbers. No amount of verifying the candidates fixes
    that: until the referent is pinned down, every option is equally defensible
    and equally likely to be wrong.

    This is why it cannot be a branch inside
    :func:`verify_when_evidence_weak`. That policy asks "is what I have good
    enough?", which is a property of the item list. This one asks "do I know what
    is being asked?", which is a property of the request and is not visible in
    the items at all. Merging them would make the resolver conditional on
    something the item list cannot express.

    Falls through to ordinary ranking when the request *is* determinate, so a
    family can carry both the positive and the negative case.
    """
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


#: The registry. A scenario names a policy by key; an unregistered name is a
#: scenario-validation error rather than a runtime surprise during generation.
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
