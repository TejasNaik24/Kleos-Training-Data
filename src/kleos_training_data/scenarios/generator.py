"""Turn a scenario into concrete situations, and situations into candidates.

Generation is deterministic: the same scenario file and the same seed produce
byte-identical candidates on any machine. That is what makes a release
reproducible from its source material rather than merely archived.

Sampling walks the axis space rather than drawing independently per axis. Drawing
independently gives you, reliably, forty examples that are all
``urgency=high, evidence=strong`` because those happened to come up — a dataset
that looks varied in its axis *labels* and is uniform in its actual content.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from kleos_training_data.contract.constants import PIPELINE_VERSION
from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.models import Scenario
from kleos_training_data.scenarios.policies import Decision, decide
from kleos_training_data.scenarios.rendering import (
    render_answer,
    render_prompt,
    render_system_prompt,
)
from kleos_training_data.scenarios.situations import Item, Situation
from kleos_training_data.scenarios.surrogates import SurrogatePool

#: Deadline ranges in days, by urgency. Overlapping on purpose: urgency is a
#: description of the situation, not a lookup key for the answer, and a model
#: that could read the deadline off the urgency label would learn nothing.
_URGENCY_DEADLINES: dict[str, tuple[int, ...]] = {
    "critical": (0, 1, 2),
    "high": (1, 2, 3, 5),
    "medium": (4, 7, 10, 14),
    "low": (14, 21, 30, 45),
}

#: Evidence values available at each declared quality level.
_EVIDENCE_BY_QUALITY: dict[str, tuple[str, ...]] = {
    "strong": ("confirmed", "corroborated"),
    "mixed": ("confirmed", "reported", "single_source"),
    "weak": ("single_source", "unverified"),
    "conflicting": ("confirmed", "contradicted", "reported"),
}

_IMPACTS: tuple[str, ...] = ("high", "medium", "low")

#: How far apart the candidates sit, by declared difficulty. Larger means the
#: leader is more clearly ahead, so an easy point is well separated and a hard
#: one is bunched — which is what makes a hard point genuinely hard rather than
#: merely labelled that way.
_DIFFICULTY_SPREAD: dict[str, int] = {"easy": 2, "medium": 1, "hard": 0}

#: Ages used to construct Decision B's stale-explicit case. The floor sits above
#: `STALE_AFTER_DAYS` and the ceiling well below it, so the boundary is crossed
#: unambiguously rather than by a day.
_STALE_FLOOR_DAYS: int = 45
_FRESH_CEILING_DAYS: int = 7

#: Irrelevant context, added when the context_length axis asks for it. Nothing
#: here bears on any decision — that is the test.
_DISTRACTORS: tuple[str, ...] = (
    "The office move is still scheduled for next quarter.",
    "Someone reorganized the shared drive again.",
    "The weekly sync moved to Thursdays.",
    "There is a new expense-reporting tool to migrate to.",
    "The parking permits renew at the end of the month.",
    "A vendor sent another follow-up about their renewal.",
)

_DETAILS: tuple[str, ...] = (
    "Nothing has moved on it in two weeks.",
    "The owner replied but did not commit to a date.",
    "It is blocked on one outstanding answer.",
    "Most of the work is done; it needs a final review.",
    "It was reopened after being marked complete.",
    "There is a draft, unreviewed.",
)


def _deterministic_index(*parts: Any, modulo: int) -> int:
    """A stable index derived from the parts, never from call order.

    Using a seeded RNG advanced per draw would make every value depend on how
    many draws came before it, so adding one axis value would silently re-roll
    the entire catalog. Hashing the coordinates keeps each draw independent.
    """
    if modulo <= 0:
        return 0
    key = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


@dataclass(frozen=True)
class Candidate:
    """A generated example, before it enters staging."""

    situation: Situation
    decision: Decision
    messages: list[dict[str, Any]]
    variation_axes: dict[str, str]
    scenario_family: str
    group_id: str
    perturbation_of: str | None
    perturbation_kind: str | None

    def to_payload(self) -> dict[str, Any]:
        """The contract-shaped payload, with no id yet.

        The id is minted from this payload's content, so it cannot be part of it.
        """
        return {
            "task": self.situation.task,
            "messages": self.messages,
            "variation_axes": dict(self.variation_axes),
        }


def _axis_points(scenario: Scenario) -> Iterator[dict[str, str]]:
    """Yield ``n_base`` axis combinations, spread across the space.

    ``stratified`` guarantees **marginal balance**: across ``n_base`` points each
    axis uses every one of its values ``floor(n/k)`` or ``ceil(n/k)`` times. It
    walks each axis round-robin from a per-axis, hash-derived starting offset, so
    the axes do not all march in lockstep while still covering each one evenly.

    An earlier version drew each axis independently by hash. That is *random*
    sampling wearing the word "stratified", and at small ``n_base`` it skews
    badly: six points over three formats produced five json and one prose, zero
    bullets — 83% of a supposedly format-diverse catalog in one format. A
    dataset that is large and narrow is exactly what the coverage report exists
    to catch, so the generator should not be manufacturing narrowness.

    ``grid`` walks the full cartesian product instead, for a catalog small enough
    to enumerate exhaustively.
    """
    names = sorted(scenario.axes)
    if not names:
        return

    strides = [1]
    for name in names[:-1]:
        strides.append(strides[-1] * len(scenario.axes[name]))

    # One starting offset per axis, stable for a given family and seed.
    offsets = {
        name: _deterministic_index(
            scenario.family, scenario.generation.seed, name, "offset", modulo=len(values)
        )
        for name, values in scenario.axes.items()
    }

    for index in range(scenario.generation.n_base):
        point: dict[str, str] = {}
        for position, name in enumerate(names):
            values = scenario.axes[name]
            if scenario.generation.sampling == "grid":
                chosen = (index // strides[position]) % len(values)
            else:
                chosen = (index + offsets[name]) % len(values)
            point[name] = values[chosen]
        yield point


def _build_items(
    scenario: Scenario, point: dict[str, str], index: int, pool: SurrogatePool
) -> tuple[Item, ...]:
    """Construct the competing items for one axis point."""
    count = scenario.entities.count
    urgency = point.get("urgency", "medium")
    quality = point.get("evidence_quality", "mixed")
    difficulty = point.get("difficulty", "medium")

    deadlines = _URGENCY_DEADLINES.get(urgency, _URGENCY_DEADLINES["medium"])
    evidences = _EVIDENCE_BY_QUALITY.get(quality, _EVIDENCE_BY_QUALITY["mixed"])

    names = pool.names(scenario.family, point_index=index, count=count)

    # Which slots sit outside the active workspace. Rotated by point so the
    # out-of-scope item is not always in the same position — a fixed position
    # would let a model answer by index rather than by reading the marker.
    out_of_scope: set[int] = set()
    if scenario.prompt.framing == "workspace" and scenario.prompt.out_of_scope_count:
        first = _deterministic_index(scenario.family, index, "scope", modulo=count)
        out_of_scope = {
            (first + offset) % count for offset in range(scenario.prompt.out_of_scope_count)
        }

    items: list[Item] = []
    for slot in range(count):
        key = f"item_{chr(ord('a') + slot)}"
        # `difficulty` controls how far apart the candidates are: on an easy
        # point the leader is clearly ahead, on a hard one the field is tight.
        #
        # This was inverted. `spread` staggers each slot's deadline by
        # `slot * spread`, so spread=0 leaves the field bunched — and easy
        # points were the ones getting spread=0, producing the *tightest*
        # fields under the label "easy" while medium points were cleanly
        # separated. The difficulty axis was measuring the opposite of what it
        # named, which is worse than not having it.
        spread = _DIFFICULTY_SPREAD.get(difficulty, 1)
        deadline = (
            deadlines[
                _deterministic_index(scenario.family, index, key, "deadline", modulo=len(deadlines))
            ]
            + slot * spread
        )
        # `out_of_scope_stronger` splits the evidence draw by scope: everything
        # inside the active workspace comes from the weak set, everything outside
        # it from the strong set. That is what makes "nothing here can answer
        # this, but something over there can" an actual property of the
        # situation rather than a claim the answer makes without support.
        if scenario.prompt.out_of_scope_stronger:
            pool_for_slot = (
                _EVIDENCE_BY_QUALITY["strong"]
                if slot in out_of_scope
                else _EVIDENCE_BY_QUALITY["weak"]
            )
        else:
            pool_for_slot = evidences
        evidence = pool_for_slot[
            _deterministic_index(scenario.family, index, key, "evidence", modulo=len(pool_for_slot))
        ]
        impact = _IMPACTS[
            _deterministic_index(scenario.family, index, key, "impact", modulo=len(_IMPACTS))
        ]
        detail = _DETAILS[
            _deterministic_index(scenario.family, index, key, "detail", modulo=len(_DETAILS))
        ]
        # `defer_to_explicit_statement` compares a stated preference against an
        # inferred one. If every record is `confirmed` the scenario asserts the
        # user made several contradictory explicit statements, the policy cannot
        # apply, and it degrades to evidence ranking under an explicit-statement
        # label. The last slot is therefore always inferred, so the contrast the
        # family is named for is present by construction.
        if (
            scenario.expected.policy == "defer_to_explicit_statement"
            and slot == count - 1
            and evidence == "confirmed"
        ):
            evidence = "corroborated"

        # Decision B's legislated case, built rather than hoped for: slot 0 is an
        # explicit statement older than the staleness threshold, slot 1 is a
        # recent corroborated record that contradicts it. Every other slot is
        # drawn normally so the family still varies.
        if scenario.prompt.stale_explicit_conflict:
            if slot == 0:
                evidence, deadline = "confirmed", max(deadline, _STALE_FLOOR_DAYS)
            elif slot == 1:
                evidence, deadline = "corroborated", min(deadline, _FRESH_CEILING_DAYS)

        items.append(
            Item(
                key=key,
                name=names[slot],
                deadline_days=deadline,
                evidence=evidence,
                impact=impact,
                detail=detail,
                scope="out" if slot in out_of_scope else "in",
            )
        )
    # A policy whose *primary* criterion is identical across every candidate has
    # nothing to rank on and falls through to its own tiebreak — which for the
    # reliability and relevance families is recency, i.e. their stated
    # anti_claim. Force a difference on the primary key so the family cannot
    # quietly teach the opposite of what it claims.
    items = _ensure_primary_key_discriminates(scenario, items)
    return tuple(items)


#: The attribute each policy ranks on first. A family whose primary key is
#: constant across all candidates cannot demonstrate its own claim.
_PRIMARY_KEY: dict[str, str] = {
    "rank_by_reliability_over_recency": "evidence",
    "rank_by_relevance_over_recency": "impact",
    "select_by_evidence_need": "evidence",
    "resolve_or_abstain_on_support": "evidence",
}


def _ensure_primary_key_discriminates(scenario: Scenario, items: list[Item]) -> list[Item]:
    """Give the last candidate a different primary-key value when all of them match."""
    key = _PRIMARY_KEY.get(scenario.expected.policy)
    if key is None or len(items) < 2:
        return items
    values = {getattr(item, key) for item in items}
    if len(values) > 1:
        return items

    last = items[-1]
    if key == "evidence":
        grades = ("confirmed", "corroborated", "reported", "single_source", "unverified")
        current = last.evidence
        swap = (
            grades[(grades.index(current) + 1) % len(grades)] if current in grades else "reported"
        )
        items[-1] = Item(**{**last.__dict__, "evidence": swap})
    else:
        levels = ("high", "medium", "low")
        swap = levels[(levels.index(last.impact) + 1) % len(levels)]
        items[-1] = Item(**{**last.__dict__, "impact": swap})
    return items


def build_situation(
    scenario: Scenario, point: dict[str, str], index: int, pool: SurrogatePool
) -> Situation:
    """Construct one situation from an axis point."""
    items = _build_items(scenario, point, index, pool)

    distractor_count = {"short": 0, "medium": 2, "long": 4}.get(
        point.get("context_length", "short"), 0
    )
    # Walk the pool from a per-point offset instead of hashing each slot
    # independently. Independent hashing collides: two slots drawing the same
    # index printed the same sentence twice in one prompt, which happened on 29
    # of 150 examples and reads as a generator artefact rather than as context.
    start = _deterministic_index(scenario.family, index, "distractor", modulo=len(_DISTRACTORS))
    distractors = tuple(
        _DISTRACTORS[(start + slot) % len(_DISTRACTORS)]
        for slot in range(min(distractor_count, len(_DISTRACTORS)))
    )

    # The workspace the prompt names *is* the workspace axis value. These used to
    # be two independent mechanisms — the axis sampled from `axes.workspace`, the
    # rendered name cycled from `prompt.workspace_names` by point index — so they
    # never had to agree and on v0.0.2 they disagreed on all 338 workspace
    # examples: metadata said `Personal` while the prompt said `Startup`. The
    # label described nothing in the text, which makes any workspace-sliced
    # coverage figure or holdout meaningless.
    #
    # Deriving the name from the axis makes the mismatch unrepresentable rather
    # than merely fixed. `workspace_names` remains a fallback for a family that
    # renders a workspace without declaring the axis.
    workspace_name = ""
    if scenario.prompt.framing == "workspace":
        workspace_name = point.get("workspace") or ""
        if not workspace_name and scenario.prompt.workspace_names:
            workspace_name = scenario.prompt.workspace_names[
                index % len(scenario.prompt.workspace_names)
            ]

    return Situation(
        task=scenario.task,
        family=scenario.family,
        point_index=index,
        axes=dict(point),
        items=items,
        distractors=distractors,
        question=scenario.prompt.question,
        framing=scenario.prompt.framing,
        need=scenario.prompt.need,
        workspace_name=workspace_name,
        request_ambiguous=scenario.prompt.request_ambiguous,
        stale_explicit_conflict=scenario.prompt.stale_explicit_conflict,
    )


def _messages(situation: Situation, decision: Decision) -> list[dict[str, Any]]:
    """Assemble the conversation.

    Note the answer is rendered from the *decision*, which was computed from the
    situation — never written alongside the prompt. The two cannot drift.
    """
    return [
        {"role": "system", "content": render_system_prompt(situation)},
        {"role": "user", "content": render_prompt(situation)},
        {"role": "assistant", "content": render_answer(situation, decision)},
    ]


def _rotate_away_from(current: str | None, choices: tuple[str, ...], ordinal: int) -> str:
    """Pick a value from ``choices`` that is not ``current``.

    Every perturbation has to actually perturb. Setting an axis to a fixed value
    silently no-ops whenever the base already holds that value, producing an
    exact duplicate of the base rather than a variant — which inflates the
    example count, adds nothing to consistency testing, and only surfaces later
    as a deduplication hit with no obvious cause.
    """
    alternatives = tuple(value for value in choices if value != current)
    if not alternatives:  # pragma: no cover - choices always has ≥2 members
        raise ScenarioError(f"No alternative to {current!r} among {choices}.")
    return alternatives[ordinal % len(alternatives)]


def _perturb(situation: Situation, kind: str, ordinal: int) -> Situation:
    """Produce a variant that must not change the decision.

    Each kind alters something a policy is not allowed to depend on. Two
    properties are enforced by the caller, not assumed here: the variant must
    reach the same decision, and it must differ from the base in content.
    """
    axes = dict(situation.axes)

    if kind == "evidence_order":
        axes["presentation_order"] = _rotate_away_from(
            axes.get("presentation_order", "as_given"),
            ("as_given", "reversed", "shuffled"),
            ordinal,
        )
        return Situation(**{**situation.__dict__, "axes": axes})

    if kind == "context_order":
        axes["presentation_order"] = _rotate_away_from(
            axes.get("presentation_order", "as_given"), ("shuffled", "reversed"), ordinal
        )
        return Situation(**{**situation.__dict__, "axes": axes})

    if kind == "formatting":
        axes["format"] = _rotate_away_from(
            axes.get("format", "bullets"), ("bullets", "prose", "json"), ordinal
        )
        return Situation(**{**situation.__dict__, "axes": axes})

    if kind == "schema":
        axes["format"] = _rotate_away_from(
            axes.get("format", "bullets"),
            ("slack_thread", "github_issue", "calendar"),
            ordinal,
        )
        return Situation(**{**situation.__dict__, "axes": axes})

    if kind == "irrelevant_context":
        # Draw distractors the base does not already carry, so the variant
        # genuinely gains context rather than repeating itself.
        unused = tuple(d for d in _DISTRACTORS if d not in situation.distractors)
        extra = tuple(unused[(ordinal + i) % len(unused)] for i in range(min(2, len(unused))))
        if not extra:  # pragma: no cover - the pool is larger than any use
            raise ScenarioError("No unused distractor available to add.")
        return Situation(**{**situation.__dict__, "distractors": situation.distractors + extra})

    if kind == "paraphrase":
        rephrased = _rotate_away_from(
            situation.question,
            (
                "Given all that, what should take priority?",
                "Where should I start?",
                "Which of these deserves attention first?",
            ),
            ordinal,
        )
        return Situation(**{**situation.__dict__, "question": rephrased})

    if kind == "length":
        # Shorten if there is anything to shorten, otherwise lengthen. Either
        # way the context length changes, which is what the kind names.
        if situation.distractors:
            return Situation(**{**situation.__dict__, "distractors": situation.distractors[:1]})
        return Situation(
            **{**situation.__dict__, "distractors": (_DISTRACTORS[ordinal % len(_DISTRACTORS)],)}
        )

    raise ScenarioError(  # pragma: no cover - guarded by the model validator
        f"No perturbation implemented for kind {kind!r}."
    )


def _with_derived_difficulty(
    axes: dict[str, str], situation: Situation, policy: str
) -> dict[str, str]:
    """Replace the declared difficulty with the one the situation actually has.

    Leaves the axes untouched when the scenario never declared a difficulty, and
    when the policy has no registered ordering — a label that cannot be computed
    is dropped rather than guessed at.
    """
    from kleos_training_data.scenarios.difficulty import derive_difficulty

    if "difficulty" not in axes:
        return axes
    derived = derive_difficulty(situation, policy)
    if derived is None:
        return {k: v for k, v in axes.items() if k != "difficulty"}
    return {**axes, "difficulty": derived}


def generate(scenario: Scenario, pool: SurrogatePool) -> list[Candidate]:
    """Generate every candidate for a scenario, base examples and perturbations.

    Two invariants are enforced per perturbation, and both are silent failures
    if left unchecked:

    * It must reach the **same decision** as its base. Otherwise the equivalence
      group's members disagree, and consistency testing measures noise.
    * It must **differ in content** from its base. A no-op perturbation is an
      exact duplicate that inflates the example count and contributes nothing,
      surfacing much later as an unexplained deduplication hit.

    Raises:
        ScenarioError: If either invariant is violated.
    """
    from kleos_training_data.ids import example_id

    candidates: list[Candidate] = []
    seen_content: dict[str, str] = {}
    # Prompt-level duplicates are checked separately from full-payload ones. Two
    # candidates can differ in their axes — so their ids differ — while rendering
    # an identical prompt. Normalization matches captures to requests by prompt
    # hash, so such a pair collapses to one candidate and the batch silently
    # shrinks.
    seen_prompts: dict[str, str] = {}

    for index, point in enumerate(_axis_points(scenario)):
        situation = build_situation(scenario, point, index, pool)
        decision = decide(situation, scenario.expected.policy)

        # The difficulty that ships is *derived* from the situation under the
        # policy that resolves it, never the value declared in the axis. The
        # declared value still shapes generation (it staggers deadlines), but it
        # described the example only by coincidence: measured under each policy's
        # own ordering key, v0.0.3's declared labels were anti-correlated with
        # decision difficulty. A label the pipeline computes cannot drift from
        # what it labels — the same reason the training target is computed.
        base_axes = _with_derived_difficulty(
            {**point, "task": scenario.task}, situation, scenario.expected.policy
        )
        base = Candidate(
            situation=situation,
            decision=decision,
            messages=_messages(situation, decision),
            variation_axes=base_axes,
            scenario_family=scenario.family,
            group_id=situation.group_id,
            perturbation_of=None,
            perturbation_kind=None,
        )
        candidates.append(base)

        # Perturbations reference the base by its derived id, which is a function
        # of the base's content — the same function that mints ids at promotion.
        base_id = example_id(base.to_payload())
        seen_content[base_id] = f"{scenario.family} point {index} (base)"
        seen_prompts[_prompt_key(base)] = f"{scenario.family} point {index} (base)"

        for group in scenario.generation.equivalence_groups:
            for ordinal in range(group.count):
                variant = _perturb(situation, group.kind, ordinal)
                variant_decision = decide(variant, scenario.expected.policy)

                if variant_decision.comparable() != decision.comparable():
                    raise ScenarioError(
                        f"{scenario.family}: perturbation {group.kind!r} changed the "
                        f"decision at point {index}.",
                        details={
                            "base": str(decision.comparable()),
                            "perturbed": str(variant_decision.comparable()),
                        },
                        suggestions=[
                            "An equivalence group whose members disagree makes "
                            "consistency testing measure noise instead of stability.",
                            "Either the perturbation is wrong, or it is a genuinely "
                            "different scenario and belongs in its own family.",
                        ],
                    )

                variant_candidate = Candidate(
                    situation=variant,
                    decision=variant_decision,
                    messages=_messages(variant, variant_decision),
                    variation_axes=_with_derived_difficulty(
                        {**variant.axes, "task": scenario.task},
                        variant,
                        scenario.expected.policy,
                    ),
                    scenario_family=scenario.family,
                    # Same group as its base: a perturbation pair straddling the
                    # split boundary is meaningless to compare.
                    group_id=situation.group_id,
                    perturbation_of=base_id,
                    perturbation_kind=group.kind,
                )

                variant_id = example_id(variant_candidate.to_payload())
                if variant_id in seen_content:
                    raise ScenarioError(
                        f"{scenario.family}: perturbation {group.kind!r} at point "
                        f"{index} produced content identical to "
                        f"{seen_content[variant_id]}.",
                        details={"duplicate_id": variant_id, "kind": group.kind},
                        suggestions=[
                            "A perturbation that changes nothing is an exact "
                            "duplicate: it inflates the example count, adds nothing "
                            "to consistency testing, and only surfaces later as an "
                            "unexplained deduplication hit.",
                            "Usually the axis it varies already holds the value it "
                            "sets — see _rotate_away_from in this module.",
                        ],
                    )
                seen_content[variant_id] = (
                    f"{scenario.family} point {index} ({group.kind}#{ordinal})"
                )

                prompt_key = _prompt_key(variant_candidate)
                if prompt_key in seen_prompts:
                    raise ScenarioError(
                        f"{scenario.family}: perturbation {group.kind!r} at point "
                        f"{index} renders the same prompt as "
                        f"{seen_prompts[prompt_key]}.",
                        details={"kind": group.kind, "point": str(index)},
                        suggestions=[
                            "The axes differ, so the ids differ — but normalization "
                            "matches captures to requests by prompt hash, so the "
                            "pair would collapse to one candidate and the batch "
                            "would silently shrink.",
                            "Usually a reordering that lands back on an order the "
                            "base already used.",
                        ],
                    )
                seen_prompts[prompt_key] = (
                    f"{scenario.family} point {index} ({group.kind}#{ordinal})"
                )

                candidates.append(variant_candidate)

    return candidates


def _prompt_key(candidate: Candidate) -> str:
    """Digest of the prompt half of a candidate.

    Matches what normalization keys on, so a collision here is exactly the
    collision that would lose a candidate downstream.
    """
    from kleos_training_data.hashing import canonical_hash

    system, user, _answer = (m["content"] for m in candidate.messages)
    return canonical_hash({"system": system, "user": user})


def generation_fingerprint(scenario: Scenario) -> str:
    """Digest identifying exactly what a scenario would generate.

    Recorded in provenance so a release can be traced to the scenario definition
    that produced it, including the pipeline version — a change in rendering
    produces different text from an unchanged scenario file.
    """
    payload = f"{scenario.model_dump_json()}|{PIPELINE_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
