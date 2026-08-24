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

    items: list[Item] = []
    for slot in range(count):
        key = f"item_{chr(ord('a') + slot)}"
        # `difficulty` controls how far apart the candidates are: on an easy
        # point the leader is clearly ahead, on a hard one the field is tight.
        spread = 0 if difficulty == "easy" else 1
        deadline = (
            deadlines[
                _deterministic_index(scenario.family, index, key, "deadline", modulo=len(deadlines))
            ]
            + slot * spread
        )
        evidence = evidences[
            _deterministic_index(scenario.family, index, key, "evidence", modulo=len(evidences))
        ]
        impact = _IMPACTS[
            _deterministic_index(scenario.family, index, key, "impact", modulo=len(_IMPACTS))
        ]
        detail = _DETAILS[
            _deterministic_index(scenario.family, index, key, "detail", modulo=len(_DETAILS))
        ]
        items.append(
            Item(
                key=key,
                name=names[slot],
                deadline_days=deadline,
                evidence=evidence,
                impact=impact,
                detail=detail,
            )
        )
    return tuple(items)


def build_situation(
    scenario: Scenario, point: dict[str, str], index: int, pool: SurrogatePool
) -> Situation:
    """Construct one situation from an axis point."""
    items = _build_items(scenario, point, index, pool)

    distractor_count = {"short": 0, "medium": 2, "long": 4}.get(
        point.get("context_length", "short"), 0
    )
    distractors = tuple(
        _DISTRACTORS[
            _deterministic_index(
                scenario.family, index, slot, "distractor", modulo=len(_DISTRACTORS)
            )
        ]
        for slot in range(distractor_count)
    )

    return Situation(
        task=scenario.task,
        family=scenario.family,
        point_index=index,
        axes=dict(point),
        items=items,
        distractors=distractors,
        question=scenario.prompt.question,
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

        base_axes = {**point, "task": scenario.task}
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
                    variation_axes={**variant.axes, "task": scenario.task},
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
