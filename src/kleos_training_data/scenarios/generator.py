from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from kleos_training_data.contract.constants import PIPELINE_VERSION
from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.models import Scenario
from kleos_training_data.scenarios.policies import Decision, decide
from kleos_training_data.scenarios.reasoning import render_reasoning
from kleos_training_data.scenarios.rendering import (
    render_answer,
    render_prompt,
    render_system_prompt,
)
from kleos_training_data.scenarios.situations import Item, Situation
from kleos_training_data.scenarios.surrogates import SurrogatePool

_URGENCY_DEADLINES: dict[str, tuple[int, ...]] = {
    "critical": (0, 1, 2),
    "high": (1, 2, 3, 5),
    "medium": (4, 7, 10, 14),
    "low": (14, 21, 30, 45),
}

_EVIDENCE_BY_QUALITY: dict[str, tuple[str, ...]] = {
    "strong": ("confirmed", "corroborated"),
    "mixed": ("confirmed", "reported", "single_source"),
    "weak": ("single_source", "unverified"),
    "conflicting": ("confirmed", "contradicted", "reported"),
}

_IMPACTS: tuple[str, ...] = ("high", "medium", "low")

_DIFFICULTY_SPREAD: dict[str, int] = {"easy": 2, "medium": 1, "hard": 0}

_STALE_FLOOR_DAYS: int = 45
_FRESH_CEILING_DAYS: int = 7

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
    if modulo <= 0:
        return 0
    key = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


@dataclass(frozen=True)
class Candidate:
    situation: Situation
    decision: Decision
    messages: list[dict[str, Any]]
    variation_axes: dict[str, str]
    scenario_family: str
    group_id: str
    perturbation_of: str | None
    perturbation_kind: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "task": self.situation.task,
            "messages": self.messages,
            "variation_axes": dict(self.variation_axes),
        }


def _axis_points(scenario: Scenario) -> Iterator[dict[str, str]]:
    names = sorted(scenario.axes)
    if not names:
        return

    strides = [1]
    for name in names[:-1]:
        strides.append(strides[-1] * len(scenario.axes[name]))

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
    count = scenario.entities.count
    urgency = point.get("urgency", "medium")
    quality = point.get("evidence_quality", "mixed")
    difficulty = point.get("difficulty", "medium")

    deadlines = _URGENCY_DEADLINES.get(urgency, _URGENCY_DEADLINES["medium"])
    evidences = _EVIDENCE_BY_QUALITY.get(quality, _EVIDENCE_BY_QUALITY["mixed"])

    names = pool.names(scenario.family, point_index=index, count=count)

    out_of_scope: set[int] = set()
    if scenario.prompt.framing == "workspace" and scenario.prompt.out_of_scope_count:
        first = _deterministic_index(scenario.family, index, "scope", modulo=count)
        out_of_scope = {
            (first + offset) % count for offset in range(scenario.prompt.out_of_scope_count)
        }

    items: list[Item] = []
    for slot in range(count):
        key = f"item_{chr(ord('a') + slot)}"
        spread = _DIFFICULTY_SPREAD.get(difficulty, 1)
        deadline = (
            deadlines[
                _deterministic_index(scenario.family, index, key, "deadline", modulo=len(deadlines))
            ]
            + slot * spread
        )
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
        if (
            scenario.expected.policy == "defer_to_explicit_statement"
            and slot == count - 1
            and evidence == "confirmed"
        ):
            evidence = "corroborated"

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
    items = _ensure_primary_key_discriminates(scenario, items)
    return tuple(items)


_PRIMARY_KEY: dict[str, str] = {
    "rank_by_reliability_over_recency": "evidence",
    "rank_by_relevance_over_recency": "impact",
    "select_by_evidence_need": "evidence",
    "resolve_or_abstain_on_support": "evidence",
}


def _ensure_primary_key_discriminates(scenario: Scenario, items: list[Item]) -> list[Item]:
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
    items = _build_items(scenario, point, index, pool)

    distractor_count = {"short": 0, "medium": 2, "long": 4}.get(
        point.get("context_length", "short"), 0
    )
    start = _deterministic_index(scenario.family, index, "distractor", modulo=len(_DISTRACTORS))
    distractors = tuple(
        _DISTRACTORS[(start + slot) % len(_DISTRACTORS)]
        for slot in range(min(distractor_count, len(_DISTRACTORS)))
    )

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
    assistant: dict[str, Any] = {"role": "assistant", "content": render_answer(situation, decision)}
    if situation.axes.get("format", "bullets") != "json":
        assistant["reasoning"] = render_reasoning(situation, decision)
    return [
        {"role": "system", "content": render_system_prompt(situation)},
        {"role": "user", "content": render_prompt(situation)},
        assistant,
    ]


def _rotate_away_from(current: str | None, choices: tuple[str, ...], ordinal: int) -> str:
    alternatives = tuple(value for value in choices if value != current)
    if not alternatives:
        raise ScenarioError(f"No alternative to {current!r} among {choices}.")
    return alternatives[ordinal % len(alternatives)]


def _perturb(situation: Situation, kind: str, ordinal: int) -> Situation:
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
        unused = tuple(d for d in _DISTRACTORS if d not in situation.distractors)
        extra = tuple(unused[(ordinal + i) % len(unused)] for i in range(min(2, len(unused))))
        if not extra:
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
        if situation.distractors:
            return Situation(**{**situation.__dict__, "distractors": situation.distractors[:1]})
        return Situation(
            **{**situation.__dict__, "distractors": (_DISTRACTORS[ordinal % len(_DISTRACTORS)],)}
        )

    raise ScenarioError(f"No perturbation implemented for kind {kind!r}.")


def _with_derived_difficulty(
    axes: dict[str, str], situation: Situation, policy: str
) -> dict[str, str]:
    from kleos_training_data.scenarios.difficulty import derive_difficulty

    if "difficulty" not in axes:
        return axes
    derived = derive_difficulty(situation, policy)
    if derived is None:
        return {k: v for k, v in axes.items() if k != "difficulty"}
    return {**axes, "difficulty": derived}


def generate(scenario: Scenario, pool: SurrogatePool) -> list[Candidate]:
    from kleos_training_data.ids import example_id

    candidates: list[Candidate] = []
    seen_content: dict[str, str] = {}
    seen_prompts: dict[str, str] = {}

    for index, point in enumerate(_axis_points(scenario)):
        situation = build_situation(scenario, point, index, pool)
        decision = decide(situation, scenario.expected.policy)

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
    from kleos_training_data.hashing import canonical_hash

    system, user, _answer = (m["content"] for m in candidate.messages)
    return canonical_hash({"system": system, "user": user})


def generation_fingerprint(scenario: Scenario) -> str:
    payload = f"{scenario.model_dump_json()}|{PIPELINE_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
