from __future__ import annotations

import hashlib
from typing import Final

from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.policies import (
    CLOSE_CALL_RELATIVE_MARGIN,
    STALE_AFTER_DAYS,
    Check,
    Decision,
    Trace,
)
from kleos_training_data.scenarios.situations import Item, Situation, framing_of

__all__ = ["MAX_REASONING_CHARS", "render_reasoning"]

MAX_REASONING_CHARS: Final[int] = 1200

_GRADE: Final[dict[str, str]] = {
    "confirmed": "confirmed",
    "corroborated": "corroborated",
    "reported": "reported",
    "single_source": "single source",
    "unverified": "unverified",
    "contradicted": "contradicted",
}

_RULE_LINES: Final[dict[str, tuple[str, ...]]] = {
    "score": (
        "Score each {noun} as deadline × evidence × impact, where a deadline of n days "
        "counts 1 / (1 + n).",
        "Multiply deadline, evidence and impact for each {noun}; a nearer deadline counts "
        "for more.",
    ),
    "scope": (
        "Scope comes first: only the active workspace counts, then each {noun} scores "
        "deadline × evidence × impact.",
    ),
    "bearing": (
        "Score each {noun} as evidence × impact, which is how directly it bears on the question.",
        "Multiply evidence and impact for each {noun}; freshness only breaks a tie.",
    ),
    "reliability": (
        "Rank by support first; recency only breaks a tie.",
        "Better support wins over newer; age matters only between equals.",
    ),
    "relevance": (
        "Rank by relevance first; recency only breaks a tie.",
        "Keep what bears on the question; age matters only between equals.",
    ),
    "explicit": (
        "Something stated outright outranks anything inferred, unless it has gone stale.",
    ),
    "least_privilege": (
        "Among sources that can answer, prefer the narrowest reach.",
        "Use the narrowest source that can answer; broader reach is not a reason to open it.",
    ),
}

_LARGEST_GAP: Final[tuple[str, ...]] = (
    "largest gap between {top} and {runner_up}: {factor} ({a} vs {b}).",
    "the biggest difference between {top} and {runner_up} is {factor} ({a} vs {b}).",
)

_CLOSE_CALL: Final[tuple[str, ...]] = (
    "{top} leads {runner_up} by {pct}, {side} the 15% needed to call it.",
    "the gap between {top} and {runner_up} is {pct}, {side} the 15% needed to call it.",
)


def _pick(options: tuple[str, ...], situation: Situation, step: str) -> str:
    key = "|".join(
        [
            situation.group_id,
            situation.question,
            situation.axes.get("presentation_order", "as_given"),
            situation.axes.get("format", "bullets"),
            step,
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _percent(value: float, threshold: float) -> str:
    whole = round(value * 100)
    if whole == round(threshold * 100):
        return f"{value * 100:.1f}%"
    return f"{whole}%"


def _names(situation: Situation, keys: tuple[str, ...]) -> str:
    names = [situation.item(key).name for key in keys]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _product(item: Item, rule: str) -> float:
    if rule == "bearing":
        return item.evidence_weight * item.impact_weight
    return item.deadline_score * item.evidence_weight * item.impact_weight


def _item_line(item: Item, situation: Situation, trace: Trace) -> str:
    f = framing_of(situation)
    time = item.time_phrase(f.time_sense)
    evidence = f"{_GRADE.get(item.evidence, item.evidence)} ({item.evidence_weight:.2f})"
    impact = f"{item.impact} {f.impact_label} ({item.impact_weight:.2f})"
    rule = trace.rule
    if rule in ("score", "scope"):
        parts = [f"{time} ({item.deadline_score:.2f})", evidence, impact]
    elif rule == "bearing":
        parts = [evidence, impact, time]
    elif rule == "relevance":
        parts = [impact, time]
    elif rule == "least_privilege":
        parts = [evidence, impact]
    else:
        parts = [evidence, time]
    if situation.framing == "workspace":
        parts.append("inside the workspace" if item.in_scope else "outside the workspace")
    tail = f" → {_product(item, rule):.3f}" if trace.scored else ""
    return f"- {item.name}: {', '.join(parts)}{tail}."


def _check_line(check: Check, situation: Situation) -> str | None:
    subjects = [situation.item(key) for key in check.subjects]
    side = "below" if check.triggered else "at or above"
    if check.name == "close_call":
        top, runner_up = subjects
        if check.measured == 0.0:
            return (
                f"{top.name} and {runner_up.name} score the same, below the 15% needed to call it."
            )
        return _pick(_CLOSE_CALL, situation, "close_call").format(
            top=top.name,
            runner_up=runner_up.name,
            pct=_percent(check.measured, CLOSE_CALL_RELATIVE_MARGIN),
            side=side,
        )
    if check.name == "both_strong":
        if check.triggered:
            return "both are well supported (0.80 or more), so there is nothing left to verify."
        weakest, other = subjects
        if weakest.evidence_weight == other.evidence_weight:
            return (
                f"{weakest.name} and {other.name} are equally supported "
                f"({check.measured:.2f}), short of 0.80, so confirming {weakest.name} "
                f"would settle this."
            )
        return (
            f"{weakest.name} is the weaker of the two ({check.measured:.2f}), "
            f"so confirming it would settle this."
        )
    if check.name == "support_gap":
        top, runner_up = subjects
        if check.measured == 0.0:
            return f"{top.name} and {runner_up.name} have the same support, below the 15% needed."
        pct = _percent(check.measured, CLOSE_CALL_RELATIVE_MARGIN)
        return f"support differs by {pct} between {top.name} and {runner_up.name}, {side} the 15% needed."
    if check.name == "same_age":
        if check.triggered:
            return "they were also recorded at the same time, so nothing stored separates them."
        return "they were recorded at different times, so recency settles it."
    if check.name == "usable":
        if check.triggered:
            return f"nothing reaches usable evidence (0.55): the best is {check.measured:.2f}."
        return "at least one option has usable evidence (0.55 or more)."
    if check.name == "ambiguous":
        if check.triggered:
            return (
                "the request does not say which of these it needs, so any ranking would be a guess."
            )
        return "the request is clear about what it needs."
    if check.name == "crossing":
        return (
            f"nothing inside {situation.workspace_name} reaches usable evidence (0.55), "
            f"but {subjects[0].name} outside it does ({check.measured:.2f})."
        )
    if check.name == "mixed_scope":
        if situation.framing != "workspace":
            return None
        if check.triggered:
            verb = "sits" if len(subjects) == 1 else "sit"
            return (
                f"{_names(situation, check.subjects)} {verb} outside "
                f"{situation.workspace_name}, so scope comes first."
            )
        where = "outside" if subjects else "inside"
        return f"everything is {where} {situation.workspace_name}, so scope does not separate them."
    if check.name == "stale_explicit":
        leader = subjects[0]
        window = "past" if check.triggered else "inside"
        return (
            f"{leader.name} was stated outright and was {leader.age_phrase}, {window} the "
            f"{STALE_AFTER_DAYS}-day window."
        )
    if check.name == "strong_challenger":
        if check.triggered:
            return (
                f"{subjects[0].name} is newer and well supported ({check.measured:.2f}), "
                f"so it contradicts what was said."
            )
        return "no newer, well-supported record contradicts it."
    if check.name == "usable_count":
        count = int(check.measured)
        if count == 0:
            return "no source can answer this (usable evidence 0.55 or more)."
        noun = "source can" if count == 1 else "sources can"
        return f"{count} {noun} answer this (usable evidence 0.55 or more)."
    raise ScenarioError(f"No reasoning phrasing for check {check.name!r}.")


def _factor_values(
    top: Item, runner_up: Item, factor: str, situation: Situation
) -> tuple[str, str]:
    if factor == "deadline":
        sense = framing_of(situation).time_sense
        if sense == "due":
            return f"{top.deadline_score:.2f}", f"{runner_up.deadline_score:.2f}"
        return top.time_phrase(sense), runner_up.time_phrase(sense)
    if factor == "evidence":
        return f"{top.evidence_weight:.2f}", f"{runner_up.evidence_weight:.2f}"
    return f"{top.impact_weight:.2f}", f"{runner_up.impact_weight:.2f}"


def _advantage(top: Item, runner_up: Item, factor: str, situation: Situation) -> int:
    if factor == "evidence":
        a, b = top.evidence_weight, runner_up.evidence_weight
    elif factor == "impact":
        a, b = top.impact_weight, runner_up.impact_weight
    elif framing_of(situation).time_sense == "due":
        a, b = top.deadline_score, runner_up.deadline_score
    else:
        a, b = float(-top.deadline_days), float(-runner_up.deadline_days)
    return (a > b) - (a < b)


def _tied(top: Item, runner_up: Item, trace: Trace) -> bool:
    if not trace.scored or top.in_scope != runner_up.in_scope:
        return False
    return f"{_product(top, trace.rule):.3f}" == f"{_product(runner_up, trace.rule):.3f}"


def _with_tie(line: str, tied: bool) -> str:
    if not tied:
        return line
    return line[:-1] + " — tied overall, so the order between them is arbitrary."


def _ranked_first(top: Item, runner_up: Item, trace: Trace, situation: Situation) -> str:
    if trace.rule == "bearing":
        return (
            f"{top.name} still scores higher ({_product(top, trace.rule):.3f} vs "
            f"{_product(runner_up, trace.rule):.3f})"
        )
    if trace.rule == "least_privilege":
        label = framing_of(situation).impact_label
        return f"{top.name} ranks first on narrower {label} ({top.impact} vs {runner_up.impact})"
    return f"{top.name} ranks first by this policy's order"


def _basis_line(situation: Situation, decision: Decision, trace: Trace) -> str | None:
    top, runner_up = (situation.item(key) for key in decision.ranking[:2])
    factor = decision.deciding_factor
    f = framing_of(situation)
    tied = _tied(top, runner_up, trace)
    if trace.basis == "decline":
        return None
    if trace.basis == "largest_gap":
        a, b = _factor_values(top, runner_up, factor, situation)
        line = _pick(_LARGEST_GAP, situation, "basis").format(
            top=top.name, runner_up=runner_up.name, factor=factor, a=a, b=b
        )
        return _with_tie(line, tied)
    if trace.basis == "first_difference":
        a, b = _factor_values(top, runner_up, factor, situation)
        if a == b:
            overall = " and are tied overall" if tied else ""
            return (
                f"{top.name} and {runner_up.name} carry the same {factor}{overall}, so the "
                f"order between them rests on the tiebreak."
            )
        sense = framing_of(situation).time_sense
        named = "recency" if factor == "deadline" and sense != "due" else factor
        if _advantage(top, runner_up, factor, situation) < 0:
            return _with_tie(
                f"{_ranked_first(top, runner_up, trace, situation)}; they differ on {named} "
                f"({a} vs {b}), which this policy names as the label.",
                tied,
            )
        return _with_tie(
            f"{top.name} and {runner_up.name} differ on {named} ({a} vs {b}), so {factor} decides.",
            tied,
        )
    if trace.basis == "scope":
        return _with_tie(
            f"{top.name} is inside {situation.workspace_name} and outranks anything outside it.",
            tied,
        )
    if trace.basis == "explicit":
        return (
            f"{top.name} was stated outright and outranks every inferred record, "
            f"so the statement decides."
        )
    if trace.basis == "least_privilege":
        grade = _GRADE.get(top.evidence, top.evidence)
        return (
            f"more than one can answer, so the narrowest wins, with stronger evidence "
            f"breaking ties: {top.name} ({top.impact} {f.impact_label}, {grade})."
        )
    raise ScenarioError(f"No reasoning phrasing for basis {trace.basis!r}.")


def _conclusion(situation: Situation, decision: Decision) -> str:
    if decision.abstained:
        return f"so I will ask instead of ranking: {decision.deciding_factor.replace('_', ' ')}."
    names = [situation.item(key).name for key in decision.ranking]
    return "So: " + ", then ".join(names) + "."


def render_reasoning(situation: Situation, decision: Decision) -> str:
    trace = decision.trace
    if trace is None:
        raise ScenarioError(f"{situation.family}: the decision has no trace to render.")
    noun = framing_of(situation).noun
    lines = [_pick(_RULE_LINES[trace.rule], situation, "rule").format(noun=noun)]
    lines.extend(_item_line(item, situation, trace) for item in situation.presented())
    for check in trace.checks:
        line = _check_line(check, situation)
        if line is not None:
            lines.append(line)
    basis = _basis_line(situation, decision, trace)
    if basis is not None:
        lines.append(basis)
    lines.append(_conclusion(situation, decision))
    text = "\n".join(lines)
    if len(text) > MAX_REASONING_CHARS:
        raise ScenarioError(
            f"{situation.family}: reasoning is {len(text)} characters (cap {MAX_REASONING_CHARS}).",
            details={"group": situation.group_id},
        )
    return text
