from __future__ import annotations

import json
from typing import Final

from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.policies import Decision
from kleos_training_data.scenarios.situations import (
    FRAMINGS,
    Framing,
    Item,
    Situation,
    framing_of,
    resolve_framing,
)

__all__ = [
    "ANSWER_FORMATS",
    "FRAMINGS",
    "PROMPT_FORMATS",
    "Framing",
    "framing_of",
    "render_answer",
    "render_prompt",
    "render_system_prompt",
    "resolve_framing",
    "summarize_items",
]

PROMPT_FORMATS: Final[tuple[str, ...]] = (
    "bullets",
    "prose",
    "json",
    "slack_thread",
    "github_issue",
    "calendar",
)

ANSWER_FORMATS: Final[tuple[str, ...]] = ("bullets", "prose", "json")


def _scope_suffix(item: Item, situation: Situation) -> str:
    if situation.framing != "workspace":
        return ""
    if item.in_scope:
        return f" [workspace: {situation.workspace_name}]"
    return " [workspace: other — outside the one you named]"


def _bullets(situation: Situation) -> str:
    f = framing_of(situation)
    lines = [
        f"- {i.name}: {i.time_phrase(f.time_sense)}; {f.evidence_label} "
        f"{i.evidence_phrase}; {i.impact} {f.impact_label}. {i.detail}"
        f"{_scope_suffix(i, situation)}"
        for i in situation.presented()
    ]
    return "\n".join(lines)


def _prose(situation: Situation) -> str:
    f = framing_of(situation)
    sentences = [
        f"{i.name} was {i.time_phrase(f.time_sense)}."
        if f.time_sense != "due"
        else f"{i.name} is {i.time_phrase(f.time_sense)}."
        for i in situation.presented()
    ]
    detailed = [
        f"{sentence} It is {i.evidence_phrase}, and it is {i.impact} "
        f"{f.impact_label}. {i.detail}{_scope_suffix(i, situation)}"
        for sentence, i in zip(sentences, situation.presented(), strict=True)
    ]
    return " ".join(detailed)


def _json(situation: Situation) -> str:
    f = framing_of(situation)
    payload = []
    for i in situation.presented():
        entry: dict[str, object] = {
            "name": i.name,
            f.time_field: i.deadline_days,
            f.evidence_label: i.evidence,
            f.impact_label: i.impact,
            "note": i.detail,
        }
        if situation.framing == "workspace":
            entry["workspace"] = situation.workspace_name if i.in_scope else "other"
        payload.append(entry)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _slack_thread(situation: Situation) -> str:
    f = framing_of(situation)
    lines = []
    for index, item in enumerate(situation.presented()):
        speaker = ("dana", "kai", "morgan", "sam")[index % 4]
        lines.append(
            f"@{speaker}: heads up on {item.name} — it's {item.time_phrase(f.time_sense)}. "
            f"{item.evidence_phrase}. {item.detail}{_scope_suffix(item, situation)}"
        )
    return "\n".join(lines)


def _github_issue(situation: Situation) -> str:
    f = framing_of(situation)
    blocks = []
    for index, item in enumerate(situation.presented(), start=1):
        blocks.append(
            f"#{100 + index} {item.name}\n"
            f"  labels: {item.impact}-{f.impact_label}, {f.evidence_label}:{item.evidence}\n"
            f"  milestone: {item.time_phrase(f.time_sense)}\n"
            f"  {item.detail}{_scope_suffix(item, situation)}"
        )
    return "\n\n".join(blocks)


def _calendar(situation: Situation) -> str:
    f = framing_of(situation)
    lines = []
    for item in situation.presented():
        lines.append(
            f"[{item.deadline_days:>3}d] {item.name}\n"
            f"        status: {item.evidence} | {f.impact_label}: {item.impact}\n"
            f"        {item.detail}{_scope_suffix(item, situation)}"
        )
    return "\n".join(lines)


_PROMPT_RENDERERS = {
    "bullets": _bullets,
    "prose": _prose,
    "json": _json,
    "slack_thread": _slack_thread,
    "github_issue": _github_issue,
    "calendar": _calendar,
}


def render_prompt(situation: Situation) -> str:
    fmt = situation.axes.get("format", "bullets")
    renderer = _PROMPT_RENDERERS.get(fmt)
    if renderer is None:
        raise ScenarioError(
            f"Unknown prompt format {fmt!r}.",
            details={"known": ", ".join(PROMPT_FORMATS)},
            suggestions=["Fix the scenario's `format` axis values."],
        )

    parts: list[str] = []
    if situation.need:
        parts.append(f"{situation.need}\n")
    if situation.framing == "workspace" and situation.workspace_name:
        parts.append(f"Active workspace: {situation.workspace_name}\n")
    parts.append(renderer(situation))
    if situation.distractors:
        parts.append("\nAlso on my plate: " + " ".join(situation.distractors))
    parts.append(f"\n{situation.question}")
    return "\n".join(parts).strip()


def _label_line(decision: Decision) -> str:
    return f"What decided it: {decision.deciding_factor}."


def _answer_bullets(situation: Situation, decision: Decision) -> str:
    if decision.abstained:
        return f"{_declined_bullets(situation, decision)}\n\n{_label_line(decision)}"

    lines = []
    for position, (key, why) in enumerate(
        zip(decision.ranking, decision.rationale, strict=True), start=1
    ):
        item = situation.item(key)
        lines.append(f"{position}. {item.name} — {why}.")
    lines.append(
        f"\nWhat decided it: {decision.deciding_factor}. {_factor_sentence(situation, decision)}"
    )
    return "\n".join(lines)


def _declined_bullets(situation: Situation, decision: Decision) -> str:
    first, second = (situation.item(k) for k in decision.ranking[:2])
    if decision.deciding_factor == "ask_before_crossing":
        return (
            f"- Nothing inside {situation.workspace_name} is solid enough to answer "
            f"this.\n"
            f"- {decision.resolver}.\n"
            f"- I have not read it, and will not without you saying so — that "
            f"boundary is yours to open, not mine."
        )
    if decision.deciding_factor == "request_ambiguous":
        return (
            f"- I could answer this several ways and I do not know which one you "
            f"want.\n"
            f"- {first.name} and {second.name} answer different questions, so "
            f"picking one would be a coin flip dressed as a recommendation.\n"
            f"- {decision.resolver}."
        )
    if decision.deciding_factor == "stale_explicit_conflict":
        return (
            f"- {decision.resolver}\n"
            f"- I have not overwritten it. Until you say otherwise, what you told "
            f"me stands."
        )
    if decision.deciding_factor == "missing_input":
        return (
            f"- I can't answer this yet, and the gap is not a close call — "
            f"something the answer depends on is missing.\n"
            f"- {first.name} leads on what you have given me, but it rests on an "
            f"input nobody has supplied.\n"
            f"- So, one question: {decision.resolver}.\n"
            f"- With that, this becomes a straightforward call."
        )
    return (
        f"- These two are too close to separate on what you have: {first.name} "
        f"and {second.name}.\n"
        f"- The gap between them is smaller than the uncertainty in the evidence, "
        f"so ranking them now would be a guess dressed as a decision.\n"
        f"- Do this first: {decision.resolver}.\n"
        f"- If that comes back clean, {first.name} goes first."
    )


def _answer_prose(situation: Situation, decision: Decision) -> str:
    if decision.abstained:
        return f"{_declined_prose(situation, decision)} {_label_line(decision)}"

    top = situation.item(decision.ranking[0])
    rest = [situation.item(k) for k in decision.ranking[1:]]
    tail = ", then ".join(item.name for item in rest)
    return (
        f"Start with {top.name} — {decision.rationale[0]}. "
        f"Then {tail}. "
        f"{_label_line(decision)} {_factor_sentence(situation, decision)}"
    )


def _declined_prose(situation: Situation, decision: Decision) -> str:
    first, second = (situation.item(k) for k in decision.ranking[:2])
    if decision.deciding_factor == "ask_before_crossing":
        return (
            f"I can't answer this from {situation.workspace_name} — what is in "
            f"there is too thinly supported to act on. There is something that "
            f"would settle it, but it sits in another workspace and I have not "
            f"opened it. {decision.resolver}. I would rather ask than quietly "
            f"reach across."
        )
    if decision.deciding_factor == "request_ambiguous":
        return (
            f"I could take this several ways and I do not know which you want. "
            f"{first.name} and {second.name} answer different questions, so "
            f"choosing between them now would be a coin flip with a confident "
            f"tone. {decision.resolver}."
        )
    if decision.deciding_factor == "stale_explicit_conflict":
        return (
            f"{decision.resolver} I have not overwritten it — until you say "
            f"otherwise, what you told me stands."
        )
    if decision.deciding_factor == "missing_input":
        return (
            f"I don't have enough to answer this, and it is not that the options "
            f"are close — something the answer depends on was never supplied. "
            f"{first.name} leads on what you have given me, but it rests on an "
            f"input nobody has checked. So rather than guess: {decision.resolver}. "
            f"With that in hand this is a straightforward call."
        )
    return (
        f"I would not rank {first.name} against {second.name} yet. They are close "
        f"enough that the difference is inside the uncertainty in the evidence, so "
        f"picking one now would be a guess with a confident tone. "
        f"The useful next step is narrower than a decision: {decision.resolver}. "
        f"If that holds up, {first.name} goes first."
    )


def _answer_json(situation: Situation, decision: Decision) -> str:
    payload = {
        "ranking": [situation.item(k).name for k in decision.ranking],
        "deciding_factor": decision.deciding_factor,
        "confident": not decision.abstained,
        "reasons": {
            situation.item(k).name: why
            for k, why in zip(decision.ranking, decision.rationale, strict=True)
        },
    }
    if decision.abstained:
        payload["next_step"] = decision.resolver
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _factor_sentence(situation: Situation, decision: Decision) -> str:
    f = framing_of(situation)
    top = situation.item(decision.ranking[0])
    runner_up = situation.item(decision.ranking[1])
    factor = decision.deciding_factor

    if factor == "scope":
        inside = [situation.item(k) for k in decision.ranking if situation.item(k).in_scope]
        outside = [situation.item(k) for k in decision.ranking if not situation.item(k).in_scope]
        named = ", ".join(item.name for item in outside)
        one = len(outside) == 1
        inside_count = "the one item" if len(inside) == 1 else f"the {len(inside)} items"
        return (
            f"Scope decides before anything else: {named} {'sits' if one else 'sit'} "
            f"in another workspace, so {'it is' if one else 'they are'} flagged "
            f"rather than acted on, and the ranking runs over {inside_count} inside "
            f"{situation.workspace_name}."
        )

    if factor == "explicit_statement":
        return (
            f"An explicit statement outranks an inference however well corroborated: "
            f"you said {top.name} outright, and the rest are inferred from behaviour, "
            f"which is evidence about circumstance rather than about intent."
        )

    if factor == "least_privilege":
        return (
            f"Both could answer this, so the narrower source wins: {top.name} has "
            f"{top.impact} reach against {runner_up.impact}, and opening the broader "
            f"one would read more than the question needs."
        )

    same_evidence = top.evidence == runner_up.evidence
    close_timing = top.time_phrase(f.time_sense) == runner_up.time_phrase(f.time_sense)
    same_impact = top.impact == runner_up.impact

    def _aside() -> str:
        held = []
        if same_evidence:
            held.append(f"{f.evidence_label} is the same on both")
        if close_timing:
            held.append("the timing reads the same")
        if same_impact:
            held.append(f"{f.impact_label} is the same on both")
        return f" {'; '.join(held).capitalize()}." if held else ""

    if factor == "deadline":
        if top.time_phrase(f.time_sense) == runner_up.time_phrase(f.time_sense):
            return (
                f"Nothing in what you have separates {top.name} from {runner_up.name} — "
                f"same standing, and the same timing once you round it — so the order "
                f"between them is arbitrary and should not be read as a judgement."
            )
        if f.time_sense == "age":
            return (
                f"Support is comparable, so recency breaks the tie: {top.name} was "
                f"{top.age_phrase} against {runner_up.age_phrase}."
            )
        if f.time_sense == "staleness":
            return (
                f"Bearing is comparable, so freshness breaks the tie: {top.name} was "
                f"{top.staleness_phrase} against {runner_up.staleness_phrase}."
            )
        lead = (
            "With evidence of comparable strength, the nearer deadline wins"
            if same_evidence
            else "The nearer deadline wins here"
        )
        return (
            f"{lead}, and {top.name} is {top.deadline_phrase} against "
            f"{runner_up.deadline_phrase}.{'' if same_evidence else _aside()}"
        )

    if factor == "evidence":
        if top.evidence == runner_up.evidence:
            return (
                f"{top.name} and {runner_up.name} carry the same {f.evidence_label}, so "
                f"the order between them rests on the tiebreak rather than on anything "
                f"that separates them."
            )
        if f.time_sense == "age":
            return (
                f"The newer record is not the better one here: {top.evidence} beats "
                f"{runner_up.evidence}, and treating recency as authority is how a "
                f"contradicted note becomes the current answer."
            )
        if f.time_sense == "staleness":
            return (
                f"Freshness is not relevance: {top.name} bears on the question "
                f"({top.evidence}) more directly than {runner_up.name} "
                f"({runner_up.evidence}), whatever their sync times say."
            )
        lead = (
            "Deadlines are close, so evidence breaks the tie"
            if close_timing
            else "Evidence decides this one"
        )
        return (
            f"{lead}: {top.evidence} beats {runner_up.evidence}, and acting on the "
            f"weaker one risks doing the wrong work.{'' if close_timing else _aside()}"
        )

    if top.impact == runner_up.impact:
        return (
            f"{top.name} and {runner_up.name} carry the same {f.impact_label}, so the "
            f"order between them rests on the tiebreak rather than on anything that "
            f"separates them."
        )
    lead = (
        f"Timing and {f.evidence_label} are comparable, so {f.impact_label} decides"
        if (close_timing and same_evidence)
        else f"{f.impact_label.capitalize()} decides this one"
    )
    tail = "" if (close_timing and same_evidence) else _aside()
    return f"{lead}: {top.impact} against {runner_up.impact}.{tail}"


_ANSWER_RENDERERS = {
    "bullets": _answer_bullets,
    "prose": _answer_prose,
    "json": _answer_json,
}


def render_answer(situation: Situation, decision: Decision) -> str:
    fmt = situation.axes.get("format", "bullets")
    renderer = _ANSWER_RENDERERS.get(fmt, _answer_bullets)
    return renderer(situation, decision)


def render_system_prompt(situation: Situation) -> str:
    return framing_of(situation).system


def summarize_items(items: tuple[Item, ...]) -> str:
    return "; ".join(f"{i.key}={i.name}({i.deadline_days}d,{i.evidence},{i.impact})" for i in items)
