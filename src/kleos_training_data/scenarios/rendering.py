"""Render a situation into a prompt, and a decision into an answer.

Presentation is a variation axis, not a detail. The same situation rendered as a
bulleted list, a Slack thread and a GitHub issue is the *same problem*, and a
model that only solves the format it was trained on has learned a template
rather than a policy. That is why the input formats here go well beyond "bullets
or prose", and why ``unseen_formats`` is a registered OOD shift.

The answer renderers are deliberately plain. A training target should read like
a good answer, not like a model performing thoroughness: it states the decision,
names the factor that decided it, and stops. Every claim it makes is derived from
the same computation that produced the ranking, so it cannot assert anything the
prompt does not support — which is exactly what the ``no_unsupported_claims``
review gate checks for.
"""

from __future__ import annotations

import json
from typing import Final

from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.policies import Decision
from kleos_training_data.scenarios.situations import Item, Situation

#: Input formats the prompt can be rendered in. The first three are ordinary;
#: the rest exist so a format holdout has somewhere unseen to hold out to.
PROMPT_FORMATS: Final[tuple[str, ...]] = (
    "bullets",
    "prose",
    "json",
    "slack_thread",
    "github_issue",
    "calendar",
)

#: Output formats an answer can be rendered in.
ANSWER_FORMATS: Final[tuple[str, ...]] = ("bullets", "prose", "json")


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _bullets(situation: Situation) -> str:
    lines = [
        f"- {i.name}: {i.deadline_phrase}; evidence {i.evidence_phrase}; "
        f"{i.impact} impact if it slips. {i.detail}"
        for i in situation.presented()
    ]
    return "\n".join(lines)


def _prose(situation: Situation) -> str:
    sentences = [
        f"{i.name} is {i.deadline_phrase}. It is {i.evidence_phrase}, and missing it "
        f"would be {i.impact} impact. {i.detail}"
        for i in situation.presented()
    ]
    return " ".join(sentences)


def _json(situation: Situation) -> str:
    payload = [
        {
            "name": i.name,
            "due_in_days": i.deadline_days,
            "evidence": i.evidence,
            "impact": i.impact,
            "note": i.detail,
        }
        for i in situation.presented()
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _slack_thread(situation: Situation) -> str:
    lines = []
    for index, item in enumerate(situation.presented()):
        speaker = ("dana", "kai", "morgan", "sam")[index % 4]
        lines.append(
            f"@{speaker}: heads up on {item.name} — it's {item.deadline_phrase}. "
            f"{item.evidence_phrase}. {item.detail}"
        )
    return "\n".join(lines)


def _github_issue(situation: Situation) -> str:
    blocks = []
    for index, item in enumerate(situation.presented(), start=1):
        blocks.append(
            f"#{100 + index} {item.name}\n"
            f"  labels: {item.impact}-impact, evidence:{item.evidence}\n"
            f"  milestone: {item.deadline_phrase}\n"
            f"  {item.detail}"
        )
    return "\n\n".join(blocks)


def _calendar(situation: Situation) -> str:
    lines = []
    for item in situation.presented():
        lines.append(
            f"[{item.deadline_days:>3}d] {item.name}\n"
            f"        status: {item.evidence} | impact: {item.impact}\n"
            f"        {item.detail}"
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
    """Render the user turn in the situation's declared input format."""
    fmt = situation.axes.get("format", "bullets")
    renderer = _PROMPT_RENDERERS.get(fmt)
    if renderer is None:
        raise ScenarioError(
            f"Unknown prompt format {fmt!r}.",
            details={"known": ", ".join(PROMPT_FORMATS)},
            suggestions=["Fix the scenario's `format` axis values."],
        )

    parts = [renderer(situation)]
    if situation.distractors:
        # Irrelevant context goes *after* the items, where it is most likely to
        # displace the relevant material from a model's attention.
        parts.append("\nAlso on my plate: " + " ".join(situation.distractors))
    parts.append(f"\n{situation.question}")
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Answer rendering
# ---------------------------------------------------------------------------


def _answer_bullets(situation: Situation, decision: Decision) -> str:
    if decision.abstained:
        first, second = (situation.item(k) for k in decision.ranking[:2])
        return (
            f"- These two are too close to separate on what you have: {first.name} "
            f"and {second.name}.\n"
            f"- The gap between them is smaller than the uncertainty in the evidence, "
            f"so ranking them now would be a guess dressed as a decision.\n"
            f"- Do this first: {decision.resolver}.\n"
            f"- If that comes back clean, {first.name} goes first on deadline."
        )

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


def _answer_prose(situation: Situation, decision: Decision) -> str:
    if decision.abstained:
        first, second = (situation.item(k) for k in decision.ranking[:2])
        return (
            f"I would not rank {first.name} against {second.name} yet. They are close "
            f"enough that the difference is inside the uncertainty in the evidence, so "
            f"picking one now would be a guess with a confident tone. "
            f"The useful next step is narrower than a decision: {decision.resolver}. "
            f"If that holds up, {first.name} goes first on deadline."
        )

    top = situation.item(decision.ranking[0])
    rest = [situation.item(k) for k in decision.ranking[1:]]
    tail = ", then ".join(item.name for item in rest)
    return (
        f"Start with {top.name} — {decision.rationale[0]}. "
        f"Then {tail}. "
        f"{_factor_sentence(situation, decision)}"
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
    """One sentence naming why the top item won, using only prompt facts.

    Phrased as a *rule* rather than as a fact about these items, so the sentence
    a model learns to produce is transferable. "Nearest deadline wins when the
    evidence is comparable" survives into a situation with different names;
    "Project X is first because it is due Friday" does not.
    """
    top = situation.item(decision.ranking[0])
    runner_up = situation.item(decision.ranking[1])
    if decision.deciding_factor == "deadline":
        return (
            f"With evidence of comparable strength, the nearer deadline wins, and "
            f"{top.name} is {top.deadline_phrase} against {runner_up.deadline_phrase}."
        )
    if decision.deciding_factor == "evidence":
        return (
            f"Deadlines are close, so evidence breaks the tie: {top.evidence} beats "
            f"{runner_up.evidence}, and acting on the weaker one risks doing the "
            f"wrong work."
        )
    return (
        f"Timing and evidence are comparable, so the consequence of delay decides: "
        f"{top.impact} against {runner_up.impact}."
    )


_ANSWER_RENDERERS = {
    "bullets": _answer_bullets,
    "prose": _answer_prose,
    "json": _answer_json,
}


def render_answer(situation: Situation, decision: Decision) -> str:
    """Render the assistant turn.

    The answer format follows the prompt format where one applies, and falls
    back to bullets for the structural formats (a Slack thread is an input shape,
    not an output shape).
    """
    fmt = situation.axes.get("format", "bullets")
    renderer = _ANSWER_RENDERERS.get(fmt, _answer_bullets)
    return renderer(situation, decision)


def render_system_prompt(situation: Situation) -> str:
    """The instruction the model is operating under.

    Deliberately states the *policy*, not the answer. A system prompt that named
    the winning item would make every example trivially solvable and teach
    nothing.
    """
    return (
        "You help prioritize competing work. Rank the items by where delay costs "
        "the most: how near the deadline is, how well the claim is actually "
        "supported, and how much the outcome matters. Name the factor that decided "
        "it. If the top options are too close to separate on the evidence "
        "available, say so and name what would resolve it instead of guessing."
    )


def summarize_items(items: tuple[Item, ...]) -> str:
    """Compact one-line summary, for reviewer packets and reports."""
    return "; ".join(f"{i.key}={i.name}({i.deadline_days}d,{i.evidence},{i.impact})" for i in items)
