"""A shared corpus of contract-valid and contract-invalid payloads.

Used twice, and that is the point:

* ``test_contract_schemas.py`` asserts the mirror's own behaviour, and runs
  everywhere.
* ``test_differential_schema_decisions.py`` asserts the pinned public models
  reach the *same* verdict on every one of these, and runs only where
  kleos-models is installed.

Keeping one corpus means a case added to pin down a bug is automatically checked
for agreement too, rather than only being checked against our own reimplementation
of the rule it came from.
"""

from __future__ import annotations

from typing import Any

VALID_ID = "kx-npr-3f9a1c8e2b7d0456"


def base(**overrides: Any) -> dict[str, Any]:
    """A minimal valid payload, with overrides applied."""
    payload: dict[str, Any] = {
        "id": VALID_ID,
        "task": "notification_prioritization",
        "messages": [
            {"role": "user", "content": "Two items are open. Which first?"},
            {"role": "assistant", "content": "Item A: nearest deadline, confirmed evidence."},
        ],
        "variation_axes": {"domain": "career"},
    }
    payload.update(overrides)
    return payload


def _messages(*pairs: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"role": role, "content": content} for role, content in pairs]


#: name -> payload that MUST validate.
VALID_CASES: dict[str, dict[str, Any]] = {
    "minimal": base(),
    "with_system": base(
        messages=_messages(
            ("system", "Rank by deadline proximity."),
            ("user", "Which first?"),
            ("assistant", "Item A."),
        )
    ),
    "with_tool_turn": base(
        messages=[
            {"role": "user", "content": "Which first?"},
            {"role": "tool", "content": "{'deadline': 'friday'}", "name": "calendar_search"},
            {"role": "assistant", "content": "Item A."},
        ]
    ),
    "multi_turn": base(
        messages=_messages(
            ("system", "Rank by deadline proximity."),
            ("user", "Which first?"),
            ("assistant", "Item A."),
            ("user", "Why?"),
            ("assistant", "Its deadline is nearest and the evidence is confirmed."),
        )
    ),
    "consecutive_user_turns_are_allowed": base(
        messages=_messages(
            ("user", "Two items are open."),
            ("user", "Which first?"),
            ("assistant", "Item A."),
        )
    ),
    "unregistered_axis_is_piloted_not_rejected": base(
        variation_axes={"domain": "career", "novel_axis": "experimental"}
    ),
    "metadata_extras_are_allowed_by_the_schema": base(
        metadata={"source": "synthetic", "quality_status": "reviewed", "anything": "goes"}
    ),
    "domain_mirror_matching_axes": base(domain="career"),
    "non_ascii_content": base(
        messages=_messages(
            ("user", "Café résumé — naïve deadline? 日本語"),
            ("assistant", "Item A — soonest."),
        )
    ),
    "id_at_minimum_length": base(id="abc"),
    "id_at_maximum_length": base(id="a" + "b" * 127),
}


#: name -> (payload that MUST be rejected, substring expected in the error).
INVALID_CASES: dict[str, tuple[dict[str, Any], str]] = {
    "unknown_task": (base(task="summarization"), "unknown task"),
    "id_too_short": (base(id="ab"), "id"),
    "id_too_long": (base(id="a" + "b" * 128), "id"),
    "id_starts_with_punctuation": (base(id="-abc"), "id"),
    "id_has_illegal_character": (base(id="kx npr 0000"), "id"),
    "single_message": (
        base(messages=[{"role": "assistant", "content": "A."}]),
        "at least 2",
    ),
    "no_assistant_turn": (
        base(messages=_messages(("user", "Which first?"), ("user", "Well?"))),
        "assistant",
    ),
    "final_message_not_assistant": (
        base(
            messages=_messages(
                ("user", "Which first?"), ("assistant", "Item A."), ("user", "Thanks")
            )
        ),
        "final message",
    ),
    "system_message_not_first": (
        base(
            messages=_messages(
                ("user", "Which first?"), ("system", "Rank."), ("assistant", "Item A.")
            )
        ),
        "system message must be first",
    ),
    "consecutive_assistant_turns": (
        base(
            messages=_messages(
                ("user", "Which first?"), ("assistant", "Item A."), ("assistant", "Also B.")
            )
        ),
        "consecutive assistant",
    ),
    "assistant_before_any_user": (
        base(
            messages=_messages(
                ("assistant", "Item A."), ("user", "Why?"), ("assistant", "Deadline.")
            )
        ),
        "preceded by a user",
    ),
    "empty_content": (
        base(messages=_messages(("user", "   "), ("assistant", "Item A."))),
        "empty",
    ),
    "tool_turn_without_name": (
        base(
            messages=[
                {"role": "user", "content": "Which first?"},
                {"role": "tool", "content": "{}"},
                {"role": "assistant", "content": "Item A."},
            ]
        ),
        "require a 'name'",
    ),
    "unknown_role": (
        base(messages=_messages(("bot", "hi"), ("assistant", "Item A."))),
        "role",
    ),
    "unknown_top_level_field": (base(surprise="value"), "surprise"),
    "unknown_message_field": (
        base(
            messages=[
                {"role": "user", "content": "Which first?", "timestamp": "now"},
                {"role": "assistant", "content": "Item A."},
            ]
        ),
        "timestamp",
    ),
    "missing_variation_axes": (
        {k: v for k, v in base().items() if k != "variation_axes"},
        "variation_axes",
    ),
    "missing_required_domain_axis": (base(variation_axes={"urgency": "high"}), "domain"),
    "domain_mirror_contradicts_axes": (base(domain="research"), "contradicts"),
    "unknown_source": (base(metadata={"source": "scraped"}), "unknown source"),
    "unknown_quality_status": (
        base(metadata={"quality_status": "great"}),
        "unknown quality_status",
    ),
    "missing_task": ({k: v for k, v in base().items() if k != "task"}, "task"),
    "missing_id": ({k: v for k, v in base().items() if k != "id"}, "id"),
}
