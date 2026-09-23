from __future__ import annotations

from typing import Any

from kleos_training_data.contract.schemas import contains_reasoning, strip_reasoning
from kleos_training_data.errors import ContractViolationError
from kleos_training_data.ids import content_hash, example_id
from kleos_training_data.staging.records import (
    NormalizedCandidate,
    RawCapture,
)


def _normalize_text(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        changes.append("normalized_line_endings")

    if contains_reasoning(text):
        text = strip_reasoning(text)
        changes.append("stripped_reasoning_span")

    lines = [line.rstrip() for line in text.split("\n")]
    rejoined = "\n".join(lines).strip()
    if rejoined != text:
        changes.append("trimmed_whitespace")
    return rejoined, changes


def normalize_capture(
    capture: RawCapture,
    *,
    system_prompt: str,
    user_message: str,
    variation_axes: dict[str, str],
    group_id: str,
    perturbation_of: str | None = None,
    perturbation_kind: str | None = None,
) -> NormalizedCandidate:
    answer, changes = _normalize_text(capture.answer_text)
    if not answer.strip():
        raise ContractViolationError(
            f"Capture {capture.capture_id} has no assistant content after normalization.",
            details={
                "adapter": capture.adapter,
                "raw_length": len(capture.answer_text),
                "frames": ", ".join(
                    f"{k}={v}" for k, v in sorted(capture.transport.frame_type_counts.items())
                ),
            },
            suggestions=[
                "Usually a truncated stream, or an answer that was nothing but a "
                "reasoning span. Neither can become a training target.",
            ],
        )

    prompt, prompt_changes = _normalize_text(user_message)
    system, system_changes = _normalize_text(system_prompt)
    for change in prompt_changes + system_changes:
        if change not in changes:
            changes.append(change)

    payload: dict[str, Any] = {
        "task": capture.scenario.task,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "variation_axes": dict(variation_axes),
    }

    return NormalizedCandidate(
        candidate_id=example_id(payload),
        provisional=True,
        source_capture_id=capture.capture_id,
        batch_id=capture.batch_id,
        lane=capture.lane,
        scenario=capture.scenario,
        payload=payload,
        scenario_family=capture.scenario.family,
        group_id=group_id,
        perturbation_of=perturbation_of,
        perturbation_kind=perturbation_kind,
        content_hash=content_hash(payload),
        transformations=changes,
    )


def candidate_from_payload(
    payload: dict[str, Any],
    *,
    batch_id: str,
    lane: Any,
    scenario: Any,
    group_id: str,
    perturbation_of: str | None = None,
    perturbation_kind: str | None = None,
) -> NormalizedCandidate:
    normalized_messages = []
    changes: list[str] = []
    for message in payload["messages"]:
        cleaned, message_changes = _normalize_text(message["content"])
        normalized_messages.append({**message, "content": cleaned})
        for change in message_changes:
            if change not in changes:
                changes.append(change)

    normalized = {**payload, "messages": normalized_messages}

    return NormalizedCandidate(
        candidate_id=example_id(normalized),
        provisional=True,
        source_capture_id=None,
        batch_id=batch_id,
        lane=lane,
        scenario=scenario,
        payload=normalized,
        scenario_family=scenario.family,
        group_id=group_id,
        perturbation_of=perturbation_of,
        perturbation_kind=perturbation_kind,
        content_hash=content_hash(normalized),
        transformations=changes,
    )
