"""Raw capture -> contract-shaped candidate.

Normalization is **deterministic and non-semantic**. It fixes transport and
presentation artifacts: line endings, wrapper envelopes, reasoning spans the
public formatter strips anyway, trailing whitespace. It does not touch meaning.

Semantic rewriting belongs to sanitization and review, where it is recorded as a
transformation and re-reviewed. The rule the whole pipeline depends on:

    **Never silently change an assistant answer and then present it as the
    original model output.**

Every change made here is appended to ``transformations`` on the candidate, so a
reviewer can see what was done to the text before they read it.
"""

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
    """Clean transport artifacts. Returns the text and what changed."""
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
    """Turn one raw capture into a contract-shaped candidate.

    Raises:
        ContractViolationError: If the capture cannot produce a valid example —
            an empty answer, most often, which means the stream was truncated.
    """
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
    """Build a candidate directly from a generated payload.

    The synthetic lane has no transport to normalize — the text was rendered
    locally from the policy — so it skips the raw stage entirely rather than
    round-tripping through a fake capture to look symmetrical.
    """
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
