"""Faithful mirror of the public dataset contract.

Ported from ``kleos_models.data.schemas`` at the commit in :mod:`.pin`. Every
validator, every ``extra`` policy and every default is reproduced, including the
ones that look incidental:

* ``Message`` and ``TrainingExample`` are ``extra="forbid"``; ``VariationAxes``
  and ``ExampleMetadata`` are ``extra="allow"``. Getting either backwards
  produces a release the public loader rejects, or one carrying fields it
  silently keeps.
* ``domain`` is auto-filled from ``variation_axes.domain`` when absent and is an
  error when it contradicts. Both branches change the written bytes.
* The conversation rules are checked in the public repo's order, so an example
  failing several of them reports the same first failure on both sides. Tests
  compare the failing field paths, not just accept/reject.

``tests/test_differential_schema_decisions.py`` asserts this mirror and the
pinned public models accept and reject exactly the same payloads.

**Do not "clean up" anything in this file.** Its correctness is defined entirely
by agreement with code that lives somewhere else.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kleos_training_data.contract.constants import (
    DATASET_SCHEMA_VERSION,
    PREPROCESSING_VERSION,
    QUALITY_STATUSES,
    SOURCE_TYPES,
    SUPPORTED_TASKS,
    VARIATION_AXES,
)

#: 3–128 characters, first alphanumeric. Mirrors ``schemas._ID_PATTERN``.
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")

#: Reasoning spans are stripped before training — a model is never taught to
#: emit display chain-of-thought. The public formatter strips these at
#: tokenization time regardless, so an assistant turn carrying them would train
#: on less than it appears to.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)
_DANGLING_CLOSE = re.compile(r"^\s*.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning spans from assistant text.

    Handles both a well-formed ``<think>…</think>`` block and the Qwen Thinking
    case where the template pre-opens ``<think>``, so generated text carries only
    the closing tag.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned:
        cleaned = _DANGLING_CLOSE.sub("", cleaned, count=1)
    return cleaned.strip()


def contains_reasoning(text: str) -> bool:
    """Whether the text carries a reasoning span."""
    return "</think>" in text.lower()


class Message(BaseModel):
    """One conversation turn."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = Field(default=None, description="Tool name for role='tool'.")

    @field_validator("content")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be empty or whitespace-only")
        return value

    @model_validator(mode="after")
    def _tool_needs_name(self) -> Message:
        if self.role == "tool" and not self.name:
            raise ValueError("messages with role='tool' require a 'name'")
        return self


class VariationAxes(BaseModel):
    """Situation descriptors used for coverage reporting.

    Extra keys are permitted so a new axis can be piloted here before being
    registered upstream. Unknown axes are surfaced by the coverage report rather
    than silently kept.
    """

    model_config = ConfigDict(extra="allow")

    domain: str = Field(description="Life/work domain, e.g. 'career', 'research'.")
    entities: str | None = Field(
        default=None, description="'seen' | 'unseen' | free-form entity-set label."
    )
    urgency: str | None = None
    deadlines: str | None = None
    evidence_quality: str | None = None
    conflicting_evidence: str | None = None
    context_length: str | None = None
    presentation_order: str | None = None
    format: str | None = None
    source_type: str | None = None
    workspace: str | None = None
    difficulty: str | None = None
    ambiguity: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """All axes with a value, including extras."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def known_axes(self) -> dict[str, Any]:
        """Only axes registered in ``VARIATION_AXES``."""
        return {k: v for k, v in self.as_dict().items() if k in VARIATION_AXES}

    def unknown_axes(self) -> dict[str, Any]:
        """Axes present on the example but not registered."""
        return {k: v for k, v in self.as_dict().items() if k not in VARIATION_AXES}


class ExampleMetadata(BaseModel):
    """Provenance and grouping information."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(
        default="synthetic",
        description="Provenance. Raw real user data must never reach a release.",
    )
    quality_status: str = Field(default="draft")
    scenario_family: str | None = Field(
        default=None,
        description=(
            "Groups logically related examples. Used by scenario-family-held-out "
            "splitting and by consistency testing, which needs perturbations of "
            "one scenario to stay together."
        ),
    )
    group_id: str | None = Field(default=None, description="Generic grouping key for splitting.")
    perturbation_of: str | None = Field(
        default=None, description="Id of the example this one perturbs."
    )
    perturbation_kind: str | None = None
    author: str | None = Field(default=None, description="Pipeline or role, never a person.")
    created_at: str | None = None
    notes: str | None = None

    @field_validator("source")
    @classmethod
    def _known_source(cls, value: str) -> str:
        if value not in SOURCE_TYPES:
            raise ValueError(f"unknown source {value!r}; valid: {list(SOURCE_TYPES)}")
        return value

    @field_validator("quality_status")
    @classmethod
    def _known_status(cls, value: str) -> str:
        if value not in QUALITY_STATUSES:
            raise ValueError(f"unknown quality_status {value!r}; valid: {list(QUALITY_STATUSES)}")
        return value


class TrainingExample(BaseModel):
    """One supervised conversational training example."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier, unique within a dataset version.")
    version: str = Field(default=DATASET_SCHEMA_VERSION, description="Schema version.")
    task: str = Field(description="Registered KLEOS task.")
    domain: str | None = Field(default=None, description="Convenience mirror of axes.domain.")
    messages: list[Message] = Field(min_length=2)
    variation_axes: VariationAxes
    metadata: ExampleMetadata = Field(default_factory=ExampleMetadata)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not ID_PATTERN.match(value):
            raise ValueError(
                f"id {value!r} must be 3-128 chars of [A-Za-z0-9._:-] and start alphanumeric"
            )
        return value

    @field_validator("task")
    @classmethod
    def _known_task(cls, value: str) -> str:
        if value not in SUPPORTED_TASKS:
            raise ValueError(f"unknown task {value!r}; registered: {list(SUPPORTED_TASKS)}")
        return value

    @model_validator(mode="after")
    def _conversation_shape(self) -> TrainingExample:
        roles = [m.role for m in self.messages]

        # A supervised example must have something to learn from.
        if "assistant" not in roles:
            raise ValueError("a training example needs at least one assistant message")
        if roles[-1] != "assistant":
            raise ValueError(
                f"the final message must be from the assistant, got role={roles[-1]!r}"
            )
        # System messages only lead.
        for index, role in enumerate(roles):
            if role == "system" and index != 0:
                raise ValueError(f"system message must be first, found one at index {index}")
        # There must be a user turn before the first assistant turn.
        first_assistant = roles.index("assistant")
        if "user" not in roles[:first_assistant]:
            raise ValueError("the first assistant message must be preceded by a user message")
        # No two consecutive assistant turns: that is a malformed conversation and
        # produces ambiguous supervision spans.
        for index in range(1, len(roles)):
            if roles[index] == "assistant" and roles[index - 1] == "assistant":
                raise ValueError(f"consecutive assistant messages at index {index - 1} and {index}")

        # Keep the convenience mirror consistent with the axes.
        if self.domain is None:
            self.domain = self.variation_axes.domain
        elif self.domain != self.variation_axes.domain:
            raise ValueError(
                f"domain={self.domain!r} contradicts variation_axes.domain="
                f"{self.variation_axes.domain!r}"
            )
        return self

    # -- derived views ------------------------------------------------------

    @property
    def system_prompt(self) -> str | None:
        """Leading system message, if any."""
        first = self.messages[0]
        return first.content if first.role == "system" else None

    @property
    def assistant_targets(self) -> list[str]:
        """Assistant message contents, in order."""
        return [m.content for m in self.messages if m.role == "assistant"]

    def conversation_text(self, *, include_assistant: bool = True) -> str:
        """Flat text view used for duplicate and leakage detection."""
        parts = [
            f"{m.role}: {m.content}"
            for m in self.messages
            if include_assistant or m.role != "assistant"
        ]
        return "\n".join(parts)

    def content_hash(self, *, include_assistant: bool = True) -> str:
        """Digest of the conversation content, ignoring id and metadata.

        This is the *public* notion of content identity, used for duplicate and
        leakage parity. It is deliberately NOT what mints an example id — see
        :mod:`kleos_training_data.ids`, which also folds in task and variation
        axes. Keeping the two distinctly named stops one being used where the
        other is meant, which would silently break either dedup or identity.
        """
        text = self.conversation_text(include_assistant=include_assistant)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def group_key(self, key: str | None = None) -> str:
        """Resolve the grouping key used by group-aware splitting.

        Falls back through ``group_id`` → ``scenario_family`` → the example's own
        id, so every example always belongs to exactly one group.
        """
        if key:
            value = getattr(self.metadata, key, None)
            if value is None:
                value = self.metadata.model_extra.get(key) if self.metadata.model_extra else None
            if value is None:
                value = self.variation_axes.as_dict().get(key)
            if value is not None:
                return str(value)
        return self.metadata.group_id or self.metadata.scenario_family or self.id

    def strip_reasoning_spans(self) -> TrainingExample:
        """Return a copy with reasoning removed from every assistant turn."""
        messages = [
            m.model_copy(update={"content": strip_reasoning(m.content)})
            if m.role == "assistant" and contains_reasoning(m.content)
            else m
            for m in self.messages
        ]
        return self.model_copy(update={"messages": messages})


class EvaluationExample(BaseModel):
    """One benchmark item.

    Mirrored so this repository can *read* the public evaluation fixtures for
    leakage checking. It never writes one — evaluation data is the public repo's
    concern.

    Note the asymmetry that matters for leakage: ``conversation_text`` here
    covers only the prompt, whereas the training version includes the assistant
    turn. See ``pin.DELIBERATE_DELTAS["eval_leakage_symmetry"]``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str = Field(default=DATASET_SCHEMA_VERSION)
    task: str
    messages: list[Message] = Field(min_length=1)
    variation_axes: VariationAxes
    metadata: ExampleMetadata = Field(default_factory=ExampleMetadata)

    reference: dict[str, Any] = Field(default_factory=dict)
    grader: str = Field(default="exact_match")
    split_tag: str | None = Field(
        default=None, description="'in_distribution' | 'ood' | 'capability'."
    )
    ood_shift: str | None = Field(default=None)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not ID_PATTERN.match(value):
            raise ValueError(f"id {value!r} is not a valid example id")
        return value

    @field_validator("task")
    @classmethod
    def _known_task(cls, value: str) -> str:
        if value not in SUPPORTED_TASKS:
            raise ValueError(f"unknown task {value!r}; registered: {list(SUPPORTED_TASKS)}")
        return value

    @model_validator(mode="after")
    def _prompt_shape(self) -> EvaluationExample:
        roles = [m.role for m in self.messages]
        if "user" not in roles:
            raise ValueError("an evaluation example needs at least one user message")
        for index, role in enumerate(roles):
            if role == "system" and index != 0:
                raise ValueError(f"system message must be first, found one at index {index}")
        if not self.reference:
            raise ValueError(
                "an evaluation example needs a non-empty 'reference'; "
                "without ground truth it cannot be graded"
            )
        return self

    @property
    def prompt_messages(self) -> list[Message]:
        """Messages fed to the model: everything up to the first assistant turn."""
        result: list[Message] = []
        for message in self.messages:
            if message.role == "assistant":
                break
            result.append(message)
        return result

    def conversation_text(self) -> str:
        """Flat prompt text, used for leakage checks against the training set."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.prompt_messages)

    def content_hash(self) -> str:
        return hashlib.sha256(self.conversation_text().encode("utf-8")).hexdigest()


class SplitCounts(BaseModel):
    """Example counts per partition."""

    model_config = ConfigDict(extra="forbid")

    train: int = 0
    validation: int = 0
    test: int = 0

    @property
    def total(self) -> int:
        return self.train + self.validation + self.test


class DatasetManifest(BaseModel):
    """Dataset-level provenance.

    ``extra="forbid"`` is the constraint that shapes this repository's release
    layout: one added key makes the public ``load_manifest`` raise
    ``DatasetIntegrityError`` at train time. Everything this repository wants to
    record and the manifest cannot hold — ruleset versions, gate report hashes,
    scenario catalog version, review counts — goes in a sibling
    ``provenance.json``, which the public loader ignores.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(description="e.g. 'kleos-policy-v0.1.0'.")
    schema_version: str = Field(default=DATASET_SCHEMA_VERSION)
    preprocessing_version: str = Field(default=PREPROCESSING_VERSION)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: str = Field(default="synthetic")
    description: str = ""

    example_count: int = Field(default=0, ge=0)
    splits: SplitCounts = Field(default_factory=SplitCounts)
    task_distribution: dict[str, int] = Field(default_factory=dict)
    domain_distribution: dict[str, int] = Field(default_factory=dict)
    source_distribution: dict[str, int] = Field(default_factory=dict)
    quality_distribution: dict[str, int] = Field(default_factory=dict)

    split_strategy: str | None = None
    split_seed: int | None = None
    holdout_values: list[str] = Field(default_factory=list)

    file_hashes: dict[str, str] = Field(default_factory=dict)
    content_hash: str | None = None

    contains_private_data: bool = Field(default=False)
    notes: str | None = None

    def compute_content_hash(self) -> str:
        """Deterministic digest over the per-file hashes.

        Covers ``file_hashes`` and nothing else. Adding provenance.json to that
        dict would make our hash differ from the one the public
        ``split_dataset.py`` computes for identical splits, which would break the
        only cheap way to confirm two repositories are talking about the same
        data.
        """
        payload = json.dumps(self.file_hashes, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def finalize(self) -> DatasetManifest:
        """Fill in the aggregate content hash."""
        self.content_hash = self.compute_content_hash()
        return self
