"""The scenario file format.

A scenario describes a *family of situations* and the policy that resolves them.
It is deliberately not a conversation with placeholders: two scenarios that
differ only in wording teach nothing new, and a catalog full of them produces a
dataset that is large and narrow.

The two fields that carry the research claim are ``policy_claim`` and
``anti_claim``. The first states what an example teaches; the second states the
surface rule a lazy model might learn instead, which is what the perturbations
and holdouts are designed to catch. A scenario that cannot state its anti-claim
usually has not identified what it is testing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kleos_training_data.contract.constants import (
    OOD_SHIFT_KINDS,
    PERTURBATION_KINDS,
    SUPPORTED_TASKS,
    VARIATION_AXES,
)
from kleos_training_data.scenarios.rendering import PROMPT_FORMATS

#: Sampling strategies over the axis space.
SAMPLING_STRATEGIES: tuple[str, ...] = ("grid", "stratified")

#: Review lanes a scenario can demand.
REVIEW_LANES: tuple[str, ...] = ("mock_only", "llm_then_human", "human_only")


class StrictModel(BaseModel):
    """Base with unknown fields rejected.

    A typo in a scenario file must be an error. Silently ignoring
    ``pertubation_kinds`` would produce a catalog that looks like it tests
    consistency and does not.
    """

    model_config = ConfigDict(extra="forbid")


class EntitySpec(StrictModel):
    """Where the fictional names in a situation come from."""

    pool: str = Field(description="Name of a pool under data/surrogates/.")
    count: int = Field(default=3, ge=2, le=8)


class EquivalenceGroupSpec(StrictModel):
    """A perturbation that must not change the decision.

    "Must not change the decision" is checked, not asserted:
    ``validate_scenarios.py`` generates each perturbation and compares its
    computed decision against the base. A perturbation that legitimately changes
    the answer is a different scenario, and including it here would make
    consistency testing measure noise instead of stability.
    """

    kind: str
    count: int = Field(default=1, ge=1, le=5)

    @field_validator("kind")
    @classmethod
    def _registered(cls, value: str) -> str:
        if value not in PERTURBATION_KINDS:
            raise ValueError(
                f"unknown perturbation kind {value!r}; registered: {list(PERTURBATION_KINDS)}"
            )
        return value


class GenerationSpec(StrictModel):
    """How many situations to sample, and how."""

    n_base: int = Field(default=8, ge=1, le=500)
    seed: int = Field(default=1712)
    sampling: str = Field(default="stratified")
    equivalence_groups: list[EquivalenceGroupSpec] = Field(default_factory=list)

    @field_validator("sampling")
    @classmethod
    def _known_sampling(cls, value: str) -> str:
        if value not in SAMPLING_STRATEGIES:
            raise ValueError(f"unknown sampling {value!r}; valid: {list(SAMPLING_STRATEGIES)}")
        return value


class HoldoutSpec(StrictModel):
    """Axis values reserved for the test split.

    Declared in advance, never discovered by a seed. An OOD claim you cannot
    state before running the split is not an OOD claim — it is a description of
    whatever happened to land on the far side of a hash.
    """

    ood_shift: str | None = None
    reserve_entity_pools: list[str] = Field(default_factory=list)
    reserve_formats: list[str] = Field(default_factory=list)
    reserve_domains: list[str] = Field(default_factory=list)

    @field_validator("ood_shift")
    @classmethod
    def _registered_shift(cls, value: str | None) -> str | None:
        if value is not None and value not in OOD_SHIFT_KINDS:
            raise ValueError(f"unknown ood_shift {value!r}; registered: {list(OOD_SHIFT_KINDS)}")
        return value


class PromptSpec(StrictModel):
    """How the situation is put to the model.

    ``question`` exists because the tasks ask genuinely different things. Asking
    "which should I deal with first" about a set of candidate *tools* would be
    incoherent, and a catalog where every family asks the same question is a
    catalog testing one task seven times.
    """

    question: str = Field(
        default="Which should I deal with first, and why?",
        min_length=8,
    )
    #: What the items represent, for the renderer's framing.
    item_noun: str = "item"


class ExpectedSpec(StrictModel):
    """Which registered policy resolves this family."""

    policy: str
    grader: str = "ranking"


class ReviewSpec(StrictModel):
    """Review requirements for examples generated from this family."""

    lane: str = "llm_then_human"
    min_mean_score: float = Field(default=3.0, ge=0.0, le=4.0)

    @field_validator("lane")
    @classmethod
    def _known_lane(cls, value: str) -> str:
        if value not in REVIEW_LANES:
            raise ValueError(f"unknown review lane {value!r}; valid: {list(REVIEW_LANES)}")
        return value


class Scenario(StrictModel):
    """One scenario family."""

    schema_version: str = "1.0"
    catalog_version: str = "scenarios-v1"
    family: str = Field(min_length=3)
    task: str
    description: str = ""
    policy_claim: str = Field(
        min_length=10, description="What an example from this family teaches."
    )
    anti_claim: str = Field(
        min_length=10,
        description="The surface rule a model might learn instead. What the perturbations catch.",
    )

    axes: dict[str, list[str]] = Field(default_factory=dict)
    entities: EntitySpec
    prompt: PromptSpec = Field(default_factory=PromptSpec)
    expected: ExpectedSpec
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    holdout: HoldoutSpec = Field(default_factory=HoldoutSpec)
    review: ReviewSpec = Field(default_factory=ReviewSpec)

    #: Free-form notes for a human reader. Never enters a training example.
    notes: str | None = None

    @field_validator("task")
    @classmethod
    def _known_task(cls, value: str) -> str:
        if value not in SUPPORTED_TASKS:
            raise ValueError(f"unknown task {value!r}; registered: {list(SUPPORTED_TASKS)}")
        return value

    @field_validator("family")
    @classmethod
    def _family_shape(cls, value: str) -> str:
        # Families become scenario_family metadata and surrogate-map filenames,
        # so they must be filesystem- and id-safe.
        if not all(ch.isalnum() or ch in "._-" for ch in value):
            raise ValueError(f"family {value!r} must contain only alphanumerics, '.', '_' and '-'")
        return value

    @model_validator(mode="after")
    def _axes_are_usable(self) -> Scenario:
        if "domain" not in self.axes or not self.axes["domain"]:
            raise ValueError(
                "every scenario must vary or fix `domain` — it is the one axis the "
                "public contract requires on every example"
            )

        unregistered = sorted(set(self.axes) - set(VARIATION_AXES))
        if unregistered:
            # Permitted, but it must be a decision rather than a typo. The
            # coverage report surfaces these separately.
            self.notes = (self.notes or "") + f"\nPiloting unregistered axes: {unregistered}"

        for value in self.axes.get("format", []):
            if value not in PROMPT_FORMATS:
                raise ValueError(
                    f"format {value!r} has no renderer; available: {list(PROMPT_FORMATS)}"
                )

        for empty in sorted(k for k, v in self.axes.items() if not v):
            raise ValueError(f"axis {empty!r} has no values; remove it or give it one")
        return self

    @property
    def axis_space_size(self) -> int:
        """How many distinct axis combinations exist."""
        total = 1
        for values in self.axes.values():
            total *= len(values)
        return total

    @property
    def examples_per_base(self) -> int:
        """Base example plus its perturbations."""
        return 1 + sum(group.count for group in self.generation.equivalence_groups)

    @property
    def expected_example_count(self) -> int:
        """How many examples generation will produce."""
        return self.generation.n_base * self.examples_per_base

    def to_summary(self) -> dict[str, Any]:
        """Compact description for reports and reviewer packets."""
        return {
            "family": self.family,
            "task": self.task,
            "policy_claim": self.policy_claim,
            "anti_claim": self.anti_claim,
            "axes": {k: len(v) for k, v in self.axes.items()},
            "axis_space": self.axis_space_size,
            "n_base": self.generation.n_base,
            "examples_per_base": self.examples_per_base,
            "expected_examples": self.expected_example_count,
            "policy": self.expected.policy,
            "ood_shift": self.holdout.ood_shift,
        }
