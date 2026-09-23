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

SAMPLING_STRATEGIES: tuple[str, ...] = ("grid", "stratified")

REVIEW_LANES: tuple[str, ...] = ("mock_only", "llm_then_human", "human_only")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntitySpec(StrictModel):
    pool: str = Field(description="Name of a pool under data/surrogates/.")
    count: int = Field(default=3, ge=2, le=8)


class EquivalenceGroupSpec(StrictModel):
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
    question: str = Field(
        default="Which should I deal with first, and why?",
        min_length=8,
    )
    item_noun: str = "item"
    framing: str = "priority"
    need: str = ""
    workspace_names: list[str] = Field(default_factory=list)
    out_of_scope_count: int = Field(default=0, ge=0, le=4)
    out_of_scope_stronger: bool = False
    request_ambiguous: bool = False
    stale_explicit_conflict: bool = False

    @field_validator("framing")
    @classmethod
    def _known_framing(cls, value: str) -> str:
        from kleos_training_data.scenarios.situations import FRAMINGS

        if value not in FRAMINGS:
            raise ValueError(f"unknown framing {value!r}; registered: {sorted(FRAMINGS)}")
        return value


class ExpectedSpec(StrictModel):
    policy: str
    grader: str = "ranking"


class ReviewSpec(StrictModel):
    lane: str = "llm_then_human"
    min_mean_score: float = Field(default=3.0, ge=0.0, le=4.0)

    @field_validator("lane")
    @classmethod
    def _known_lane(cls, value: str) -> str:
        if value not in REVIEW_LANES:
            raise ValueError(f"unknown review lane {value!r}; valid: {list(REVIEW_LANES)}")
        return value


class Scenario(StrictModel):
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
            self.notes = (self.notes or "") + f"\nPiloting unregistered axes: {unregistered}"

        for value in self.axes.get("format", []):
            if value not in PROMPT_FORMATS:
                raise ValueError(
                    f"format {value!r} has no renderer; available: {list(PROMPT_FORMATS)}"
                )

        for empty in sorted(k for k, v in self.axes.items() if not v):
            raise ValueError(f"axis {empty!r} has no values; remove it or give it one")

        if self.prompt.framing == "workspace":
            if not self.prompt.workspace_names:
                raise ValueError(
                    "the workspace framing needs `prompt.workspace_names` — a prompt "
                    "that says 'only in the workspace I named' while naming no "
                    "workspace has no stated correct answer"
                )
            declared = self.axes.get("workspace")
            if not declared:
                raise ValueError(
                    "a workspace-framed scenario must declare a `workspace` axis; the "
                    "rendered workspace name is taken from it, so the metadata and the "
                    "prompt cannot disagree"
                )
            unknown = sorted(set(declared) - set(self.prompt.workspace_names))
            if unknown:
                raise ValueError(
                    f"axes.workspace values {unknown} are not in "
                    f"prompt.workspace_names {self.prompt.workspace_names}; the axis "
                    f"value is what the prompt renders, so every value must be a real "
                    f"workspace name"
                )
            if self.prompt.out_of_scope_count >= self.entities.count:
                raise ValueError(
                    f"out_of_scope_count={self.prompt.out_of_scope_count} leaves no "
                    f"item inside the active workspace (entities.count="
                    f"{self.entities.count}); there would be nothing to answer with"
                )
        elif self.prompt.out_of_scope_count:
            raise ValueError(
                f"out_of_scope_count is set but framing is {self.prompt.framing!r}; "
                f"only the workspace framing marks items out of scope"
            )
        return self

    @property
    def axis_space_size(self) -> int:
        total = 1
        for values in self.axes.values():
            total *= len(values)
        return total

    @property
    def examples_per_base(self) -> int:
        return 1 + sum(group.count for group in self.generation.equivalence_groups)

    @property
    def expected_example_count(self) -> int:
        return self.generation.n_base * self.examples_per_base

    def to_summary(self) -> dict[str, Any]:
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
