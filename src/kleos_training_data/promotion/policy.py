from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kleos_training_data.promotion.gates import BYPASSABLE_GATE_IDS, GATES_BY_ID, MANDATORY_GATE_IDS
from kleos_training_data.review.rubric import DEFAULT_MIN_MEAN_SCORE


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bypass_gate_ids: frozenset[str] = Field(default_factory=frozenset)
    min_mean_score: float = Field(default=DEFAULT_MIN_MEAN_SCORE, ge=0.0, le=4.0)
    near_duplicate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    strict_warnings: bool = False

    @model_validator(mode="after")
    def _no_mandatory_bypass(self) -> PromotionPolicy:
        forbidden = sorted(self.bypass_gate_ids & MANDATORY_GATE_IDS)
        if forbidden:
            raise ValueError(
                f"gate(s) {forbidden} are mandatory and cannot be bypassed. "
                f"Bypassable gates are: {sorted(BYPASSABLE_GATE_IDS)}. Privacy and "
                f"schema gates are not among them by construction — fix the "
                f"candidate instead."
            )
        return self

    @model_validator(mode="after")
    def _bypass_names_real_gates(self) -> PromotionPolicy:
        unknown = sorted(self.bypass_gate_ids - set(GATES_BY_ID))
        if unknown:
            raise ValueError(
                f"bypass_gate_ids names gate(s) that do not exist: {unknown}. A "
                f"typo here would silently bypass nothing while looking like it "
                f"bypassed something."
            )
        return self

    def bypasses(self, gate_id: str) -> bool:
        return gate_id in self.bypass_gate_ids

    @classmethod
    def strict(cls) -> PromotionPolicy:
        return cls()

    @classmethod
    def forced(cls, **overrides: Any) -> PromotionPolicy:
        return cls(bypass_gate_ids=BYPASSABLE_GATE_IDS, **overrides)
