"""Promotion policy, and why ``--force`` cannot reach a privacy gate.

Four independent mechanisms, because one is a convention and four is an
architecture. Any single one could be defeated by a determined refactor; all
four failing silently at once is not a realistic accident.

1. **Mandatory-ness is derived from the gate table** (``gates.py``). A gate is
   mandatory unless somebody explicitly marks it bypassable in a reviewed diff.

2. **An illegal policy cannot be constructed.** :class:`PromotionPolicy` is
   frozen and validates that ``bypass_gate_ids`` contains nothing mandatory. So
   there is no code path anywhere that *holds* an object permitting a privacy
   bypass — the failure happens at construction, not at use.

3. **``--force`` is wired to a constant, not to user input.** It maps to
   :data:`~kleos_training_data.promotion.gates.BYPASSABLE_GATE_IDS`. There is no
   ``--bypass-gate G04`` flag and no config key that reaches ``bypass_gate_ids``.

4. **The runner asserts every mandatory gate actually ran.** That catches the
   case the other three miss: a refactor that deletes a gate from the table, or
   short-circuits the loop on an early failure, so a mandatory gate never
   executes at all and its absence looks like a pass.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kleos_training_data.promotion.gates import BYPASSABLE_GATE_IDS, GATES_BY_ID, MANDATORY_GATE_IDS
from kleos_training_data.review.rubric import DEFAULT_MIN_MEAN_SCORE


class PromotionPolicy(BaseModel):
    """What a promotion run is permitted to skip. Frozen, and validated.

    Constructing one that bypasses a mandatory gate raises. That is the point:
    the illegal object never exists, so no function can be handed one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bypass_gate_ids: frozenset[str] = Field(default_factory=frozenset)
    min_mean_score: float = Field(default=DEFAULT_MIN_MEAN_SCORE, ge=0.0, le=4.0)
    near_duplicate_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    #: Whether a WARN outcome blocks promotion. Off by default: warnings are for
    #: a human to weigh, and treating every one as fatal trains people to bypass.
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
        """Nothing bypassed. The default for a real release."""
        return cls()

    @classmethod
    def forced(cls, **overrides: object) -> PromotionPolicy:
        """What ``--force`` produces: every *bypassable* gate, and nothing else.

        Deliberately takes no gate argument. The caller cannot choose which
        gates to bypass, because the set of gates it is safe to bypass is a
        property of the gate table rather than of the person running the command.
        """
        return cls(bypass_gate_ids=BYPASSABLE_GATE_IDS, **overrides)  # type: ignore[arg-type]
