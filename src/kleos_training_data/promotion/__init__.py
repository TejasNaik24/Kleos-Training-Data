"""Promotion: the only path from staging into a dataset.

Fourteen gates, all of which run. Eleven are mandatory and cannot be bypassed by
any flag, config key or policy object — see :mod:`.policy` for the four
independent mechanisms that make that true rather than merely stated.
"""

from __future__ import annotations

from kleos_training_data.promotion.context import PromotionContext
from kleos_training_data.promotion.gates import (
    BYPASSABLE_GATE_IDS,
    GATES,
    GATES_BY_ID,
    MANDATORY_GATE_IDS,
    PRIVACY_GATE_IDS,
    GateOutcome,
    GateSpec,
    GateStatus,
)
from kleos_training_data.promotion.policy import PromotionPolicy
from kleos_training_data.promotion.runner import GateResult, PromotionReport, run_gates

__all__ = [
    "BYPASSABLE_GATE_IDS",
    "GATES",
    "GATES_BY_ID",
    "MANDATORY_GATE_IDS",
    "PRIVACY_GATE_IDS",
    "GateOutcome",
    "GateResult",
    "GateSpec",
    "GateStatus",
    "PromotionContext",
    "PromotionPolicy",
    "PromotionReport",
    "run_gates",
]
