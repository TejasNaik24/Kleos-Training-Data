from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kleos_training_data.errors import PromotionIntegrityError
from kleos_training_data.promotion.context import PromotionContext
from kleos_training_data.promotion.gates import (
    GATES,
    MANDATORY_GATE_IDS,
    PRIVACY_GATE_IDS,
    GateOutcome,
    GateStatus,
)
from kleos_training_data.promotion.policy import PromotionPolicy


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: GateStatus
    message: str
    evidence: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class PromotionReport:
    candidate_id: str
    results: list[GateResult] = field(default_factory=list)

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if r.status is GateStatus.FAIL]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if r.status is GateStatus.WARN]

    @property
    def bypassed(self) -> list[GateResult]:
        return [r for r in self.results if r.status is GateStatus.BYPASSED]

    @property
    def privacy_failures(self) -> list[GateResult]:
        return [r for r in self.failures if r.gate_id in PRIVACY_GATE_IDS]

    def ok(self, *, strict_warnings: bool = False) -> bool:
        if self.failures:
            return False
        return not (strict_warnings and self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "ok": self.ok(),
            "failures": [r.gate_id for r in self.failures],
            "warnings": [r.gate_id for r in self.warnings],
            "bypassed": [r.gate_id for r in self.bypassed],
            "results": [r.to_dict() for r in self.results],
        }

    def render(self) -> str:
        icons = {
            GateStatus.PASS: "✓",
            GateStatus.FAIL: "✗",
            GateStatus.WARN: "!",
            GateStatus.BYPASSED: "·",
        }
        lines = [f"  {self.candidate_id}"]
        for result in self.results:
            lines.append(f"    {icons[result.status]} {result.gate_id:<26} {result.message}")
            for key, value in sorted(result.evidence.items()):
                lines.append(f"        {key}: {value}")
        return "\n".join(lines)


def run_gates(ctx: PromotionContext, policy: PromotionPolicy) -> PromotionReport:
    report = PromotionReport(candidate_id=ctx.candidate.candidate_id)
    executed: set[str] = set()

    for spec in GATES:
        executed.add(spec.id)

        if policy.bypasses(spec.id):
            report.results.append(GateResult(spec.id, GateStatus.BYPASSED, "bypassed by policy"))
            continue

        try:
            outcome: GateOutcome = spec.check(ctx)
        except Exception as exc:
            outcome = GateOutcome(
                GateStatus.FAIL,
                f"the gate raised {type(exc).__name__}",
                {"error": str(exc)[:200]},
            )

        report.results.append(
            GateResult(spec.id, outcome.status, outcome.message, dict(outcome.evidence))
        )

    missing = MANDATORY_GATE_IDS - executed
    if missing:
        raise PromotionIntegrityError(
            f"{len(missing)} mandatory gate(s) did not execute.",
            details={"missing": ", ".join(sorted(missing))},
            suggestions=[
                "This is a bug in the pipeline, never a property of the data.",
                "A mandatory gate that does not run is indistinguishable from one "
                "that passed, which is why this is checked rather than assumed.",
                "Do not downgrade this to a warning.",
            ],
        )

    bypassed_mandatory = {r.gate_id for r in report.bypassed} & MANDATORY_GATE_IDS
    if bypassed_mandatory:
        raise PromotionIntegrityError(
            f"mandatory gate(s) {sorted(bypassed_mandatory)} were bypassed.",
            suggestions=["PromotionPolicy should have made this impossible to construct."],
        )

    return report
