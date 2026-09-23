from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from kleos_training_data.contract.constants import OOD_SHIFT_KINDS
from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.errors import ConfigError
from kleos_training_data.scenarios.models import Scenario

_ATTRIBUTE_BY_DECLARATION: dict[str, str] = {
    "reserve_entity_pools": "entities",
    "reserve_formats": "format",
    "reserve_domains": "domain",
}

STRATEGY_BY_ATTRIBUTE: dict[str, str] = {
    "entities": "entity_holdout",
    "format": "format_holdout",
    "domain": "domain_holdout",
}


@dataclass
class HoldoutPlan:
    attribute: str | None = None
    values: list[str] = field(default_factory=list)
    ood_shifts: list[str] = field(default_factory=list)
    declared_by: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def strategy(self) -> str | None:
        return STRATEGY_BY_ATTRIBUTE.get(self.attribute or "")

    @property
    def declared(self) -> bool:
        return bool(self.values)

    def to_dict(self) -> dict[str, object]:
        return {
            "attribute": self.attribute,
            "values": sorted(self.values),
            "ood_shifts": sorted(self.ood_shifts),
            "declared_by": {k: sorted(v) for k, v in sorted(self.declared_by.items())},
            "strategy": self.strategy,
            "notes": self.notes,
        }


def collect_declarations(scenarios: list[Scenario]) -> dict[str, dict[str, list[str]]]:
    collected: dict[str, dict[str, list[str]]] = {}
    for scenario in scenarios:
        for declaration, attribute in _ATTRIBUTE_BY_DECLARATION.items():
            for value in getattr(scenario.holdout, declaration, []) or []:
                collected.setdefault(attribute, {}).setdefault(value, []).append(scenario.family)
    return collected


def resolve_holdouts(
    examples: list[TrainingExample],
    scenarios: list[Scenario],
    *,
    attribute: str | None = None,
    require_coverage: bool = True,
) -> HoldoutPlan:
    declarations = collect_declarations(scenarios)

    if attribute is None:
        if len(declarations) > 1:
            raise ConfigError(
                f"The catalog declares holdouts on {sorted(declarations)}.",
                details={"attributes": ", ".join(sorted(declarations))},
                suggestions=[
                    "Holding out two attributes at once confounds the result: a drop "
                    "in test performance cannot be attributed to either shift.",
                    "Pass --holdout-attribute to choose one for this release.",
                ],
            )
        attribute = next(iter(declarations), None)

    if attribute is None:
        return HoldoutPlan(notes=["The catalog declares no holdout; splitting by group."])

    reserved = declarations.get(attribute, {})
    if not reserved:
        return HoldoutPlan(
            attribute=attribute,
            notes=[f"No scenario reserves a {attribute} value."],
        )

    observed = Counter(_attribute_of(example, attribute) for example in examples)

    uncovered = sorted(value for value in reserved if observed.get(value, 0) == 0)
    if uncovered and require_coverage:
        raise ConfigError(
            f"{len(uncovered)} reserved {attribute} value(s) have no examples.",
            details={
                "uncovered": ", ".join(uncovered),
                "observed": ", ".join(sorted(observed)) or "(none)",
            },
            suggestions=[
                "A reserved value with no coverage produces a silently empty OOD "
                "test set — which is worse than no OOD test set, because the "
                "release still looks complete.",
                "Generate examples covering it, or drop the reservation.",
            ],
        )

    covered = [value for value in sorted(reserved) if observed.get(value, 0) > 0]
    if covered and len(covered) >= len(observed):
        raise ConfigError(
            f"Reserving {covered} would hold out every {attribute} value.",
            details={"observed": ", ".join(sorted(observed))},
            suggestions=["There would be no training data left."],
        )

    shifts = sorted(
        {scenario.holdout.ood_shift for scenario in scenarios if scenario.holdout.ood_shift}
    )
    unregistered = [s for s in shifts if s not in OOD_SHIFT_KINDS]
    if unregistered:
        raise ConfigError(
            f"Unregistered OOD shift kind(s): {unregistered}",
            details={"registered": ", ".join(OOD_SHIFT_KINDS)},
        )

    held_out_count = sum(observed[value] for value in covered)
    return HoldoutPlan(
        attribute=attribute,
        values=covered,
        ood_shifts=shifts,
        declared_by={value: sorted(set(reserved[value])) for value in covered},
        notes=[
            f"Holding out {len(covered)} {attribute} value(s) covering "
            f"{held_out_count} of {len(examples)} example(s).",
            "Validation is drawn from seen values, so early stopping does not leak "
            "the OOD signal the test split exists to measure.",
        ],
    )


def _attribute_of(example: TrainingExample, attribute: str) -> str:
    if attribute == "domain":
        return example.variation_axes.domain
    if attribute == "entities":
        return str(example.variation_axes.entities or "<unspecified>")
    if attribute == "format":
        return str(example.variation_axes.format or "<unspecified>")
    value = example.variation_axes.as_dict().get(attribute)
    return str(value) if value is not None else "<unspecified>"
