from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from kleos_training_data.contract.constants import (
    OOD_SHIFT_KINDS,
    PERTURBATION_KINDS,
    SUPPORTED_TASKS,
    VARIATION_AXES,
)
from kleos_training_data.contract.schemas import TrainingExample

DEFAULT_JOINT_AXES: tuple[tuple[str, str], ...] = (
    ("task", "domain"),
    ("task", "urgency"),
    ("task", "evidence_quality"),
    ("task", "conflicting_evidence"),
    ("task", "difficulty"),
    ("task", "ambiguity"),
    ("task", "format"),
    ("task", "context_length"),
)

DEFAULT_MIN_CELL_COUNT: int = 3

CRITICAL = "CRITICAL"
WARNING = "WARNING"
OK = "OK"


def axis_value(example: TrainingExample, axis: str) -> str | None:
    if axis == "task":
        return example.task
    value = example.variation_axes.as_dict().get(axis)
    return str(value) if value is not None else None


@dataclass
class AxisCoverage:
    axis: str
    counts: dict[str, int] = field(default_factory=dict)
    missing: int = 0

    @property
    def distinct(self) -> int:
        return len(self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def is_constant(self) -> bool:
        return self.distinct == 1 and self.total > 1

    @property
    def imbalance(self) -> float:
        if not self.counts:
            return 0.0
        low = min(self.counts.values())
        return max(self.counts.values()) / low if low else float("inf")

    @property
    def severity(self) -> str:
        if self.is_constant:
            return CRITICAL
        if not self.counts:
            return WARNING
        if self.imbalance >= 5.0:
            return WARNING
        return OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "distinct": self.distinct,
            "total": self.total,
            "missing": self.missing,
            "counts": dict(sorted(self.counts.items())),
            "imbalance": round(self.imbalance, 2),
            "is_constant": self.is_constant,
            "severity": self.severity,
        }


@dataclass
class JointCoverage:
    axes: tuple[str, str]
    cells: dict[str, int] = field(default_factory=dict)
    empty: list[str] = field(default_factory=list)
    thin: dict[str, int] = field(default_factory=dict)

    @property
    def observed(self) -> int:
        return len(self.cells)

    @property
    def possible(self) -> int:
        return self.observed + len(self.empty)

    @property
    def fill_rate(self) -> float:
        return self.observed / self.possible if self.possible else 0.0

    @property
    def severity(self) -> str:
        if self.possible <= 1:
            return WARNING
        if self.fill_rate < 0.5:
            return WARNING
        if self.thin:
            return WARNING
        return OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": list(self.axes),
            "observed_cells": self.observed,
            "possible_cells": self.possible,
            "fill_rate": round(self.fill_rate, 3),
            "empty_cells": sorted(self.empty),
            "thin_cells": dict(sorted(self.thin.items())),
            "severity": self.severity,
        }


@dataclass
class CoverageReport:
    example_count: int = 0
    axes: dict[str, AxisCoverage] = field(default_factory=dict)
    joint: list[JointCoverage] = field(default_factory=list)
    scenario_families: dict[str, int] = field(default_factory=dict)
    groups: dict[str, int] = field(default_factory=dict)
    perturbations: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    quality: dict[str, int] = field(default_factory=dict)
    message_lengths: dict[str, float] = field(default_factory=dict)
    unregistered_axes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def tasks_covered(self) -> list[str]:
        return sorted(self.axes.get("task", AxisCoverage("task")).counts)

    @property
    def tasks_missing(self) -> list[str]:
        return sorted(set(SUPPORTED_TASKS) - set(self.tasks_covered))

    @property
    def perturbation_kinds_unused(self) -> list[str]:
        return sorted(set(PERTURBATION_KINDS) - set(self.perturbations))

    @property
    def consistency_measurable(self) -> bool:
        return any(size >= 2 for size in self.groups.values())

    @property
    def severity(self) -> str:
        levels = [a.severity for a in self.axes.values()] + [j.severity for j in self.joint]
        if not self.consistency_measurable and self.example_count > 1:
            levels.append(CRITICAL)
        if CRITICAL in levels:
            return CRITICAL
        if WARNING in levels:
            return WARNING
        return OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "severity": self.severity,
            "tasks_covered": self.tasks_covered,
            "tasks_missing": self.tasks_missing,
            "consistency_measurable": self.consistency_measurable,
            "axes": {name: cov.to_dict() for name, cov in sorted(self.axes.items())},
            "joint": [j.to_dict() for j in self.joint],
            "scenario_families": dict(sorted(self.scenario_families.items())),
            "group_sizes": dict(sorted(Counter(self.groups.values()).items())),
            "perturbations": dict(sorted(self.perturbations.items())),
            "perturbation_kinds_unused": self.perturbation_kinds_unused,
            "sources": dict(sorted(self.sources.items())),
            "quality": dict(sorted(self.quality.items())),
            "message_lengths": self.message_lengths,
            "unregistered_axes": self.unregistered_axes,
            "notes": self.notes,
        }


def build_coverage_report(
    examples: list[TrainingExample],
    *,
    joint_axes: tuple[tuple[str, str], ...] = DEFAULT_JOINT_AXES,
    min_cell_count: int = DEFAULT_MIN_CELL_COUNT,
) -> CoverageReport:
    report = CoverageReport(example_count=len(examples))
    if not examples:
        report.notes.append("The corpus is empty.")
        return report

    observed_axes: set[str] = {"task"}
    for example in examples:
        observed_axes.update(example.variation_axes.as_dict())

    for axis in sorted(observed_axes):
        counts: Counter[str] = Counter()
        missing = 0
        for example in examples:
            value = axis_value(example, axis)
            if value is None:
                missing += 1
            else:
                counts[value] += 1
        report.axes[axis] = AxisCoverage(axis=axis, counts=dict(counts), missing=missing)

    report.unregistered_axes = sorted(observed_axes - set(VARIATION_AXES))

    for left, right in joint_axes:
        if left not in report.axes or right not in report.axes:
            continue
        cells: Counter[str] = Counter()
        for example in examples:
            a, b = axis_value(example, left), axis_value(example, right)
            if a is None or b is None:
                continue
            cells[f"{left}={a}|{right}={b}"] += 1

        possible = [
            f"{left}={a}|{right}={b}"
            for a, b in product(sorted(report.axes[left].counts), sorted(report.axes[right].counts))
        ]
        report.joint.append(
            JointCoverage(
                axes=(left, right),
                cells=dict(cells),
                empty=[cell for cell in possible if cell not in cells],
                thin={k: v for k, v in cells.items() if v < min_cell_count},
            )
        )

    report.scenario_families = dict(
        sorted(Counter(e.metadata.scenario_family or "(none)" for e in examples).items())
    )
    report.groups = dict(Counter(e.group_key() for e in examples))
    report.perturbations = dict(
        sorted(Counter(e.metadata.perturbation_kind or "(base)" for e in examples).items())
    )
    report.sources = dict(sorted(Counter(e.metadata.source for e in examples).items()))
    report.quality = dict(sorted(Counter(e.metadata.quality_status for e in examples).items()))

    lengths = [len(m.content) for e in examples for m in e.messages]
    turns = [len(e.messages) for e in examples]
    report.message_lengths = {
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "max_chars": float(max(lengths)),
        "mean_turns": round(sum(turns) / len(turns), 2),
    }
    traces = [len(m.reasoning) for e in examples for m in e.messages if m.reasoning is not None]
    if traces:
        report.message_lengths["reasoning_traces"] = float(len(traces))
        report.message_lengths["reasoning_mean_chars"] = round(sum(traces) / len(traces), 1)
        report.message_lengths["reasoning_max_chars"] = float(max(traces))

    if not report.consistency_measurable:
        report.notes.append(
            "Every group has one member, so consistency testing can measure "
            "nothing. Generate perturbations sharing a group_id."
        )
    if report.tasks_missing:
        report.notes.append(
            f"{len(report.tasks_missing)} registered task(s) have no examples: "
            f"{', '.join(report.tasks_missing)}"
        )
    if report.perturbation_kinds_unused:
        report.notes.append(
            f"Perturbation kind(s) never used: {', '.join(report.perturbation_kinds_unused)}"
        )
    return report


def render_coverage_report(report: CoverageReport, *, max_rows: int = 12) -> str:
    icons = {CRITICAL: "✗", WARNING: "!", OK: "✓"}
    lines = [
        f"  examples : {report.example_count}",
        f"  verdict  : {icons[report.severity]} {report.severity}",
        "",
        f"  tasks    : {len(report.tasks_covered)}/{len(SUPPORTED_TASKS)} covered",
    ]
    if report.tasks_missing:
        lines.append(f"  missing  : {', '.join(report.tasks_missing)}")

    lines += ["", "  ── per-axis " + "─" * 56]
    for name, cov in sorted(report.axes.items()):
        flag = icons[cov.severity]
        detail = f"{cov.distinct} value(s)"
        if cov.is_constant:
            detail += f" — constant {next(iter(cov.counts))!r}, no claim possible"
        elif cov.imbalance >= 5.0:
            detail += f" — imbalance {cov.imbalance:.1f}x"
        lines.append(f"    {flag} {name:<22} {detail}")

    lines += ["", "  ── joint coverage " + "─" * 50]
    for joint in report.joint:
        flag = icons[joint.severity]
        lines.append(
            f"    {flag} {joint.axes[0]} x {joint.axes[1]:<20} "
            f"{joint.observed}/{joint.possible} cells ({joint.fill_rate:.0%})"
        )
        for cell in sorted(joint.empty)[:max_rows]:
            lines.append(f"        empty: {cell}")
        if len(joint.empty) > max_rows:
            lines.append(f"        … {len(joint.empty) - max_rows} more empty cell(s)")

    lines += [
        "",
        "  ── structure " + "─" * 55,
        f"    scenario families : {len(report.scenario_families)}",
        f"    groups            : {len(report.groups)}",
        f"    consistency       : {'measurable' if report.consistency_measurable else 'NOT measurable'}",
        f"    perturbations     : {', '.join(k for k in report.perturbations if k != '(base)') or 'none'}",
        f"    sources           : {report.sources}",
        f"    quality           : {report.quality}",
        f"    mean turns        : {report.message_lengths.get('mean_turns', 0)}",
    ]
    if report.message_lengths.get("reasoning_traces"):
        lines.append(
            f"    reasoning traces  : {int(report.message_lengths['reasoning_traces'])} "
            f"(mean {report.message_lengths['reasoning_mean_chars']}, "
            f"max {int(report.message_lengths['reasoning_max_chars'])} chars)"
        )

    if report.notes:
        lines += ["", "  ── notes " + "─" * 59]
        lines += [f"    · {note}" for note in report.notes]

    lines += [
        "",
        "  A dataset is not diverse because it is large. Coverage, not count,",
        "  is what supports a generalization claim.",
    ]
    return "\n".join(lines)


def ood_readiness(examples: list[TrainingExample]) -> dict[str, Any]:
    formats = {axis_value(e, "format") for e in examples} - {None}
    entities = {axis_value(e, "entities") for e in examples} - {None}
    domains = {axis_value(e, "domain") for e in examples} - {None}
    return {
        "unseen_formats": len(formats) >= 2,
        "unseen_entities": len(entities) >= 2,
        "unseen_domains": len(domains) >= 2,
        "registered_shifts": list(OOD_SHIFT_KINDS),
        "note": (
            "A holdout needs at least two distinct values on its attribute: one "
            "to hold out and one to train on."
        ),
    }
