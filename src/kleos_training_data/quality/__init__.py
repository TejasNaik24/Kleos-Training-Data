"""Corpus health: coverage, balance and dataset-shape reporting.

Advisory rather than gating. A thin coverage cell is a research decision to
make, not an error to fail a build on — the exception being promotion gate G11,
which reads the per-axis result to warn about a pilot axis.
"""

from __future__ import annotations

from kleos_training_data.quality.coverage import (
    CRITICAL,
    DEFAULT_JOINT_AXES,
    DEFAULT_MIN_CELL_COUNT,
    OK,
    WARNING,
    AxisCoverage,
    CoverageReport,
    JointCoverage,
    axis_value,
    build_coverage_report,
    ood_readiness,
    render_coverage_report,
)

__all__ = [
    "CRITICAL",
    "DEFAULT_JOINT_AXES",
    "DEFAULT_MIN_CELL_COUNT",
    "OK",
    "WARNING",
    "AxisCoverage",
    "CoverageReport",
    "JointCoverage",
    "axis_value",
    "build_coverage_report",
    "ood_readiness",
    "render_coverage_report",
]
