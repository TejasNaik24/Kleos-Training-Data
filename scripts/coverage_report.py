#!/usr/bin/env python3
"""Report what a corpus actually covers.

    python scripts/coverage_report.py --release releases/kleos-policy-v0.1.0
    python scripts/coverage_report.py --promoted --json reports/coverage/latest.json

A count is not a research claim. This reports the joint structure — task x
domain, task x urgency, and so on — and names the cells that are empty or thin.

Advisory by default: exits 0 whatever it finds, because a thin cell is a decision
to make rather than an error. ``--fail-on`` makes it a gate when you want one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.contract.writer import read_examples
from kleos_training_data.errors import EXIT_ERROR, EXIT_GATE_FAILED, EXIT_OK
from kleos_training_data.paths import Workspace
from kleos_training_data.quality.coverage import (
    CRITICAL,
    WARNING,
    build_coverage_report,
    ood_readiness,
    render_coverage_report,
)
from kleos_training_data.staging.records import PromotedExample
from kleos_training_data.staging.store import iter_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release", type=Path, help="A sealed release directory.")
    source.add_argument("--promoted", action="store_true", help="The promoted pool.")
    parser.add_argument("--min-cell-count", type=int, default=3)
    parser.add_argument(
        "--fail-on",
        choices=[CRITICAL, WARNING],
        help="Exit 3 at or above this severity. Off by default.",
    )
    parser.add_argument("--json", type=Path, help="Write a machine-readable report here.")
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("COVERAGE")

    workspace = Workspace.from_env(args.workspace)
    if args.release:
        examples: list[TrainingExample] = []
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            path = args.release / name
            if path.is_file():
                examples.extend(read_examples(path))
        label = str(args.release)
    else:
        examples = [
            TrainingExample.model_validate(record.example)
            for record in iter_records(workspace.staging / "promoted", PromotedExample)
        ]
        label = "the promoted pool"

    if not examples:
        print(f"\n  No examples found in {label}.")
        print_result(False, "Nothing to report on.")
        return EXIT_ERROR

    report = build_coverage_report(examples, min_cell_count=args.min_cell_count)
    print(f"\n  source   : {label}")
    print(render_coverage_report(report))

    readiness = ood_readiness(examples)
    print("\n  ── OOD readiness " + "─" * 51)
    for shift in ("unseen_formats", "unseen_entities", "unseen_domains"):
        print(f"    {'✓' if readiness[shift] else '✗'} {shift}")
    print(f"    · {readiness['note']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"source": label, "coverage": report.to_dict(), "ood_readiness": readiness},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  report written to {args.json}")

    if args.fail_on:
        ranked = {CRITICAL: 2, WARNING: 1, "OK": 0}
        if ranked[report.severity] >= ranked[args.fail_on]:
            print_result(False, f"Coverage severity is {report.severity}.")
            return EXIT_GATE_FAILED

    print_result(True, f"Coverage reported for {len(examples)} example(s).")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
