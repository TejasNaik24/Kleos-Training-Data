#!/usr/bin/env python3
"""Validate the scenario catalog before anything is generated from it.

Checks the things that are silent failures if left unchecked:

* every file parses, and no two files claim the same family
* every named policy and surrogate pool exists
* every declared perturbation is **decision-preserving** — generated and
  compared, not taken on trust
* every perturbation actually perturbs, so an equivalence group cannot be
  quietly full of duplicates
* reserved holdout values are real, so an OOD split cannot come out empty
* the catalog's coverage is reported, so "large but narrow" is visible

    python scripts/validate_scenarios.py
    python scripts/validate_scenarios.py --family notif.deadline_vs_evidence --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.contract.constants import (
    OOD_SHIFT_KINDS,
    PERTURBATION_KINDS,
    SUPPORTED_TASKS,
)
from kleos_training_data.errors import EXIT_GATE_FAILED, EXIT_OK, ScenarioError
from kleos_training_data.ids import example_id
from kleos_training_data.scenarios.generator import generate
from kleos_training_data.scenarios.loader import DEFAULT_CATALOG_DIR, load_catalog
from kleos_training_data.scenarios.models import Scenario
from kleos_training_data.scenarios.surrogates import SurrogatePool, load_pools


def _check_scenario(scenario: Scenario, pools: dict[str, SurrogatePool]) -> list[str]:
    """Return this scenario's problems, empty when it is sound."""
    problems: list[str] = []

    pool = pools.get(scenario.entities.pool)
    if pool is None:
        problems.append(
            f"entities.pool {scenario.entities.pool!r} does not exist "
            f"(available: {', '.join(sorted(pools)) or 'none'})"
        )
        return problems

    if scenario.entities.count > len(pool.values):
        problems.append(
            f"entities.count={scenario.entities.count} exceeds pool "
            f"{pool.name!r} size {len(pool.values)}"
        )

    # Reserved holdout pools must exist, or the OOD split silently has nothing
    # to hold out and the generalization claim evaporates.
    for reserved in scenario.holdout.reserve_entity_pools:
        if reserved not in pools:
            problems.append(f"holdout.reserve_entity_pools names unknown pool {reserved!r}")
        elif reserved == scenario.entities.pool:
            problems.append(
                f"holdout reserves the same pool it generates from ({reserved!r}); "
                f"the test split would not be out of distribution"
            )

    for fmt in scenario.holdout.reserve_formats:
        if fmt not in scenario.axes.get("format", []):
            problems.append(
                f"holdout.reserve_formats names {fmt!r}, which this scenario never "
                f"generates — the OOD test split would be empty"
            )

    for domain in scenario.holdout.reserve_domains:
        if domain not in scenario.axes.get("domain", []):
            problems.append(
                f"holdout.reserve_domains names {domain!r}, which this scenario never generates"
            )

    # A family must not perturb the axis its release holds out on. If it does,
    # a base example and its perturbed variant carry different values of that
    # axis, so a holdout split puts them on opposite sides and the group is
    # broken — which makes consistency testing on that group meaningless.
    format_varying = {"formatting", "schema"}
    declared_kinds = {g.kind for g in scenario.generation.equivalence_groups}
    if scenario.holdout.reserve_formats and (declared_kinds & format_varying):
        problems.append(
            f"reserves format(s) {scenario.holdout.reserve_formats} while also "
            f"declaring format-varying perturbation(s) "
            f"{sorted(declared_kinds & format_varying)}; a group would straddle "
            f"the split boundary"
        )
    if (
        scenario.holdout.reserve_domains
        and "domain" in scenario.axes
        and len(scenario.axes["domain"]) > 1
    ):
        # Domain is not varied by any perturbation kind, so this is safe — noted
        # here so the asymmetry is deliberate rather than an oversight.
        pass

    if scenario.axis_space_size < scenario.generation.n_base:
        problems.append(
            f"n_base={scenario.generation.n_base} exceeds the axis space "
            f"({scenario.axis_space_size}); points would repeat"
        )

    # The real check: generate everything. Decision-preservation and
    # non-duplication are enforced inside generate().
    try:
        candidates = generate(scenario, pool)
    except ScenarioError as exc:
        problems.append(exc.message)
        return problems

    if len(candidates) != scenario.expected_example_count:
        problems.append(
            f"generated {len(candidates)} candidates, expected {scenario.expected_example_count}"
        )

    ids = [example_id(c.to_payload()) for c in candidates]
    if len(set(ids)) != len(ids):
        problems.append(f"{len(ids) - len(set(ids))} generated candidate(s) are duplicates")

    # Consistency testing needs at least two members per group, or it silently
    # reports nothing at all.
    if scenario.generation.equivalence_groups:
        group_sizes = Counter(c.group_id for c in candidates)
        singletons = [g for g, n in group_sizes.items() if n < 2]
        if singletons:
            problems.append(
                f"{len(singletons)} group(s) have a single member, so consistency "
                f"testing cannot measure them"
            )

    return problems


def _coverage(scenarios: list[Scenario]) -> dict[str, object]:
    """Catalog-level coverage. Size is not diversity, and this is the difference."""
    tasks = Counter(s.task for s in scenarios)
    domains: Counter[str] = Counter()
    perturbations: Counter[str] = Counter()
    shifts: Counter[str] = Counter()
    for scenario in scenarios:
        domains.update(scenario.axes.get("domain", []))
        perturbations.update(g.kind for g in scenario.generation.equivalence_groups)
        if scenario.holdout.ood_shift:
            shifts[scenario.holdout.ood_shift] += 1

    return {
        "scenarios": len(scenarios),
        "expected_examples": sum(s.expected_example_count for s in scenarios),
        "tasks_covered": sorted(tasks),
        "tasks_missing": sorted(set(SUPPORTED_TASKS) - set(tasks)),
        "domains": sorted(domains),
        "perturbation_kinds_used": sorted(perturbations),
        "perturbation_kinds_unused": sorted(set(PERTURBATION_KINDS) - set(perturbations)),
        "ood_shifts_used": sorted(shifts),
        "ood_shifts_unused": sorted(set(OOD_SHIFT_KINDS) - set(shifts)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenarios", type=Path, help="Catalog root (default: scenarios/).")
    parser.add_argument("--family", action="append", default=None, help="Limit to a family.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat catalog coverage gaps as failures, not warnings.",
    )
    parser.add_argument("--json", type=Path, help="Write a machine-readable report here.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("SCENARIO CATALOG")

    root = args.scenarios or DEFAULT_CATALOG_DIR
    scenarios = load_catalog(root, families=args.family)
    pools = load_pools()

    print(f"\n  catalog : {root}")
    print(f"  pools   : {', '.join(sorted(pools)) or '(none)'}")
    print(f"  families: {len(scenarios)}\n")

    failures: dict[str, list[str]] = {}
    for scenario in scenarios:
        problems = _check_scenario(scenario, pools)
        icon = "✓" if not problems else "✗"
        summary = scenario.to_summary()
        print(
            f"  {icon} {scenario.family:<38} {scenario.task:<28} "
            f"{summary['expected_examples']:>4} examples"
        )
        for problem in problems:
            print(f"      ✗ {problem}")
        if problems:
            failures[scenario.family] = problems

    coverage = _coverage(scenarios)
    print("\n  ── catalog coverage " + "─" * 50)
    print(f"  scenarios          : {coverage['scenarios']}")
    print(f"  expected examples  : {coverage['expected_examples']}")
    print(f"  tasks covered      : {len(coverage['tasks_covered'])}/{len(SUPPORTED_TASKS)}")
    if coverage["tasks_missing"]:
        print(f"  tasks MISSING      : {', '.join(coverage['tasks_missing'])}")
    print(f"  domains            : {', '.join(coverage['domains'])}")
    print(f"  perturbation kinds : {', '.join(coverage['perturbation_kinds_used']) or 'none'}")
    if coverage["perturbation_kinds_unused"]:
        print(f"  kinds unused       : {', '.join(coverage['perturbation_kinds_unused'])}")
    print(f"  OOD shifts         : {', '.join(coverage['ood_shifts_used']) or 'none'}")

    print(
        "\n  A catalog is not diverse because it is large. Coverage, not count, "
        "\n  is what supports a generalization claim."
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"coverage": coverage, "failures": failures}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  report written to {args.json}")

    if failures:
        print_result(
            False,
            f"{len(failures)} scenario(s) failed validation.",
            hint="Nothing should be generated from a catalog in this state.",
        )
        return EXIT_GATE_FAILED

    if args.strict and coverage["tasks_missing"]:
        print_result(
            False,
            f"{len(coverage['tasks_missing'])} registered task(s) have no scenario.",
            hint="Run without --strict to treat this as a warning.",
        )
        return EXIT_GATE_FAILED

    print_result(
        True,
        f"{len(scenarios)} scenario(s) valid.",
        hint="Next: python scripts/generate_synthetic.py --out-batch <id>",
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
