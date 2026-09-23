from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.contract.constants import SPLIT_STRATEGIES
from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.contract.splitting import SplitConfig, split_examples
from kleos_training_data.datasets.holdouts import resolve_holdouts
from kleos_training_data.datasets.manifest import build_provenance
from kleos_training_data.datasets.release import ReleaseWriter
from kleos_training_data.errors import EXIT_ERROR, EXIT_OK
from kleos_training_data.paths import Workspace
from kleos_training_data.scenarios.generator import generation_fingerprint
from kleos_training_data.scenarios.loader import DEFAULT_CATALOG_DIR, load_catalog
from kleos_training_data.staging.records import PromotedExample
from kleos_training_data.staging.store import iter_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. kleos-policy-v0.1.0")
    parser.add_argument(
        "--strategy",
        choices=sorted(SPLIT_STRATEGIES),
        help="Split strategy. Defaults to whatever the catalog's holdouts imply.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--group-key", help="Metadata key defining a group.")
    parser.add_argument(
        "--holdout-attribute",
        help="Force one holdout attribute when the catalog declares several.",
    )
    parser.add_argument(
        "--allow-uncovered-holdout",
        action="store_true",
        help="Permit a reserved value with no examples. Produces an empty OOD split.",
    )
    parser.add_argument("--description", default="", help="One line for the manifest.")
    parser.add_argument(
        "--declare-private",
        action="store_true",
        help="Set contains_private_data=true. One-way: it can never be set false by hand.",
    )
    parser.add_argument("--scenarios", type=Path, help="Catalog root.")
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the split, write nothing.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("BUILD RELEASE")

    workspace = Workspace.from_env(args.workspace)
    workspace.assert_initialized()

    examples = [
        TrainingExample.model_validate(record.example)
        for record in iter_records(workspace.staging / "promoted", PromotedExample)
    ]
    if not examples:
        print("\n  No promoted examples found.")
        print_result(False, "Nothing to release.")
        return EXIT_ERROR

    scenarios = load_catalog(args.scenarios or DEFAULT_CATALOG_DIR)
    holdout = resolve_holdouts(
        examples,
        scenarios,
        attribute=args.holdout_attribute,
        require_coverage=not args.allow_uncovered_holdout,
    )

    strategy = args.strategy or holdout.strategy or "group"

    print(f"\n  version   : {args.version}")
    print(f"  examples  : {len(examples)} promoted")
    print(f"  strategy  : {strategy}")
    print(f"  seed      : {args.seed}")
    print(
        f"  fractions : {args.train_fraction}/{args.validation_fraction}/{args.test_fraction} "
        f"(stated explicitly)"
    )
    if holdout.declared:
        print(f"  holding out {holdout.attribute}: {', '.join(holdout.values)}")
        print(f"  OOD shifts: {', '.join(holdout.ood_shifts) or '(none declared)'}")
    for note in holdout.notes:
        print(f"    · {note}")

    config = SplitConfig(
        strategy=strategy,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        group_key=args.group_key,
        holdout_values=list(holdout.values),
    )
    split = split_examples(examples, config)

    print(f"\n  split     : {split.counts}")
    for note in split.notes:
        print(f"    · {note}")

    if not split.test:
        print("\n  ! The test split is empty. No generalization claim can be made.")

    if args.dry_run:
        print_result(True, "Dry run — nothing written.")
        return EXIT_OK

    fingerprints = {s.family: generation_fingerprint(s) for s in scenarios}

    writer = ReleaseWriter(workspace)
    sealed = writer.seal(
        version=args.version,
        split=split,
        description=args.description,
        declare_private=args.declare_private,
        provenance_builder=lambda manifest: build_provenance(
            version=args.version,
            split=split,
            holdout=holdout,
            manifest=manifest,
            examples=examples,
            scenario_fingerprints=fingerprints,
        ),
    )

    print(f"\n  sealed    : {sealed.path}")
    print(f"  content   : {sealed.content_hash[:16]}…")
    for filename, digest in sorted(sealed.file_hashes.items()):
        print(f"    {filename:<20} {digest[:16]}…")

    print_result(
        True,
        f"Release {args.version} sealed.",
        hint=f"Next: python scripts/verify_release.py --release {sealed.path} --strict",
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
