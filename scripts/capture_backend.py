from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.collection.adapters import resolve_adapter
from kleos_training_data.collection.guard import assert_capture_allowed
from kleos_training_data.collection.runner import run_batch
from kleos_training_data.errors import EXIT_ERROR, EXIT_OK
from kleos_training_data.paths import Workspace
from kleos_training_data.scenarios.loader import DEFAULT_CATALOG_DIR, load_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="mock", help="Backend adapter (default: mock).")
    parser.add_argument("--out-batch", required=True, metavar="ID", help="Batch identifier.")
    parser.add_argument("--scenarios", type=Path, help="Catalog root (default: scenarios/).")
    parser.add_argument("--family", action="append", default=None, help="Limit to a family.")
    parser.add_argument("--limit", type=int, help="Stop after N requests.")
    parser.add_argument(
        "--base-url",
        default="mock://local",
        help="Backend base URL. Non-local URLs require the production guard.",
    )
    parser.add_argument(
        "--allow-production",
        action="store_true",
        help="One of four conditions required to capture from a real deployment.",
    )
    parser.add_argument("--confirm", help="The production confirmation phrase, typed in full.")
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    parser.add_argument("--dry-run", action="store_true", help="Report the plan and stop.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("CAPTURE")

    workspace = Workspace.from_env(args.workspace)
    workspace.assert_initialized()

    scenarios = load_catalog(args.scenarios or DEFAULT_CATALOG_DIR, families=args.family)
    adapter = resolve_adapter(args.adapter)
    expected = sum(s.expected_example_count for s in scenarios)

    print(f"\n  adapter  : {adapter.name} (lane={adapter.lane.value})")
    print(f"  endpoint : {adapter.endpoint()}")
    print(f"  batch    : {args.out_batch}")
    print(f"  families : {len(scenarios)}")
    print(f"  expected : {expected} request(s)")

    authorization = assert_capture_allowed(
        args.base_url,
        allow_production=args.allow_production,
        confirm=args.confirm,
        env=dict(os.environ),
        scenario_families=tuple(s.family for s in scenarios),
        expected_captures=expected,
    )
    if authorization is not None:
        print("\n  ! PRODUCTION CAPTURE AUTHORIZED")
        print("    Everything captured is lane=production_observation and can never")
        print("    be promoted. It is seed material for writing new scenarios.")

    if args.dry_run:
        print_result(True, "Dry run — nothing captured.")
        return EXIT_OK

    result = run_batch(
        scenarios,
        adapter=adapter,
        workspace=workspace,
        batch_id=args.out_batch,
        limit=args.limit,
    )

    if authorization is not None:
        (workspace.raw_batch(args.out_batch) / "_capture_authorization.json").write_text(
            json.dumps(authorization.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"\n  requests : {result.requests}")
    print(f"  captured : {result.captures}")
    print(f"  failures : {len(result.failures)}")
    for name, error in result.failures[:10]:
        print(f"    ✗ {name}: {error[:100]}")

    if not result.ok:
        print_result(False, f"{len(result.failures)} capture(s) failed.")
        return EXIT_ERROR

    print_result(
        True,
        f"{result.captures} capture(s) written to {workspace.raw_batch(args.out_batch)}",
        hint=f"Next: python scripts/normalize_captures.py --batch {args.out_batch}",
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
