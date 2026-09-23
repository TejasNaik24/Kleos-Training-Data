from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, run, setup_logging
from kleos_training_data.contract.compat import ENV_MODELS_PATH, run_handshake
from kleos_training_data.contract.pin import (
    CONTRACT_SOURCE_COMMIT,
    DELIBERATE_DELTAS,
    KNOWN_UPSTREAM_ISSUES,
)
from kleos_training_data.errors import EXIT_GATE_FAILED, EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, help="Also round-trip this release.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat an absent kleos-models as a failure. CI uses this.",
    )
    parser.add_argument("--json", type=Path, help="Write a machine-readable report here.")
    parser.add_argument(
        "--show-deltas",
        action="store_true",
        help="Print the deliberate strictness deltas and known upstream issues.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("CONTRACT COMPATIBILITY")

    report = run_handshake(args.release)
    print()
    print(report.render())

    if args.show_deltas:
        print("\n  Deliberate strictness deltas (we are never looser):")
        for name, why in sorted(DELIBERATE_DELTAS.items()):
            print(f"    · {name}: {why}")
        print("\n  Known upstream issues, not reproduced:")
        for name, why in sorted(KNOWN_UPSTREAM_ISSUES.items()):
            print(f"    · {name}: {why}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\n  report written to {args.json}")

    print()
    if not report.available:
        if args.strict:
            print("✗ INCOMPATIBLE — verification could not run.", file=sys.stderr)
            print(
                "\n  kleos-models is not installed, so nothing was verified. Under\n"
                "  --strict that is a failure: a check that passes because it could\n"
                "  not run produces a green build that means nothing.\n\n"
                '  Install it: pip install -e ".[dev,compat]"\n'
                f"  Or point {ENV_MODELS_PATH} at a local checkout.\n",
                file=sys.stderr,
            )
            return EXIT_GATE_FAILED
        print("· SKIPPED — kleos-models is not installed.")
        print("  The offline pipeline does not need it. CI verifies with --strict.\n")
        return EXIT_OK

    if report.failures:
        print("✗ INCOMPATIBLE", file=sys.stderr)
        print(
            f"\n  {len(report.failures)} check(s) disagree with the pinned commit\n"
            f"  {CONTRACT_SOURCE_COMMIT[:12]}. A release built now may not load in\n"
            "  kleos-models.\n\n"
            "  Read the public repo's diff, update the mirror, re-run every\n"
            "  differential suite, then move the pin in contract/pin.py.\n"
            "  Never move the pin to make a failing check pass.\n",
            file=sys.stderr,
        )
        return EXIT_GATE_FAILED

    print(f"✓ COMPATIBLE — {len(report.checks)} check(s) agree with {CONTRACT_SOURCE_COMMIT[:12]}.")
    if args.release:
        print(f"  {args.release} loads and validates in kleos-models.")
    print()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
