#!/usr/bin/env python3
"""Verify a sealed release by re-deriving everything from its bytes.

    python scripts/verify_release.py --release releases/kleos-policy-v0.1.0 --strict

Every count, distribution and hash is recomputed from the JSONL and compared
against the manifest and the lock. This shares no computation with the writer:
reusing it would only prove the writer is self-consistent, which is not the
question.

Exits 3 on any mismatch, 4 if the mismatch is a privacy one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.datasets.verify import verify_release
from kleos_training_data.errors import EXIT_GATE_FAILED, EXIT_OK, EXIT_PRIVACY_VIOLATION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release", type=Path, required=True, help="Sealed release directory.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    parser.add_argument("--json", type=Path, help="Write a machine-readable report here.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("VERIFY RELEASE")

    report = verify_release(args.release, strict=args.strict)
    print()
    print(report.render())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\n  report written to {args.json}")

    if not report.ok:
        privacy = any("privacy" in p or "contains_private_data" in p for p in report.problems)
        print_result(
            False,
            f"{len(report.problems)} problem(s) found across {report.checks_run} checks.",
            hint="A release that does not verify must not be used for training.",
        )
        return EXIT_PRIVACY_VIOLATION if privacy else EXIT_GATE_FAILED

    print_result(
        True,
        f"{report.checks_run} checks passed.",
        hint=(f"Next: python scripts/check_contract_compat.py --release {args.release} --strict"),
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
