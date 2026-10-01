from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.datasets.reasoning_checks import check_reasoning_release
from kleos_training_data.errors import EXIT_GATE_FAILED, EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True, help="Sealed release directory.")
    parser.add_argument(
        "--reference-test",
        type=Path,
        required=True,
        help="The test.jsonl this release must reproduce byte for byte.",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("CHECK REASONING RELEASE")
    report = check_reasoning_release(args.release, args.reference_test)
    print()
    print(report.render())

    if not report.ok:
        print_result(False, f"{len(report.problems)} problem(s) found.")
        return EXIT_GATE_FAILED
    print_result(True, "Reasoning release checks passed.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
