from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.errors import EXIT_ERROR, EXIT_OK, ReviewError
from kleos_training_data.paths import Workspace
from kleos_training_data.review.records import MachineReviewRecord
from kleos_training_data.review.reviewers import resolve_reviewer
from kleos_training_data.staging.store import write_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-id", required=True, help="Packet to review.")
    parser.add_argument("--reviewer", default="mock", help="Reviewer backend (default: mock).")
    parser.add_argument("--model", help="Model name, for a real reviewer.")
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    parser.add_argument("--dry-run", action="store_true", help="Report the plan and stop.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("MACHINE REVIEW")

    workspace = Workspace.from_env(args.workspace)
    workspace.assert_initialized()

    if args.reviewer != "mock" and os.environ.get("CI"):
        raise ReviewError(
            f"Reviewer {args.reviewer!r} needs a network call, and CI is set.",
            suggestions=[
                "CI uses --reviewer mock so the review stage is testable without "
                "a key and without a bill.",
            ],
        )

    packet_dir = workspace.review_packet(args.packet_id)
    packet_file = packet_dir / "packet.jsonl"
    if not packet_file.is_file():
        print(f"\n  No packet at {packet_file}.")
        print_result(False, f"Packet {args.packet_id} not found.")
        return EXIT_ERROR

    reviewer = resolve_reviewer(args.reviewer)
    items = [
        json.loads(line)
        for line in packet_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"\n  packet   : {args.packet_id}")
    print(f"  reviewer : {reviewer.name}")
    print(f"  items    : {len(items)}")

    if args.dry_run:
        print_result(True, "Dry run — nothing reviewed.")
        return EXIT_OK

    gate_failures: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    written = 0

    for item in items:
        review = reviewer.review(item)
        verdict = review.verdict(min_mean_score=0.0)
        decisions[verdict.decision] += 1
        for gate in review.gates.failed_ids():
            gate_failures[gate] += 1

        from kleos_training_data.ids import content_hash

        write_record(
            MachineReviewRecord(
                candidate_id=item["candidate_id"],
                content_hash=content_hash(item["payload"]),
                packet_id=args.packet_id,
                reviewer_backend=reviewer.name,
                reviewer_model=args.model,
                review=review,
            ),
            workspace.llm_review(item["candidate_id"]),
        )
        written += 1

    print(f"\n  reviewed : {written}")
    print("\n  machine verdicts:")
    for decision, count in sorted(decisions.items()):
        print(f"    {decision:<16} {count}")

    if gate_failures:
        print("\n  gate failures:")
        for gate, count in sorted(gate_failures.items()):
            print(f"    {gate:<28} {count}")
    else:
        print("\n  no gate failures")

    print_result(
        True,
        f"{written} machine review(s) written.",
        hint=(
            "A machine review approves nothing. Next: read the packet, then "
            "python scripts/record_decision.py --candidate <id> --decision approve"
        ),
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
