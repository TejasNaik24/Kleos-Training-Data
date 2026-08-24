#!/usr/bin/env python3
"""Record a human decision about a candidate.

    python scripts/record_decision.py --candidate kx-npr-abc --decision approve \\
        --gate no_private_data=PASS --gate policy_not_facts=PASS \\
        --gate no_unsupported_claims=PASS --gate schema_and_contract_valid=PASS

The decision is signed against the exact content hash it was made about. Edit the
candidate afterwards and the signature stops matching, so an approval never
carries over to text nobody read.

You may override a score with a justification. You may **not** approve over a
failing `no_private_data` or `policy_not_facts` gate — the record cannot be
constructed. Fix the content instead, which changes its id and requires a fresh
review.

``--adopt-machine-gates`` starts from the machine reviewer's gates rather than
typing all four. It is a convenience for the common case, not a way to skip
reading: the gates it adopts are printed before the decision is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import ValidationError

from _cli import add_common_arguments, print_header, print_result, run, setup_logging
from kleos_training_data.errors import EXIT_ERROR, EXIT_OK, ReviewError
from kleos_training_data.paths import Workspace
from kleos_training_data.review.records import HumanDecision, MachineReviewRecord
from kleos_training_data.review.rubric import DIMENSIONS, HARD_GATES, GateResults
from kleos_training_data.staging.reasons import RejectionReason
from kleos_training_data.staging.records import SanitizedCandidate
from kleos_training_data.staging.store import iter_records, read_record, write_record


def _parse_pairs(values: list[str], *, what: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ReviewError(f"{what} must be NAME=VALUE, got {item!r}")
        name, _, value = item.partition("=")
        parsed[name.strip()] = value.strip()
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidate", required=True, help="Candidate id.")
    parser.add_argument("--decision", required=True, choices=["approve", "reject", "revise"])
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        metavar="NAME=PASS|FAIL",
        help=f"Hard gate result. Gates: {', '.join(HARD_GATES)}",
    )
    parser.add_argument(
        "--adopt-machine-gates",
        action="store_true",
        help="Start from the machine reviewer's gates (printed before use).",
    )
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        metavar="DIM=N",
        help=f"Override a score 0-4. Dimensions: {', '.join(DIMENSIONS)}",
    )
    parser.add_argument("--justification", help="Required when overriding a score.")
    parser.add_argument(
        "--reason",
        action="append",
        default=[],
        metavar="CODE",
        help="Rejection reason code. Required when rejecting.",
    )
    parser.add_argument("--notes", help="Free-text notes for the record.")
    parser.add_argument("--reviewer-role", default="operator", help="A role, never a name.")
    parser.add_argument("--batch", help="Batch to resolve the candidate in.")
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("HUMAN DECISION")

    workspace = Workspace.from_env(args.workspace)
    workspace.assert_initialized()

    # Find the sanitized candidate so the decision binds to its exact hash.
    candidate = None
    search_roots = (
        [workspace.sanitized_batch(args.batch)]
        if args.batch
        else sorted((workspace.staging / "sanitized").glob("*"))
    )
    for root in search_roots:
        for record in iter_records(root, SanitizedCandidate):
            if record.candidate_id == args.candidate:
                candidate = record
                break
        if candidate:
            break

    if candidate is None:
        print(f"\n  No sanitized candidate {args.candidate!r} found.")
        print_result(False, "Nothing to decide about.")
        return EXIT_ERROR

    gates_raw: dict[str, str] = {}
    machine_path = workspace.llm_review(args.candidate)
    if args.adopt_machine_gates:
        if not machine_path.is_file():
            raise ReviewError(
                f"No machine review for {args.candidate}.",
                suggestions=["Run scripts/run_llm_review.py first, or state the gates."],
            )
        machine = read_record(machine_path, MachineReviewRecord)
        gates_raw = machine.review.gates.model_dump()
        print("\n  adopting machine gates:")
        for name, value in sorted(gates_raw.items()):
            print(f"    {name:<28} {value}")
        if machine.review.gate_evidence:
            print("\n  machine evidence:")
            for name, evidence in sorted(machine.review.gate_evidence.items()):
                print(f"    {name}: {evidence}")

    gates_raw.update(_parse_pairs(args.gate, what="--gate"))

    missing = sorted(set(HARD_GATES) - set(gates_raw))
    if missing:
        raise ReviewError(
            f"Every hard gate needs a result; missing {missing}.",
            suggestions=[
                "State them with --gate NAME=PASS|FAIL.",
                "Or start from the machine review with --adopt-machine-gates.",
            ],
        )

    scores = {name: int(value) for name, value in _parse_pairs(args.score, what="--score").items()}
    reasons = [RejectionReason(code) for code in args.reason]

    # The model invariants are the point of this script, so their messages have
    # to reach the operator as guidance rather than as "this looks like a bug".
    try:
        decision = HumanDecision(
            candidate_id=args.candidate,
            content_hash=candidate.content_hash,
            reviewer_role=args.reviewer_role,
            decision=args.decision,
            gates=GateResults.model_validate(gates_raw),
            score_overrides=scores,
            override_justification=args.justification,
            reason_codes=reasons,
            notes=args.notes,
        ).signed()
    except ValidationError as exc:
        raise ReviewError(
            "That decision is not permitted.",
            details={
                "reason": "; ".join(
                    str(error.get("ctx", {}).get("error") or error["msg"]) for error in exc.errors()
                )
            },
            suggestions=[
                "A failing no_private_data or policy_not_facts gate cannot be "
                "approved over. Fix the content instead — that changes its "
                "content_hash and its id, and requires a fresh review of the text "
                "that actually results.",
                "To reject instead: --decision reject --reason <CODE>",
            ],
        ) from exc

    write_record(decision, workspace.human_decision(args.candidate))

    print(f"\n  candidate : {args.candidate}")
    print(f"  decision  : {decision.decision}")
    print(f"  bound to  : {decision.content_hash[:16]}…")
    print(f"  signature : {(decision.signature or '')[:16]}…")

    print_result(
        True,
        f"Decision recorded for {args.candidate}.",
        hint="Next: python scripts/promote_examples.py --batch <id>",
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
