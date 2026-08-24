#!/usr/bin/env python3
"""Promote reviewed candidates into the pool a release draws from.

    python scripts/promote_examples.py --batch slice-001

Fourteen gates, all of which run — one pass tells you everything wrong with a
candidate rather than one thing at a time.

``--force`` bypasses only the gates the gate table marks bypassable. It cannot
reach a privacy, schema, review or leakage gate: ``PromotionPolicy`` refuses to
be constructed with one, and the runner refuses to believe a run in which a
mandatory gate did not execute.

Exits 3 when a gate fails, 4 when the failure is a privacy gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import (
    add_batch_argument,
    add_common_arguments,
    print_header,
    print_result,
    run,
    setup_logging,
)
from kleos_training_data.contract.dedup import CorpusEntry, CorpusIndex, conversation_text
from kleos_training_data.errors import EXIT_GATE_FAILED, EXIT_OK, EXIT_PRIVACY_VIOLATION
from kleos_training_data.ids import CollisionLedger
from kleos_training_data.paths import Workspace
from kleos_training_data.privacy.entities import EntityVault
from kleos_training_data.promotion.context import PromotionContext
from kleos_training_data.promotion.gates import GATES
from kleos_training_data.promotion.policy import PromotionPolicy
from kleos_training_data.promotion.runner import run_gates
from kleos_training_data.review.records import HumanDecision, MachineReviewRecord
from kleos_training_data.staging.reasons import RejectionReason
from kleos_training_data.staging.records import (
    PromotedExample,
    RejectionRecord,
    SanitizedCandidate,
)
from kleos_training_data.staging.store import append_index, iter_records, read_record, write_record

LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "id_ledger.json"


def _load_corpus(workspace: Workspace) -> CorpusIndex:
    """Everything already promoted."""
    entries = []
    for record in iter_records(workspace.staging / "promoted", PromotedExample):
        entries.append(
            CorpusEntry(
                example_id=record.example["id"],
                text=conversation_text(record.example),
                scenario_family=(record.example.get("metadata") or {}).get("scenario_family"),
            )
        )
    return CorpusIndex(entries=entries)


def _load_eval_corpus(path: Path | None) -> CorpusIndex:
    """Held-out evaluation material to check leakage against."""
    if path is None or not path.is_file():
        return CorpusIndex(entries=[])
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        entries.append(
            CorpusEntry(
                example_id=payload.get("id", "?"),
                text=conversation_text(payload),
                scenario_family=(payload.get("metadata") or {}).get("scenario_family"),
            )
        )
    return CorpusIndex(entries=entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_batch_argument(parser)
    parser.add_argument("--candidate", action="append", default=None, help="Limit to these ids.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the gates the table marks bypassable. Cannot reach a privacy gate.",
    )
    parser.add_argument(
        "--eval-fixtures",
        type=Path,
        help="JSONL of held-out evaluation examples to check leakage against.",
    )
    parser.add_argument("--min-mean-score", type=float, default=3.0)
    parser.add_argument("--near-dup-threshold", type=float, default=0.85)
    parser.add_argument("--strict-warnings", action="store_true", help="Treat WARN as failure.")
    parser.add_argument("--report", type=Path, help="Write the full gate report here.")
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    parser.add_argument("--dry-run", action="store_true", help="Run gates, write nothing.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("PROMOTION")

    workspace = Workspace.from_env(args.workspace)
    workspace.assert_initialized()

    # --force maps to a constant, never to user input. There is deliberately no
    # flag naming a gate to bypass.
    policy = (
        PromotionPolicy.forced(
            min_mean_score=args.min_mean_score,
            near_duplicate_threshold=args.near_dup_threshold,
            strict_warnings=args.strict_warnings,
        )
        if args.force
        else PromotionPolicy(
            min_mean_score=args.min_mean_score,
            near_duplicate_threshold=args.near_dup_threshold,
            strict_warnings=args.strict_warnings,
        )
    )

    ledger = CollisionLedger.load(LEDGER_PATH)
    vault = EntityVault.load(workspace.entity_vault)
    corpus = _load_corpus(workspace)
    eval_corpus = _load_eval_corpus(args.eval_fixtures)

    print(f"\n  batch      : {args.batch}")
    print(f"  gates      : {len(GATES)} ({len(policy.bypass_gate_ids)} bypassed)")
    if policy.bypass_gate_ids:
        print(f"  bypassing  : {', '.join(sorted(policy.bypass_gate_ids))}")
    print(f"  corpus     : {len(corpus)} already promoted")
    print(f"  eval fixtures: {len(eval_corpus)}")

    wanted = set(args.candidate) if args.candidate else None
    promoted = 0
    rejected = 0
    gate_failures: Counter[str] = Counter()
    privacy_failures = 0
    reports = []

    for candidate in iter_records(workspace.sanitized_batch(args.batch), SanitizedCandidate):
        if wanted and candidate.candidate_id not in wanted:
            continue

        privacy_path = (
            workspace.sanitized_batch(args.batch) / f"{candidate.candidate_id}.privacy.json"
        )
        privacy = (
            json.loads(privacy_path.read_text(encoding="utf-8")) if privacy_path.is_file() else None
        )

        human_path = workspace.human_decision(candidate.candidate_id)
        human = read_record(human_path, HumanDecision) if human_path.is_file() else None
        machine_path = workspace.llm_review(candidate.candidate_id)
        machine = read_record(machine_path, MachineReviewRecord) if machine_path.is_file() else None

        ctx = PromotionContext(
            candidate=candidate,
            privacy=privacy,
            human=human,
            machine=machine,
            ledger=ledger,
            corpus=corpus,
            eval_corpus=eval_corpus,
            vault=vault,
            min_mean_score=policy.min_mean_score,
            near_duplicate_threshold=policy.near_duplicate_threshold,
            allowed_sources=frozenset({candidate.lane.contract_source}),
        )

        report = run_gates(ctx, policy)
        reports.append(report.to_dict())

        for failure in report.failures:
            gate_failures[failure.gate_id] += 1
        if report.privacy_failures:
            privacy_failures += 1

        if not report.ok(strict_warnings=policy.strict_warnings):
            rejected += 1
            if args.verbose:
                print()
                print(report.render())
            if not args.dry_run:
                write_record(
                    RejectionRecord(
                        candidate_id=candidate.candidate_id,
                        content_hash=candidate.content_hash,
                        stage="promotion",
                        reason_codes=_reasons_for(report),
                        detail="; ".join(f"{r.gate_id}: {r.message}" for r in report.failures),
                        retryable=not report.privacy_failures,
                    ),
                    workspace.rejection(candidate.candidate_id),
                )
            continue

        if args.dry_run:
            promoted += 1
            continue

        example = ctx.as_example()
        write_record(
            PromotedExample(
                example=example,
                audit={
                    "content_hash": candidate.content_hash,
                    "gate_results": [r.to_dict() for r in report.results],
                    "human_decision_signature": human.signature if human else None,
                    "privacy_ruleset_version": candidate.ruleset_version,
                    "scenario_catalog_version": candidate.scenario.catalog_version,
                    "scenario_fingerprint": candidate.scenario.scenario_fingerprint,
                    "lane": candidate.lane.value,
                    "batch_id": candidate.batch_id,
                },
            ),
            workspace.promoted(candidate.candidate_id),
        )
        append_index(
            workspace.promoted_index,
            {
                "example_id": example["id"],
                "task": example["task"],
                "scenario_family": candidate.scenario_family,
                "group_id": candidate.group_id,
                "batch_id": candidate.batch_id,
            },
        )
        # Index as we go, so two identical candidates in one batch cannot both
        # be promoted by racing an index that is only refreshed between runs.
        corpus.add(
            CorpusEntry(
                example_id=example["id"],
                text=conversation_text(example),
                scenario_family=candidate.scenario_family,
            )
        )
        promoted += 1

    if not args.dry_run:
        ledger.save(LEDGER_PATH)

    print(f"\n  promoted : {promoted}")
    print(f"  rejected : {rejected}")
    if gate_failures:
        print("\n  gate failures:")
        for gate_id, count in sorted(gate_failures.items()):
            print(f"    {gate_id:<26} {count}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"batch": args.batch, "reports": reports}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n  report written to {args.report}")

    if rejected:
        print_result(
            False,
            f"{rejected} candidate(s) did not pass every gate.",
            hint="Rejection records are in staging/rejected/. Re-run with -v for detail.",
        )
        return EXIT_PRIVACY_VIOLATION if privacy_failures else EXIT_GATE_FAILED

    print_result(
        True,
        f"{promoted} example(s) promoted.",
        hint="Next: python scripts/build_release.py --version kleos-policy-v0.1.0",
    )
    return EXIT_OK


def _reasons_for(report) -> list[RejectionReason]:
    """Map failing gate ids to closed-vocabulary reason codes."""
    mapping = {
        "G01_STAGING_INTEGRITY": RejectionReason.STAGING_INTEGRITY,
        "G02_SCHEMA_VALID": RejectionReason.SCHEMA_INVALID,
        "G03_ID_INTEGRITY": RejectionReason.ID_MISMATCH,
        "G04_SECRET_SCAN": RejectionReason.SECRET_DETECTED,
        "G05_PII_SCAN": RejectionReason.PII_UNRESOLVED,
        "G06_SURROGATE_INTEGRITY": RejectionReason.SURROGATE_RESIDUE,
        "G07_PRIVATE_FACT": RejectionReason.PRIVATE_FACT,
        "G08_REVIEW_PRESENT": RejectionReason.REVIEW_MISSING,
        "G09_REVIEW_APPROVED": RejectionReason.REVIEW_GATE_FAILED,
        "G10_PROVENANCE": RejectionReason.LANE_NOT_PROMOTABLE,
        "G11_COVERAGE_AXES": RejectionReason.AXES_INCOMPLETE,
        "G12_CORPUS_DEDUP": RejectionReason.CORPUS_DUPLICATE,
        "G13_EVAL_LEAKAGE": RejectionReason.EVAL_LEAKAGE,
        "G14_CONTRACT_RENDER": RejectionReason.RENDER_ROUNDTRIP_MISMATCH,
    }
    return [mapping[r.gate_id] for r in report.failures if r.gate_id in mapping] or [
        RejectionReason.OPERATOR_REJECTED
    ]


if __name__ == "__main__":
    raise SystemExit(run(main))
