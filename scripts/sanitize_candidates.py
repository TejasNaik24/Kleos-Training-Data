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
from kleos_training_data.errors import (
    EXIT_OK,
    EXIT_PRIVACY_VIOLATION,
    PrivacyViolationError,
)
from kleos_training_data.ids import example_id
from kleos_training_data.paths import Workspace
from kleos_training_data.privacy.entities import VAULT_SLOTS, EntityVault
from kleos_training_data.privacy.sanitize import STATUS_BLOCKED, sanitize
from kleos_training_data.scenarios.surrogates import load_pools
from kleos_training_data.staging.reasons import RejectionReason
from kleos_training_data.staging.records import (
    NormalizedCandidate,
    RejectionRecord,
    SanitizedCandidate,
)
from kleos_training_data.staging.store import iter_records, write_record


def _add_vault_entry(workspace: Workspace, slot: str) -> int:
    if slot not in VAULT_SLOTS:
        print(f"✗ Unknown slot {slot!r}. Valid: {', '.join(VAULT_SLOTS)}", file=sys.stderr)
        return 1

    vault = EntityVault.load(workspace.entity_vault)
    print(f"Enter the literal to register as {slot} (it will not be echoed to history).")
    print("One per line, blank line to finish:")
    added = 0
    for line in sys.stdin:
        literal = line.strip()
        if not literal:
            break
        vault.add(literal, slot)
        added += 1
    path = vault.save(workspace.entity_vault)
    print(f"\n✓ Registered {added} entr(ies). Vault now holds {len(vault)}.")
    print(f"  {path} (chmod 600, git-ignored, never in a release)")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_batch_argument(parser, required=False)
    parser.add_argument(
        "--add-vault-entry",
        metavar="SLOT",
        help=f"Register literals from stdin. Slots: {', '.join(VAULT_SLOTS)}",
    )
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    parser.add_argument("--report", type=Path, help="Write a machine-readable report here.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    workspace = Workspace.from_env(args.workspace)
    workspace.assert_initialized()

    if args.add_vault_entry:
        return _add_vault_entry(workspace, args.add_vault_entry)

    if not args.batch:
        parser.error("--batch is required unless using --add-vault-entry")

    print_header("SANITIZE")

    source = workspace.normalized_batch(args.batch)
    destination = workspace.sanitized_batch(args.batch)
    vault = EntityVault.load(workspace.entity_vault)
    pools = load_pools()

    print(f"\n  batch   : {args.batch}")
    print(f"  vault   : {len(vault)} entr(ies)")
    print(f"  pools   : {', '.join(sorted(pools))}")

    statuses: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    blocked: list[tuple[str, list[str]]] = []
    written = 0

    for candidate in iter_records(source, NormalizedCandidate):
        result = sanitize(
            candidate.payload,
            scenario_family=candidate.scenario_family,
            vault=vault,
            pools=pools,
        )
        statuses[result.status] += 1
        verdicts[result.fact_risk.verdict] += 1
        for detection in result.detections:
            kinds[detection.kind] += 1

        if result.status == STATUS_BLOCKED:
            reasons = sorted({d.rule_id for d in result.blocking})
            blocked.append((candidate.candidate_id, reasons))
            write_record(
                RejectionRecord(
                    candidate_id=candidate.candidate_id,
                    content_hash=candidate.content_hash,
                    stage="sanitization",
                    reason_codes=[RejectionReason.SECRET_DETECTED],
                    detail=f"blocking rules fired: {', '.join(reasons)}",
                    remediation=(
                        "A secret in a candidate means the capture path is "
                        "compromised. Rotate the credential, then investigate how "
                        "it reached a prompt — do not simply re-run."
                    ),
                    retryable=False,
                ),
                workspace.rejection(candidate.candidate_id),
            )
            continue

        new_id = example_id(result.payload)
        sanitized = SanitizedCandidate(
            candidate_id=new_id,
            superseded_candidate_id=(
                candidate.candidate_id if new_id != candidate.candidate_id else None
            ),
            batch_id=candidate.batch_id,
            lane=candidate.lane,
            scenario=candidate.scenario,
            sanitization_status=result.status,
            ruleset_version=result.ruleset_version,
            surrogate_map_id=result.surrogate_map_id,
            payload=result.payload,
            scenario_family=candidate.scenario_family,
            group_id=candidate.group_id,
            perturbation_of=candidate.perturbation_of,
            perturbation_kind=candidate.perturbation_kind,
            content_hash=result.output_hash,
        )
        write_record(sanitized, destination / f"{new_id}.json")
        (destination / f"{new_id}.privacy.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written += 1

    print(f"\n  sanitized : {written}")
    for status, count in sorted(statuses.items()):
        print(f"    {status:<16} {count}")

    print("\n  private-fact verdicts:")
    for verdict, count in sorted(verdicts.items()):
        print(f"    {verdict:<20} {count}")

    if kinds:
        print("\n  detections by kind:")
        for kind, count in sorted(kinds.items()):
            print(f"    {kind:<24} {count}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "batch": args.batch,
                    "statuses": dict(statuses),
                    "verdicts": dict(verdicts),
                    "kinds": dict(kinds),
                    "blocked": [{"candidate": c, "rules": r} for c, r in blocked],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  report written to {args.report}")

    if blocked:
        raise PrivacyViolationError(
            f"{len(blocked)} candidate(s) contain a secret.",
            details={"candidates": ", ".join(c for c, _ in blocked[:5])},
            suggestions=[
                "Rotate every credential involved before anything else.",
                "Then investigate how it reached a prompt. A secret in a candidate "
                "means the capture path is compromised, not that one example is bad.",
                "Rejection records are in staging/rejected/. See docs/incident-response.md.",
            ],
        )

    print_result(
        True,
        f"{written} candidate(s) sanitized.",
        hint=f"Next: python scripts/build_review_packet.py --batch {args.batch}",
    )
    return EXIT_OK if not blocked else EXIT_PRIVACY_VIOLATION


if __name__ == "__main__":
    raise SystemExit(run(main))
