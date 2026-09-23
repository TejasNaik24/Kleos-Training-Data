from __future__ import annotations

import argparse
import sys
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
from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.errors import EXIT_ERROR, EXIT_OK
from kleos_training_data.paths import Workspace
from kleos_training_data.privacy.detect import scan_payload
from kleos_training_data.privacy.facts import assess
from kleos_training_data.review.packets import PacketItem, ReviewPacket, nearest_neighbours
from kleos_training_data.scenarios.loader import DEFAULT_CATALOG_DIR, load_catalog
from kleos_training_data.staging.records import PromotedExample, SanitizedCandidate
from kleos_training_data.staging.store import iter_records


def _promoted_corpus(workspace: Workspace) -> dict[str, str]:
    corpus: dict[str, str] = {}
    for record in iter_records(workspace.staging / "promoted", PromotedExample):
        example = record.example
        corpus[example["id"]] = "\n".join(
            f"{m['role']}: {m['content']}" for m in example.get("messages", [])
        )
    return corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_batch_argument(parser)
    parser.add_argument("--packet-id", help="Packet identifier (default: pk-<batch>).")
    parser.add_argument("--max-candidates", type=int, help="Cap the packet size.")
    parser.add_argument(
        "--include-neighbors",
        type=int,
        default=3,
        help="Nearest promoted examples to show per candidate (default: 3).",
    )
    parser.add_argument("--scenarios", type=Path, help="Catalog root.")
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("REVIEW PACKET")

    workspace = Workspace.from_env(args.workspace)
    workspace.assert_initialized()

    packet_id = args.packet_id or f"pk-{args.batch}"
    claims = {
        s.family: (" ".join(s.policy_claim.split()), " ".join(s.anti_claim.split()))
        for s in load_catalog(args.scenarios or DEFAULT_CATALOG_DIR)
    }
    corpus = _promoted_corpus(workspace)

    items: list[PacketItem] = []
    for candidate in iter_records(workspace.sanitized_batch(args.batch), SanitizedCandidate):
        if args.max_candidates and len(items) >= args.max_candidates:
            break

        payload = candidate.payload
        contract_valid = True
        try:
            TrainingExample.model_validate({**payload, "id": candidate.candidate_id})
        except Exception:
            contract_valid = False

        text = "\n".join(f"{m['role']}: {m['content']}" for m in payload.get("messages", []))
        policy_claim, anti_claim = claims.get(
            candidate.scenario_family, ("(scenario not found)", "(scenario not found)")
        )

        items.append(
            PacketItem(
                candidate_id=candidate.candidate_id,
                content_hash=candidate.content_hash,
                task=payload.get("task", ""),
                scenario_family=candidate.scenario_family,
                policy_claim=policy_claim,
                anti_claim=anti_claim,
                payload=payload,
                variation_axes=payload.get("variation_axes", {}),
                detections=scan_payload(payload),
                fact_risk=assess(payload),
                neighbours=nearest_neighbours(text, corpus, limit=args.include_neighbors),
                perturbation_kind=candidate.perturbation_kind,
                contract_valid=contract_valid,
            )
        )

    if not items:
        print(f"\n  No sanitized candidates found for batch {args.batch}.")
        print_result(False, "Nothing to review.")
        return EXIT_ERROR

    packet = ReviewPacket(packet_id=packet_id, batch_id=args.batch, items=items)
    destination = packet.write(workspace.review_packet(packet_id))

    flagged = [i for i in items if i.fact_risk.requires_human or i.detections]
    invalid = [i for i in items if not i.contract_valid]

    print(f"\n  packet     : {packet_id}")
    print(f"  candidates : {len(items)}")
    print(f"  flagged    : {len(flagged)} needing a closer look")
    print(f"  invalid    : {len(invalid)} failing the contract")
    print(f"  written    : {destination}")

    print_result(
        True,
        f"Packet ready with {len(items)} candidate(s).",
        hint=f"Next: python scripts/run_llm_review.py --packet-id {packet_id} --reviewer mock",
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
