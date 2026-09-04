#!/usr/bin/env python3
"""Turn raw captures into contract-shaped candidates.

    python scripts/normalize_captures.py --batch slice-001

Normalization is deterministic and non-semantic: line endings, reasoning spans,
trailing whitespace, transport envelopes. It never changes meaning. Every change
it does make is recorded on the candidate, so a reviewer sees what was done to
the text before they read it.
"""

from __future__ import annotations

import argparse
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
from kleos_training_data.collection.runner import to_request
from kleos_training_data.errors import EXIT_ERROR, EXIT_OK, ContractViolationError
from kleos_training_data.paths import Workspace
from kleos_training_data.scenarios.generator import generate
from kleos_training_data.scenarios.loader import DEFAULT_CATALOG_DIR, load_catalog
from kleos_training_data.scenarios.surrogates import SurrogatePool
from kleos_training_data.staging.normalize import normalize_capture
from kleos_training_data.staging.records import RawCapture
from kleos_training_data.staging.store import iter_records, write_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_batch_argument(parser)
    parser.add_argument("--scenarios", type=Path, help="Catalog root (default: scenarios/).")
    parser.add_argument("--workspace", type=Path, help="Workspace root.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    setup_logging(args)

    print_header("NORMALIZE")

    workspace = Workspace.from_env(args.workspace)
    workspace.assert_initialized()

    source = workspace.raw_batch(args.batch)
    destination = workspace.normalized_batch(args.batch)

    # Rebuild the request side from the catalog. The capture holds the answer;
    # the prompt that produced it is reproducible from the scenario, and
    # re-deriving it is what lets normalization verify the two still agree.
    # Keyed by (family, point), holding *every* request at that point rather
    # than one per perturbation kind. The key used to include the kind and the
    # base id but not the ordinal, so the two members of a `count: 2` paraphrase
    # group shared a key and the second replaced the first — leaving 204
    # captures in the batch whose prompt was no longer in the lookup, reported
    # as "no prompt in the catalog matches this capture".
    requests_by_point: dict[str, list[object]] = {}
    for scenario in load_catalog(args.scenarios or DEFAULT_CATALOG_DIR):
        pool = SurrogatePool.load(scenario.entities.pool)
        for candidate in generate(scenario, pool):
            key = f"{scenario.family}|{candidate.situation.point_index}"
            requests_by_point.setdefault(key, []).append(to_request(scenario, candidate))

    print(f"\n  batch  : {args.batch}")
    print(f"  source : {source}")

    written = 0
    seen_ids: dict[str, str] = {}
    failures: list[tuple[str, str]] = []
    transformations: Counter[str] = Counter()

    for capture in iter_records(source, RawCapture):
        key = f"{capture.scenario.family}|{capture.scenario.point_index}"
        matches = requests_by_point.get(key, [])
        if not matches:
            failures.append((capture.capture_id, "no scenario point matches this capture"))
            continue

        # A capture identifies its point; the perturbation dimension is resolved
        # by matching the request hash, so a reordered catalog cannot silently
        # pair a capture with the wrong prompt.
        from kleos_training_data.hashing import canonical_hash

        request = None
        for option in matches:
            if (
                canonical_hash({"system": option.system_prompt, "user": option.user_message})
                == capture.request_hash
            ):
                request = option
                break

        if request is None:
            failures.append((capture.capture_id, "no prompt in the catalog matches this capture"))
            continue

        try:
            candidate = normalize_capture(
                capture,
                system_prompt=request.system_prompt,
                user_message=request.user_message,
                variation_axes=request.variation_axes,
                group_id=request.group_id,
                perturbation_of=request.perturbation_of,
                perturbation_kind=request.perturbation_kind,
            )
        except ContractViolationError as exc:
            failures.append((capture.capture_id, exc.message))
            continue

        target = destination / f"{candidate.candidate_id}.json"
        if candidate.candidate_id in seen_ids:
            # Two captures normalized to identical content. Left alone this is a
            # silent loss: the second write overwrites the first and the batch
            # quietly shrinks. Report it instead — it means two scenario points
            # render the same text, which is a catalog problem, not a transport
            # one.
            failures.append(
                (
                    capture.capture_id,
                    f"normalizes to {candidate.candidate_id}, already produced by "
                    f"capture {seen_ids[candidate.candidate_id]}",
                )
            )
            continue
        seen_ids[candidate.candidate_id] = capture.capture_id

        transformations.update(candidate.transformations)
        write_record(candidate, target)
        written += 1

    print(f"  written: {written} candidate(s) to {destination}")
    if transformations:
        print("\n  transformations applied:")
        for name, count in sorted(transformations.items()):
            print(f"    {name:<28} {count}")

    if failures:
        print(f"\n  failures: {len(failures)}")
        for capture_id, error in failures[:10]:
            print(f"    ✗ {capture_id}: {error}")
        print_result(False, f"{len(failures)} capture(s) could not be normalized.")
        return EXIT_ERROR

    print_result(
        True,
        f"{written} candidate(s) normalized.",
        hint=f"Next: python scripts/sanitize_candidates.py --batch {args.batch}",
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(run(main))
