# Staging and promotion

Every candidate moves through `staging/` as a typed, hash-verified record, and
reaches the promoted pool only after passing 14 promotion gates. This document
describes the record layout, how records are protected, what normalization
does, what each promotion gate checks, and how rejections are recorded.

## Contents

- [Records and layout](#records-and-layout)
- [Record integrity](#record-integrity)
- [Normalization](#normalization)
- [Promotion gates](#promotion-gates)
- [Deduplication (G12)](#deduplication-g12)
- [Evaluation leakage (G13)](#evaluation-leakage-g13)
- [Bypass protection](#bypass-protection)
- [Rejection codes](#rejection-codes)

## Records and layout

`staging/` is git-ignored. `python scripts/init_workspace.py` creates it along
with the other runtime zones.

| Path | Contents |
| --- | --- |
| `staging/raw/<batch>/` | Raw capture records, `_batch.json`, and `_capture_authorization.json` for authorized non-local captures |
| `staging/normalized/<batch>/` | Normalized candidates |
| `staging/sanitized/<batch>/` | Sanitized candidates and a `<id>.privacy.json` sidecar for each |
| `staging/review/packets/<packet-id>/` | Review packets: `packet.md`, `packet.jsonl` and `packet.meta.json` |
| `staging/review/llm/` | Machine review records |
| `staging/review/decisions/` | Decision records |
| `staging/rejected/` | One rejection record per rejected candidate |
| `staging/promoted/` | Promoted examples and an append-only `index.jsonl` |

Record types: `scenario_ref`, `raw_capture`, `normalized_candidate`,
`sanitized_candidate`, `llm_review`, `human_decision`, `rejection` and
`promoted_example`.

## Record integrity

| Property | Mechanism |
| --- | --- |
| Strict schemas | Every record model rejects unknown fields. |
| Tamper evidence | Each record stores a `record_hash`: SHA-256 over its canonical JSON (sorted keys, compact separators, Unicode NFC) without the hash field. `read_record` recomputes it and raises `StagingIntegrityError` on a mismatch, a missing file, invalid JSON or a schema error. |
| Atomic writes | `write_record` writes to a temporary file in the destination directory, calls `fsync`, and moves the file into place with `os.replace`. An interrupted write leaves the previous record or the new one, never a partial file. |
| File conventions | Names that start with `_` hold batch metadata, and names with a second dot (`<id>.privacy.json`) are sidecars. Record readers skip both. |

Hash verification on read means a candidate edited by hand after review stops
promotion with an integrity error instead of promoting content no reviewer saw.

## Normalization

`normalize_captures.py` matches each raw capture to its catalog request and
converts it into a contract-shaped candidate with system, user and assistant
messages. Normalization changes presentation only, and each change is appended
to the candidate's `transformations` list so a reviewer can see what was done:

| Transformation | Applied when |
| --- | --- |
| `normalized_line_endings` | The text contains CR or CRLF line endings |
| `stripped_reasoning_span` | The answer contains a `<think>` reasoning span |
| `trimmed_whitespace` | Trailing whitespace on a line or around the text was removed |

`stripped_reasoning_span` removes a `<think>` span that a backend left inside the
captured answer text. It never touches the separate `reasoning` field. That field
carries policy-derived reasoning from the catalog request, and normalization
attaches it only to mock-backend captures, whose answer is the policy's own.
Normalization only converts its line endings and trims trailing whitespace. A
`reasoning` value containing a `<think>` tag raises `ContractViolationError`.

An empty answer raises `ContractViolationError`. If two captures normalize to
identical content, the script reports a failure instead of letting one overwrite
the other, since it means two catalog points render the same text.

## Promotion gates

`promote_examples.py` runs every gate on every candidate, with no
short-circuiting, so a single run reports everything wrong with a candidate. A
gate that raises an exception is recorded as a failure rather than skipped.
Privacy gates re-scan the exact bytes being promoted instead of trusting the
earlier sanitization record.

| Gate | Checks | Mandatory | Rejection code |
| --- | --- | --- | --- |
| G01 `STAGING_INTEGRITY` | The record hash matches and the privacy sidecar exists | Yes | `STAGING_INTEGRITY` |
| G02 `SCHEMA_VALID` | The example satisfies the dataset contract | Yes | `SCHEMA_INVALID` |
| G03 `ID_INTEGRITY` | The ID re-derives from the content | Yes | `ID_MISMATCH` |
| G04 `SECRET_SCAN` | No secret-level detection remains (privacy) | Yes | `SECRET_DETECTED` |
| G05 `PII_SCAN` | No PII or review-level detection remains (privacy) | Yes | `PII_UNRESOLVED` |
| G06 `SURROGATE_INTEGRITY` | No vault literal or `[[PLACEHOLDER]]` residue remains (privacy) | Yes | `SURROGATE_RESIDUE` |
| G07 `PRIVATE_FACT` | The example teaches a policy, not a fact about a person (privacy) | Yes | `PRIVATE_FACT` |
| G08 `REVIEW_PRESENT` | A valid decision record covers this exact content | Yes | `REVIEW_MISSING` |
| G09 `REVIEW_APPROVED` | The decision is `approve` and the combined review verdict approves | Yes | `REVIEW_GATE_FAILED` |
| G10 `PROVENANCE` | The capture lane and source permit promotion | Yes | `LANE_NOT_PROMOTABLE` |
| G11 `COVERAGE_AXES` | `variation_axes.domain` is present. Unregistered axes produce a warning. | No | `AXES_INCOMPLETE` |
| G12 `CORPUS_DEDUP` | Not a duplicate of an already promoted example | Yes | `CORPUS_DUPLICATE` |
| G13 `EVAL_LEAKAGE` | No overlap with evaluation fixtures | Yes | `EVAL_LEAKAGE` |
| G14 `CONTRACT_RENDER` | Metadata extras are allowlisted and the example round-trips to identical bytes | Yes | `RENDER_ROUNDTRIP_MISMATCH` |

G07 passes an example the fact assessment classifies as policy-like and fails
one classified as fact-teaching. An example in between needs a decision record
whose `policy_not_facts` review hard gate is `PASS`.

`promote_examples.py` options that change gate behavior:

| Option | Effect |
| --- | --- |
| `--eval-fixtures <path>` | JSONL of evaluation examples for G13 |
| `--min-mean-score <n>` | Minimum mean review score, default 3.0 |
| `--near-dup-threshold <n>` | Jaccard threshold for near duplicates, default 0.85 |
| `--strict-warnings` | Treat any gate warning as a rejection |
| `--force` | Bypass G11 only |
| `--dry-run` | Evaluate without writing records |

The script exits with code 0 when nothing is rejected, 4 when a privacy gate
rejected a candidate, and 3 for any other rejection.

## Deduplication (G12)

Each candidate is compared with the promoted corpus:

| Finding | Test | Outcome |
| --- | --- | --- |
| ID collision | Same ID, different content | Fail |
| Exact duplicate | Same SHA-256 of the conversation text | Fail |
| Normalized duplicate | Same text after Unicode NFKD, removing combining marks, case folding, replacing digit runs with `0`, replacing punctuation with spaces and collapsing whitespace | Fail |
| Near duplicate | Jaccard similarity of 5-character shingles at or above the threshold | Warning |

A near duplicate is a warning for a reviewer to weigh, and blocks promotion only
under `--strict-warnings`.

## Evaluation leakage (G13)

G13 compares each candidate with the evaluation fixtures passed through
`--eval-fixtures`, twice: once with the full conversation and once with the
prompt only. The second comparison exists because the kleos-models
`conversation_text()` includes the assistant turn for training examples but not
for evaluation examples, so a single comparison can miss a training prompt that
matches an evaluation prompt. Any finding fails the gate. Without
`--eval-fixtures`, the evaluation corpus is empty and the gate passes.

## Bypass protection

Only G11 can be bypassed. Five mechanisms enforce this:

| Mechanism | Enforced by |
| --- | --- |
| Mandatory status is derived from the gate table. `GateSpec.bypassable` defaults to `False`, and `MANDATORY_GATE_IDS` is computed from `GATES`. | `promotion/gates.py` |
| `PromotionPolicy` is frozen, and its validator rejects a `bypass_gate_ids` value that names a mandatory or unknown gate. | `promotion/policy.py` |
| `--force` calls `PromotionPolicy.forced()`, which takes no gate argument and bypasses exactly `BYPASSABLE_GATE_IDS`. A test checks that `bypass_gate_ids=` is not set anywhere else. | `tests/test_promotion.py` |
| The runner raises `PromotionIntegrityError` if a mandatory gate did not run or was bypassed. | `promotion/runner.py` |
| Tests pin the gate count and the list of mandatory gates, so removing a gate fails the suite. | `tests/test_promotion.py` |

## Rejection codes

A rejection record stores closed-vocabulary reason codes and the content hash,
not the rejected text, so rejection causes can be counted without keeping the
content. There are 27 codes. The groups below are for readability and do not
exist in code.

| Code | Group | Emitted by |
| --- | --- | --- |
| `SCHEMA_INVALID` | Contract | G02 |
| `ID_MISMATCH` | Contract | G03 |
| `RENDER_ROUNDTRIP_MISMATCH` | Contract | G14 |
| `METADATA_EXTRA_NOT_ALLOWED` | Contract | Reviewer |
| `TASK_MISMATCH` | Contract | Reviewer |
| `AXES_INCOMPLETE` | Contract | G11 |
| `SECRET_DETECTED` | Privacy | G04, sanitization |
| `PII_UNRESOLVED` | Privacy | G05 |
| `SURROGATE_RESIDUE` | Privacy | G06 |
| `PRIVATE_FACT` | Privacy | G07 |
| `CONSENT_UNCLEAR` | Privacy | Reviewer |
| `REVIEW_MISSING` | Review | G08 |
| `REVIEW_SIGNATURE_MISMATCH` | Review | Reviewer |
| `REVIEW_GATE_FAILED` | Review | G09 |
| `REVIEW_SCORE_BELOW_THRESHOLD` | Review | Reviewer |
| `OPERATOR_REJECTED` | Review | Promotion, when no gate failed |
| `INCORRECT_BEHAVIOR` | Review | Reviewer |
| `BAD_POLICY` | Review | Reviewer |
| `CORPUS_DUPLICATE` | Corpus | G12 |
| `NEAR_DUPLICATE` | Corpus | Reviewer |
| `EVAL_LEAKAGE` | Corpus | G13 |
| `LOW_INFORMATION` | Corpus | Reviewer |
| `INSUFFICIENT_VARIATION` | Corpus | Reviewer |
| `UNRESOLVED_AMBIGUITY` | Corpus | Reviewer |
| `PROVENANCE_INVALID` | Provenance | Reviewer |
| `LANE_NOT_PROMOTABLE` | Provenance | G10 |
| `STAGING_INTEGRITY` | Provenance | G01 |

"Reviewer" codes are given with `record_decision.py --reason` on a rejection.
They are stored in the decision record and appear in the G09 evidence. The
promotion rejection record carries `REVIEW_GATE_FAILED` for that candidate. A
promotion rejection is marked retryable when none of the privacy promotion
gates (G04 to G07) failed.

## Related documentation

- [review.md](review.md): decision records and the review rubric
- [privacy.md](privacy.md): the detection layers behind G04 to G07
- [dataset-lifecycle.md](dataset-lifecycle.md): building a release from the
  promoted pool
- [architecture.md](architecture.md): how staging fits into the pipeline
