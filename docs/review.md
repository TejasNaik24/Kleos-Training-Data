# Review

Every candidate needs an approving decision record before it can be promoted. A
machine review scores the candidate against a rubric and proposes results for
four review hard gates. A decision record then approves, rejects or returns the
candidate, either written by a reviewer or filled from the machine review's
results. Promotion re-checks privacy on the exact bytes, independently of both.

## Contents

- [Review flow](#review-flow)
- [Rubric](#rubric)
- [Review hard gates](#review-hard-gates)
- [Decision records](#decision-records)
- [Combined verdict](#combined-verdict)
- [Machine review](#machine-review)
- [Review packets](#review-packets)

## Review flow

| Step | Command | Output |
| --- | --- | --- |
| Build a packet | `python scripts/build_review_packet.py --batch <batch>` | `staging/review/packets/pk-<batch>/` |
| Run a machine review | `python scripts/run_llm_review.py --packet-id pk-<batch> --reviewer mock` | One machine review record per candidate in `staging/review/llm/` |
| Record a decision | `python scripts/record_decision.py --candidate <id> --decision approve --adopt-machine-gates` | A decision record in `staging/review/decisions/` |
| Promote | `python scripts/promote_examples.py --batch <batch>` | Promotion gates G08 and G09 check the decision record |

Only a decision record approves a candidate. `--adopt-machine-gates` fills one
from the machine review's gate results, which is how the approvals in releases to
date were recorded. A release intended for training should have decisions
recorded by a person who has read the review packet.

## Rubric

Rubric version `review-rubric-v1` scores six dimensions from 0 to 4:

| Dimension | Question |
| --- | --- |
| `decision_correctness` | Is the decision the one the stated policy implies? |
| `evidence_grounding` | Is every claim traceable to something in the prompt? |
| `reasoning_quality` | Is the justification the right reason, not a right answer reached for a wrong or unstated reason? |
| `actionability` | Could the reader act on this without a follow-up question? |
| `format_compliance` | Does it match the format the task and axes call for? |
| `generalizability` | Would the same reasoning transfer to a different person with different entities? |

A verdict is decided in this order:

| Condition | Verdict |
| --- | --- |
| Any review hard gate is `FAIL` | `rejected` |
| Any dimension scores below 2, or the mean is below the minimum (3.0 by default) | `needs_revision` |
| Otherwise | `approved` |

The per-dimension floor keeps a high mean from hiding one failing dimension. A
score of 0 or 1 in any dimension, such as `decision_correctness` for an answer
that reaches the wrong decision, sends the candidate back for revision
regardless of the other scores.

## Review hard gates

| Gate | Passes when |
| --- | --- |
| `no_private_data` | No PII, secret or identifying detail survives in the text |
| `policy_not_facts` | The example teaches a transferable policy, not a fact about a real individual |
| `no_unsupported_claims` | The answer asserts nothing the prompt does not support |
| `schema_and_contract_valid` | The example satisfies the dataset contract |

`ReviewVerdict` enforces gate dominance with a model validator: a verdict with
any failing gate and a decision other than `rejected` raises a validation error.
The check runs on every construction path, including `model_validate` on a
deserialized record. `tests/test_review.py` constructs all 15 non-empty
combinations of failing gates with perfect scores and asserts that none can be
`approved` or `needs_revision`.

## Decision records

`record_decision.py` writes one decision record per candidate:

| Field | Content |
| --- | --- |
| `decision` | `approve`, `reject` or `revise` |
| `gates` | A `PASS` or `FAIL` result for each of the four review hard gates |
| `score_overrides`, `override_justification` | Optional score changes and the reason for them |
| `reason_codes` | [Rejection codes](staging.md#rejection-codes), required for a rejection |
| `notes` | Free text |
| `reviewer_role` | A role such as `operator`, never a personal name |
| `content_hash` | Hash of the sanitized candidate that was reviewed |
| `signature` | SHA-256 digest of the record's canonical JSON, including `content_hash` |

Gate results come from `--gate NAME=PASS|FAIL`, given once for each of the four
gates, or from `--adopt-machine-gates`, which starts from the machine review's
results and prints them with their evidence. Explicit `--gate` values override
adopted ones.

The record validators enforce these rules:

- `approve` requires all four review hard gates to be `PASS`.
- `score_overrides` require a non-empty justification, known dimensions and
  values from 0 to 4.
- `reject` requires at least one reason code.

The digest binds the decision to the reviewed content. Promotion gate G08
accepts a decision only if the digest is valid and its `content_hash` equals the
candidate's current hash, so editing a candidate after review leaves it with no
valid decision. The digest is unkeyed. It detects changes made outside the
review workflow, but it is not a cryptographic signature and does not
authenticate the reviewer.

## Combined verdict

Promotion gate G09 combines the decision record with the machine review:

- The decision record's gate results replace the machine review's.
- Scores come from the machine review, with any `score_overrides` applied. When
  no machine review exists, every dimension defaults to 4, so the mean threshold
  only constrains candidates that have a machine review.
- The result must be `approved`, and the decision must be `approve`.

Promotion gate G07 adds one more rule. A reviewer can clear a
`needs_fact_review` assessment by recording `policy_not_facts=PASS`, but a
candidate assessed as `fact_teaching` fails G07 regardless of the decision.

## Machine review

A machine review record is validated twice, because the two layers catch
different errors:

| Layer | Checks |
| --- | --- |
| JSON Schema (Draft 2020-12) | Required fields are present, no extra fields appear, the rationale has at least 20 characters and the confidence is between 0 and 1. Errors are field-level and can be fed back to a model on retry. |
| Pydantic | Every failing gate carries evidence, evidence names real gates, and gates dominate the verdict. |

A test checks that both layers accept and reject the same sample payloads.

| Reviewer | Behavior |
| --- | --- |
| `mock` | Deterministic. Derives gate results from the privacy detections, the fact assessment and contract validity, and starts every score at 4 with fixed deductions. Used in CI. It checks the review stage, not answer quality. |
| `anthropic` | Sends the packet item to Claude (default model `claude-sonnet-5`) with up to 3 attempts, returning schema errors on retry. Requires the `review` extra and `ANTHROPIC_API_KEY`. `run_llm_review.py` refuses any reviewer other than `mock` when the `CI` environment variable is set. |

The mock reviewer fails `no_private_data` on any remaining detection,
`policy_not_facts` on a `fact_teaching` assessment, `no_unsupported_claims` on
any unsupported-entity signal, and `schema_and_contract_valid` when the payload
fails contract validation.

## Review packets

A review packet is the place to read candidate text. It is written to
`staging/review/packets/<packet-id>/`, which is git-ignored, and the default
packet ID is `pk-<batch>`.

| File | Content |
| --- | --- |
| `packet.md` | The human-readable packet |
| `packet.jsonl` | One item per candidate: task, policy claims, payload, axes, contract validity, privacy summary and fact signals |
| `packet.meta.json` | Packet ID, batch, candidate list and the expected response schema |

For each candidate, `packet.md` shows the rubric, the family's `policy_claim`
("Should teach") and `anti_claim` ("Must not teach"), the variation axes, the
conversation, the privacy detections and fact verdict, and up to three of the
most similar already promoted examples (`--include-neighbors`, default 3).
Showing both claims asks the reviewer whether the example teaches the intended
policy instead of whether it merely looks correct, and showing neighbors exposes
redundancy before approval.

## Related documentation

- [staging.md](staging.md): promotion gates G07 to G09 and rejection codes
- [privacy.md](privacy.md): the detections and fact signals shown in packets
- [scenarios.md](scenarios.md): where `policy_claim` and `anti_claim` come from
- [../PRIVACY.md](../PRIVACY.md): what review contributes to the privacy
  guarantees
