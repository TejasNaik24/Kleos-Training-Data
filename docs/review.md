# Review

Deterministic gates are authoritative. A machine reviewer produces a structured
opinion. A human owns the decision — and cannot override the two things that are
not judgement calls.

## The rubric

Six dimensions, 0–4:

| Dimension | The question |
| --- | --- |
| `decision_correctness` | Is the decision the one the stated policy implies? |
| `evidence_grounding` | Is every claim traceable to something in the prompt? |
| `reasoning_quality` | Is the justification the *right* reason — not a right answer for a wrong one? |
| `actionability` | Could the reader act on this without a follow-up? |
| `format_compliance` | Does it match the format the task and axes call for? |
| `generalizability` | Would this reasoning transfer to a different person with different entities? |

Four PASS/FAIL hard gates: `no_private_data`, `policy_not_facts`,
`no_unsupported_claims`, `schema_and_contract_valid`.

## Gates dominate, structurally

An example can be well-written, correctly reasoned, perfectly formatted, and
still contain someone's phone number.

Saying so in a docstring is not enough, and neither is checking it in one
function — that check gets bypassed the first time someone constructs a verdict
another way, deserializes one from JSON, or refactors the decision logic. So it
is a **model invariant**:

```python
@model_validator(mode="after")
def _gates_dominate(self):
    if self.gates.any_failed() and self.decision != "rejected":
        raise ValueError(...)
```

A `ReviewVerdict` with a failing gate and any decision other than `rejected`
cannot exist — not from `decide()`, not from direct construction, not from
`model_validate` on a crafted payload. The test parametrizes all 15 non-empty
subsets of failing gates against both non-rejecting decisions, with perfect
scores throughout.

A per-dimension floor sits alongside the mean, because a 0 in one dimension is
not redeemed by 4s elsewhere. An example that is beautifully written and reaches
the wrong decision is not a 3.2.

## What a human may and may not do

| Action | Permitted |
| --- | --- |
| Override a numeric score | yes, with a mandatory justification |
| Overrule `no_unsupported_claims` or `schema_and_contract_valid` | yes |
| Approve over `no_private_data` or `policy_not_facts` | **never** |

The last is not a judgement about one example — it is the reason the repository
exists. The only way forward is to fix the content, which changes its
`content_hash`, which changes its id, which invalidates the review and demands a
fresh one. **The escape hatch is the correct workflow.**

An unexplained score override is indistinguishable from a mistake, so
`override_justification` is required. A rejection requires a reason code from the
closed vocabulary, because "which failure dominates?" must stay answerable.

## Decisions are signed against bytes

`signature` covers the record including the `content_hash` of the payload
reviewed. Edit the candidate afterwards and the signature stops matching, so an
approval never silently carries over to text nobody read. Gate `G08` calls
`applies_to()` and treats a mismatch as *no approval at all* — not a weaker one.

## The machine reviewer

Validated twice, because the layers fail differently:

- **JSON Schema** catches a malformed generation with a field-level message worth
  retrying on.
- **Pydantic** enforces the cross-field rules JSON Schema expresses badly: every
  `FAIL` carries evidence, confidence is present, gates dominate.

A test asserts the two accept and reject exactly the same fixtures, so they
cannot drift into disagreeing.

A `FAIL` without evidence is rejected. A failure nobody can act on is worse than
no review — and it is also the shape a model produces when it is guessing.

`MockReviewer` is deterministic and derived from the deterministic signals. It
does not pretend to have an opinion: it reads the privacy scan, the fact
assessment and the structural properties. That makes it useful for proving the
review *stage* works and useless for judging quality, which is the honest
division. CI uses it, so review is testable with no network and no key.

`AnthropicReviewer` is behind the `review` extra and refused when `CI` is set —
an automated run has no human to own the decision afterwards.

## The packet

The one place candidate text is meant to be read, which is why logs and reports
never carry content.

It leads with the scenario's `policy_claim` and `anti_claim`, so the reviewer is
asked "does this teach that?" rather than the much weaker "does this look fine?".
It inlines privacy detections and fact-risk spans next to the text they refer to,
and shows the nearest already-promoted examples — so redundancy is visible
*before* approval rather than at deduplication afterwards.
