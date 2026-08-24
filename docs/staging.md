# Staging and promotion

## Records

```
staging/raw/<batch>/          RawCapture + _batch.json
staging/normalized/<batch>/   NormalizedCandidate
staging/sanitized/<batch>/    SanitizedCandidate + <id>.privacy.json
staging/review/               packets/, llm/, decisions/
staging/rejected/             RejectionRecord
staging/promoted/             PromotedExample, index.jsonl, gate_reports/
```

Every record is `extra="forbid"`, written atomically (temp file + `os.replace`),
and carries a `record_hash` recomputed on read.

That last is not paranoia about tampering. It is the ordinary case of someone
fixing a typo by hand in a candidate after it was reviewed — which silently
invalidates the signature bound to the old bytes and would otherwise promote an
example nobody approved.

Two filename conventions, both load-bearing: a leading `_` marks directory
metadata (`_batch.json`), and a second dot marks a sidecar
(`<id>.privacy.json`). Without them, iterating a batch tries to parse its own
manifest as a record.

## Normalization

Deterministic and **non-semantic**: line endings, reasoning spans, trailing
whitespace, transport envelopes. It never touches meaning.

> Never silently change an assistant answer and then present it as the original
> model output.

Every change is appended to `transformations` on the candidate, so a reviewer
sees what was done to the text before reading it.

Normalization also refuses to overwrite: if two captures normalize to identical
content, that is reported rather than allowed to silently shrink the batch. It
means two scenario points render the same text, which is a catalog problem.

## The fourteen gates

All of them run — no short-circuit — so one pass tells you everything wrong with
a candidate rather than one thing at a time.

| # | Gate | Mandatory |
| --- | --- | --- |
| G01 | `STAGING_INTEGRITY` — the record hash still matches | ✓ |
| G02 | `SCHEMA_VALID` — satisfies the public contract | ✓ |
| G03 | `ID_INTEGRITY` — the id re-derives from the content | ✓ |
| G04 | `SECRET_SCAN` | ✓ privacy |
| G05 | `PII_SCAN` | ✓ privacy |
| G06 | `SURROGATE_INTEGRITY` — no vault literal, no placeholder residue | ✓ privacy |
| G07 | `PRIVATE_FACT` | ✓ privacy |
| G08 | `REVIEW_PRESENT` — a signature-valid decision covers *this* content | ✓ |
| G09 | `REVIEW_APPROVED` | ✓ |
| G10 | `PROVENANCE` — lane and source permit promotion | ✓ |
| G11 | `COVERAGE_AXES` | bypassable |
| G12 | `CORPUS_DEDUP` | ✓ |
| G13 | `EVAL_LEAKAGE` | ✓ |
| G14 | `CONTRACT_RENDER` — round-trips, metadata extras allowlisted | ✓ |

Gates re-scan the **bytes about to be promoted** rather than trusting the
sanitization record. The record says what sanitization believed; the scan says
what is actually there.

### Why G12 is mandatory

An earlier draft marked it bypassable "for near-duplicates only". That reasoning
was wrong about its own mechanism: a bypassed gate does not run *at all*, so the
bypass would have skipped exact and normalized duplicates too.

No bypass is needed. The gate distinguishes them itself — exact and normalized
duplicates `FAIL`, near-duplicates `WARN`, and a `WARN` does not block unless
`strict_warnings` is set. A human weighs the near-duplicate; nobody waves through
an exact one.

### Why G13 runs twice

`conversation_text()` includes the assistant turn for a `TrainingExample` but is
prompt-only for an `EvaluationExample`. A single comparison is therefore
asymmetric exactly where it matters: a training candidate whose *prompt* matches
an eval prompt is leakage regardless of the answers. So the gate compares once
with public semantics — so our verdict is never weaker than theirs — and once
prompt-to-prompt.

## Why `--force` cannot reach a privacy gate

Four independent mechanisms, because one is a convention and four is an
architecture.

1. **Mandatory-ness is derived from the table.** `bypassable` defaults to
   `False`; `MANDATORY_GATE_IDS` is computed from `GATES`. A gate added without
   thought is mandatory, and there is no second list to forget to update.
2. **An illegal policy cannot be constructed.** `PromotionPolicy` is frozen and
   validates `bypass_gate_ids ∩ MANDATORY_GATE_IDS` is empty. No code path
   *holds* such an object.
3. **`--force` maps to a constant.** It produces `BYPASSABLE_GATE_IDS`. There is
   no `--bypass-gate G04` flag and no config key that reaches the field — a test
   greps the source tree to prove it.
4. **The runner asserts every mandatory gate ran.** This catches what the other
   three miss: a refactor that deletes a gate or short-circuits the loop, where
   the *absence* of a check is indistinguishable from a pass.

A gate that raises is recorded as `FAIL`, not skipped — and does not take the run
down, because the other thirteen still have something to say.

## Rejections

A closed vocabulary of reason codes. Free-text reasons cannot be aggregated, and
"which failure dominates?" is the question this has to answer.

The record keeps the reason and the content hash, not the offending text.
