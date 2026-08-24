# Privacy

The engineering requirements for handling data in this repository.

> **This document describes what the code does. It is not legal advice, and it
> does not establish that any particular consent mechanism is sufficient in any
> particular jurisdiction.** Before real end-user conversations are ever used for
> training, the product's terms, its disclosures and its opt-out mechanics need
> review by qualified counsel. Nothing in the current pipeline depends on that
> review, because the `real_sanitized` path is not enabled — see *Capture lanes*
> below.

## The principle

> **Train a generalizable decision policy, not private facts about a person.**

PII removal is necessary and **not sufficient**. Consider:

> Prioritize the Motorola project — your internship there ends in three weeks
> and your manager already flagged the deadline.

Strip every name and it still teaches a model that a specific person had a
specific internship ending on a specific timeline. A model that memorizes this
can surface it later, to someone else. The example is unusable no matter how
thoroughly it is scrubbed.

The transferable version of the same lesson:

> When one opportunity has a nearer deadline, confirmed stakeholder attention,
> and a higher cost of delay, rank it first — and say which of those three
> facts is doing the work.

This is why *private-fact review* is a separate stage from PII scanning, with
its own gate. PII is a **string** problem with a mechanical fix. A private fact
is a **semantic** problem with no mechanical fix at all.

## Capture lanes

The KLEOS backend answers from the *authenticated user's own* stored projects,
memories and notifications. It is not a scenario simulator. Every capture from it
is one person's private data, whatever the prompt was.

| Lane | Where the content comes from | May promote? |
| --- | --- | --- |
| `synthetic` | Scenario YAML, rendered locally, no backend | Yes |
| `mock_backend` | Deterministic offline adapter | Yes |
| `production_observation` | The real KLEOS backend | **Never** |

`production_observation` captures are rejected outright by promotion gate
`G10_PROVENANCE`. They exist for exactly one purpose: a human reads one in a
reviewer packet, learns what situation genuinely arises in use, and writes a
**new** generalized scenario from that understanding. The resulting example is
`synthetic_seeded`, has a different content hash, and has no textual descent from
the capture.

This keeps the honest research value — knowing which situations actually occur —
without ever training on the situation itself. It also means the consent question
above blocks nothing today.

Self-generated scenarios sent through a real backend are **not** `real_sanitized`
merely because a real deployment answered them. Product usage is not permission
for model training.

## The four detection layers

Deterministic gates are authoritative. An LLM may assist; it may never approve.

**1. Secrets** — API keys, tokens, JWTs, private key blocks, session cookies,
database URLs with passwords. Severity `block`: **never** auto-redacted, always a
hard reject. A secret in a candidate means the capture path itself is
compromised, and quietly replacing the string would hide that.

**2. PII** — email, phone, street address, postal code, government-style
identifiers, student and employee ids, URLs carrying tokens, home directory
paths, social handles, UUIDs, absolute dates, person-name and organization-name
heuristics. Severity `redact`.

**3. Entity vault** — an operator-maintained list of literals that regex cannot
find: a real employer, an advisor's name, a project codename that reads like an
ordinary noun. Lives in `vault/`, `chmod 700`, git-ignored, never in a release.
Entries are added via stdin rather than argv so they do not land in shell
history.

**4. Private-fact heuristics** — produce *signals*, never redactions. The
strongest is `fact.unsupported_entity`: a proper noun asserted in an assistant
turn that appears nowhere in the prompt. That is a model stating something it was
not told, which is either a hallucination or a memory — and both are
disqualifying.

## Redaction, then surrogates

Detection produces `[[PERSON_1]]`-style placeholders. Those placeholders are
**not** what gets trained on. A second stage substitutes a consistent *fictional*
surrogate drawn deterministically from a committed pool.

Training on `[[PERSON_1]]` teaches a model to emit bracket tokens and destroys
the naturalness the task depends on. Training on the real name teaches a private
fact. A fictional surrogate keeps the text natural and keeps coreference intact —
"Dana" is the same person across all three turns of a conversation.

Surrogates are keyed on **scenario family**, not globally and not per-example.
Within a conversation and its equivalence group the surrogate is stable; across
families the same real person maps to a *different* fictional person. That last
property is what actually defeats memorization: there is no cross-example entity
left to memorize.

The maps live in `vault/surrogate_maps/`. They are needed to re-derive a release,
never to consume one, and they never leave this machine.

## What the pipeline guarantees, and what it does not

**It does guarantee:**

- No candidate reaches a dataset without passing every privacy gate.
- A human, not a model, owns the final approval.
- A human cannot approve over a failing privacy or private-fact gate — the
  decision object cannot be constructed. The only way forward is to fix the
  content, which changes its hash and its id, which invalidates the review and
  demands a fresh one.
- Every promoted example carries provenance back to its capture, its ruleset
  version and its signed review.
- Releases are immutable, so "what was this model trained on?" always has an
  exact answer.

**It does not guarantee:**

- That the heuristics catch everything. They are a floor under human judgement.
  A reviewer who approves without reading is the failure mode no gate closes.
- That a sanitized example is safe. Sanitization fixes strings. Whether the
  *situation* is identifying is a judgement call, and it is the one the private-
  fact gate asks a human to make.
- That an already-trained checkpoint can be corrected. It cannot. This is why
  the gates run before promotion rather than before release.

## Retention

| Artifact | Default | Rationale |
| --- | --- | --- |
| Raw captures | Delete once the sanitized candidate and its privacy record exist | Highest-risk artifact, shortest useful life |
| Sanitized candidates | Keep while their release is current | Needed to re-derive |
| Privacy results | Keep | Audit trail; contains digests, never matched values |
| Review records | Keep | The provenance of a decision |
| Rejection records | Keep the reason code and hash; drop the content | "Which failure dominates?" must stay answerable without keeping the offending text |
| Vault and surrogate maps | Keep while any release derived from them is current | Re-derivation |
| Releases | Immutable, kept | Reproducibility |

Deletion requests are handled in `DATA_GOVERNANCE.md`. The honest summary: an
immutable release cannot be rewritten without breaking every comparison that
referenced it, so a request is satisfied in *future* releases plus an explicit
assessment of active datasets and trained checkpoints. Deleting a raw JSON file
does not solve this problem, and this repository does not pretend otherwise.

## Fixture policy

Everything committed under `data/` must be obviously fake: `Alice Example`,
`example.invalid`, `555-0100`. Never commit a real person's information to test
privacy scanning.

Fixtures that must genuinely *match* a secret pattern live inline in the test
module that asserts the match, and that module is listed in the scanner's
`SELF_EXEMPT`. Keeping them there rather than in `data/` is what lets the
committed fixtures stay strictly non-matching.
