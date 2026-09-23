# Privacy

This document describes how KLEOS Training Data handles personal data, what the
pipeline guarantees, and where its guarantees end. Implementation details are in
[docs/privacy.md](docs/privacy.md).

> This document describes engineering practice. It is not legal advice and does
> not establish that any consent mechanism is sufficient in any jurisdiction.
> Before real end-user conversations are used for training, the KLEOS product's
> terms, disclosures and opt-out mechanics need review by qualified counsel. The
> current pipeline does not depend on that review, because it never promotes
> data from real users.

## Contents

- [Principle](#principle)
- [Data sources](#data-sources)
- [Protection layers](#protection-layers)
- [Guarantees](#guarantees)
- [Limitations](#limitations)
- [Retention, deletion and incidents](#retention-deletion-and-incidents)

## Principle

> Train a generalizable decision policy, not private facts about a person.

A training example should teach a rule that transfers to anyone. Compare two
targets for the same kind of question:

| Target | Example | Problem |
| --- | --- | --- |
| Private fact | "Prioritize the robotics project. Marcus Holloway's internship at Acme Labs ends in three weeks and his manager already flagged the deadline." | Teaches a fact about one person. A model that memorizes it can repeat it to someone else. |
| Transferable policy | "When one task has a nearer deadline, confirmed evidence that it matters and a higher cost of delay, rank it first and name the factor that decided it." | None. It applies to anyone. |

Removing the name does not fix the first example. It still describes a specific
person with a specific internship ending on a specific date. PII detection
handles strings, and a separate private-fact review handles situations like this
one, because no string substitution can repair them.

## Data sources

Every example in every release so far was generated from the synthetic scenario
catalog and captured through the offline mock backend. No release contains
conversations, records or other data from real users.

| Lane | Source | Promotable |
| --- | --- | --- |
| `mock_backend` | The deterministic offline adapter, rendering catalog scenarios | Yes |
| `synthetic` | Scenarios rendered locally without a backend. Defined, but not produced by the current tools. | Yes |
| `production_observation` | The live KLEOS backend | No. Promotion gate G10 rejects the lane. |

The live KLEOS backend answers from the authenticated user's own projects,
memories and notifications, so every capture from it is personal data regardless
of the prompt. Such a capture can only inform a new scenario that a person writes
by hand. It never becomes a training example. Using the product does not grant
permission to train on the result. The capture lanes and the safeguards around
production capture are described in
[docs/collection.md](docs/collection.md#capture-lanes).

## Protection layers

| Stage | Protection |
| --- | --- |
| Sanitization | Secrets block the candidate. PII, entity-vault literals and name heuristics are replaced with fictional surrogates. Private-fact signals are recorded for review. |
| Review | A decision record cannot approve an example while its `no_private_data` or `policy_not_facts` review hard gate is `FAIL`. |
| Promotion | Mandatory promotion gates G04 to G07 re-scan the exact bytes being promoted for secrets, PII, surrogate residue and private facts. |
| Release | The written split files are scanned byte by byte, and the result is recorded as `contains_private_data` in the manifest. |
| Repository | Runtime zones (`staging/`, `vault/`, `releases/`, `reports/`) are git-ignored, and a standard-library scanner checks the repository before commits and in CI. |

## Guarantees

The following hold for every promoted example and every release, and each is
enforced by code:

- The example passed promotion gates G04 to G07, which cannot be bypassed.
- The example has an approving decision record bound to its content hash.
  Editing the content afterwards invalidates the approval at promotion gate G08.
- A secret was never auto-redacted. A candidate containing one is rejected.
- The example keeps an audit record in `staging/promoted/`: its content hash,
  all 14 promotion gate results, the digest of its decision record, the privacy
  ruleset version, the scenario fingerprint, the capture lane and the batch.
- The release is immutable and content-hashed, so the data a model was trained
  on can always be identified exactly.

## Limitations

- Pattern matching and heuristics reduce risk but do not catch everything. They
  support human review and do not replace it.
- A sanitized example can still describe an identifying situation. Deciding
  whether it does is the purpose of the private-fact review.
- A decision record shows that approval was recorded, not how carefully the
  example was read. The approvals in releases to date were filled from machine
  review results with `--adopt-machine-gates`, while the privacy promotion gates
  still checked every example.
- Deleting data does not remove it from a model already trained on it. See
  [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md#deletion-requests).

## Retention, deletion and incidents

- Retention periods and deletion requests: [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md)
- Responding to exposed secrets or personal data:
  [docs/incident-response.md](docs/incident-response.md)
- Reporting a privacy or security concern:
  [SECURITY.md](SECURITY.md#reporting-a-vulnerability)

## Related documentation

- [docs/privacy.md](docs/privacy.md): detection rules, surrogates and fact
  signals
- [docs/review.md](docs/review.md): review hard gates and decision records
- [docs/staging.md](docs/staging.md): the promotion gates
- [CONTRIBUTING.md](CONTRIBUTING.md#fixture-policy): rules for test fixtures
