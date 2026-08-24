# Data governance

Who may hold what, for how long, and what happens when someone asks for their
data back.

> This states the repository's engineering position. It is not legal advice.
> Before real end-user data is used for training, the product's terms and its
> opt-out mechanics need review by qualified counsel — see `PRIVACY.md`.

## Custody

| Artifact | Where | Who can read it | Committed? |
| --- | --- | --- | --- |
| Raw captures | `staging/raw/` | operator | never |
| Sanitized candidates | `staging/sanitized/` | operator, reviewers | never |
| Entity vault, surrogate maps | `vault/` (chmod 700) | operator only | never |
| Review records | `staging/review/` | operator, reviewers | never |
| Rejection records | `staging/rejected/` | operator | never |
| Releases | `releases/` | whoever a release is shared with | never |
| Reports | `reports/` | operator | never |

`vault/` is the most re-identifying artifact this repository holds: it is
precisely the map from a fictional surrogate back to a real person. It is needed
to *re-derive* a release, never to consume one, and it never leaves the machine
that produced it.

## Releases are shipped by path, not by git

A release is consumed as a directory:

```bash
python <kleos-models>/scripts/train.py --config <config> --dataset <release-dir>
```

Committing releases was considered and rejected. Git history is permanent and
recoverable by anyone with a clone; a mistake in a committed release cannot be
withdrawn, only followed by an apology. Move releases to encrypted or
access-controlled artifact storage instead.

## Retention

| Artifact | Default | Why |
| --- | --- | --- |
| Raw captures | Delete once the sanitized candidate and its privacy record exist | Highest-risk artifact, shortest useful life |
| Sanitized candidates | Keep while a release derived from them is current | Needed to re-derive |
| Privacy records | Keep | Audit trail. Holds digests and redacted excerpts, never matched values |
| Review records | Keep | The provenance of a decision |
| Rejection records | Keep the reason code and hash; drop the content | "Which failure dominates?" must stay answerable without retaining the text that failed |
| Vault, surrogate maps | Keep while any derived release is current | Re-derivation |
| Releases | Immutable, kept | Reproducibility |

Minimizing raw retention is the single highest-value habit here. Once a sanitized
candidate and its privacy record exist, the raw capture has served its purpose
and is pure liability.

## Immutability, and what it costs

A dataset version is immutable. `build_release.py` refuses to overwrite one and
has no `--force`, because every comparison and every trained checkpoint that
named `kleos-policy-v0.1.0` meant one specific set of bytes.

That has a real cost, and it is worth stating plainly rather than glossing:
**an immutable release cannot be edited to remove someone's data.** See below.

## Versioning

Semantic, on the dataset rather than the code.

| Change | Bump |
| --- | --- |
| Metadata, formatting, a non-semantic preprocessing fix | patch — `v0.1.0` → `v0.1.1` |
| New reviewed examples, wider coverage, same contract | minor — `v0.1.x` → `v0.2.0` |
| Task definitions, schema assumptions, preprocessing behaviour, or research interpretation changes | major — `v0.x` → `v1.0.0` |

Every release carries a `CHANGELOG` entry, a manifest content hash, per-file
hashes, the split configuration, and the pinned contract commit.

## Deletion requests

If someone asks for their data to be removed, and applicable policy or law
requires honouring it, three things are true and they are not the same:

**Future releases.** Straightforward. Identify the affected candidates via
`staging/`, delete them, rebuild. The provenance record on each promoted example
carries enough to trace it back.

**Active releases.** A release cannot be edited. The response is a *new version*
with the material removed, plus a decision about whether the old version is
withdrawn from wherever it was shared. Withdrawal is a communication problem, not
a technical one.

**Trained checkpoints.** A model trained on the data cannot be edited either.
Deleting the source does not remove what a model learned from it. The honest
options are retraining from a corrected release, or accepting and documenting
that a checkpoint retains it.

**This repository does not pretend deleting a JSON file solves this.** What it
provides is the ability to answer *which* releases and *which* checkpoints are
affected, which is the part that is actually tractable.

Because the current pipeline promotes only `synthetic` and `mock_backend` lanes —
never `production_observation` — no release built so far contains anyone's data,
and no deletion request can apply to one. That is a deliberate consequence of the
lane design, not luck.

## Incidents

See `docs/incident-response.md`.
