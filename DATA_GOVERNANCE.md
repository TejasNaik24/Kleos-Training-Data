# Data governance

This document defines who holds each artifact the pipeline produces, how
releases are distributed, how long artifacts are kept, and how deletion requests
are handled. It describes engineering practice. See the note at the top of
[PRIVACY.md](PRIVACY.md) about legal review.

## Contents

- [Custody](#custody)
- [Distribution](#distribution)
- [Retention](#retention)
- [Immutability](#immutability)
- [Deletion requests](#deletion-requests)
- [Incidents](#incidents)

## Custody

| Artifact | Location | Access | Committed to git |
| --- | --- | --- | --- |
| Raw captures | `staging/raw/` | Operator | No |
| Normalized and sanitized candidates | `staging/normalized/`, `staging/sanitized/` | Operator and reviewers | No |
| Review packets and decision records | `staging/review/` | Operator and reviewers | No |
| Rejection records | `staging/rejected/` | Operator | No |
| Promoted pool and audit records | `staging/promoted/` | Operator | No |
| Entity vault | `vault/` (directory 0700, file 0600) | Operator only | No |
| Releases | `releases/` | Whoever a release is shared with | No |
| Reports | `reports/` | Operator | No |

The entity vault is the most sensitive artifact. It lists the real names and
terms that sanitization must replace, so it is needed to sanitize new captures
but never to use a release, and it stays on the machine where it was created.

## Distribution

A release is consumed as a directory path:

```bash
python ../Kleos-Models/scripts/train.py --config <config> --dataset releases/kleos-policy-v0.0.6
```

Releases are not committed to git. Git history is permanent and copied to every
clone, so a release committed by mistake could never be fully withdrawn. Share
releases through access-controlled storage, and give recipients the release's
`content_hash` so they can confirm what they received with
`scripts/verify_release.py`.

## Retention

The pipeline does not delete staged artifacts automatically. These are the
retention defaults for the operator:

| Artifact | Default | Reason |
| --- | --- | --- |
| Raw captures | Delete once the sanitized candidate and its privacy record exist | The highest-risk artifact, with the shortest useful life |
| Sanitized candidates | Keep while a release derived from them is current | Needed to rebuild the release |
| Privacy records | Keep | Audit trail. They contain digests and masked excerpts, never matched values. |
| Review and decision records | Keep | Record of who decided what, by role |
| Rejection records | Keep | They contain reason codes, a content hash and gate messages, not the rejected text, so rejection causes stay countable |
| Entity vault | Keep while any release derived from it is current | Needed to sanitize again |
| Releases | Keep, unmodified | Reproducibility |

## Immutability

A release cannot be modified. `build_release.py` refuses to write a version that
already exists and has no option to overwrite one, and `verify_release.py`
detects any change after sealing. Every comparison and every model trained on a
version refers to one specific set of bytes, so a correction is always published
as a new version. See [docs/dataset-lifecycle.md](docs/dataset-lifecycle.md) for
the versioning rules.

This has a cost: data cannot be removed from an existing release.

## Deletion requests

If someone asks for their data to be removed, and applicable policy or law
requires it, the response depends on where the data is:

| Scope | Response |
| --- | --- |
| Future releases | Remove the affected candidates from `staging/`, using the promoted audit records and `index.jsonl` to find them, and rebuild under a new version |
| Existing releases | Publish a new version without the material, and decide whether to withdraw the old version from wherever it was shared |
| Trained models | Deleting data does not remove what a model learned from it. Retrain from a corrected release, or document that the checkpoint retains the material. |

The pipeline makes the affected releases identifiable. Every release is
content-hashed, and every promoted example has an audit record that ties it to
its capture batch, lane, scenario and decision record.

Every release built so far comes from synthetic scenarios captured through the
offline mock backend, so none contains data from real users and no deletion
request can apply to one.

## Incidents

Exposed credentials, personal data found in a release, and consent problems are
handled by the runbooks in [docs/incident-response.md](docs/incident-response.md).

## Related documentation

- [PRIVACY.md](PRIVACY.md): the privacy policy and its guarantees
- [SECURITY.md](SECURITY.md): reporting issues and handling credentials
- [docs/dataset-lifecycle.md](docs/dataset-lifecycle.md): sealing, verification
  and versioning
