# Incident response

Runbooks for three kinds of incident: an exposed secret, personal data found in
a promoted example or release, and a consent problem. The last sections cover
trained models and how an incident is recorded.

## Contents

- [Principles](#principles)
- [Exposed secret](#exposed-secret)
- [Personal data in a promoted example or release](#personal-data-in-a-promoted-example-or-release)
- [Consent problem](#consent-problem)
- [Trained models](#trained-models)
- [Post-incident record](#post-incident-record)

## Principles

- Contain first and clean up second.
- Removing a file in a later commit does not remove it from git history or from
  existing clones.
- Never edit a sealed release. Publish a corrected release under a new version.
- Every incident ends with a test that would have caught it.

## Exposed secret

1. Rotate the credential immediately, and assume it has been used.
2. Stop distribution. Do not push, and do not share the branch.
3. Remove the secret from the working tree.
4. If it was committed, rewrite history with `git filter-repo`, and coordinate
   any force-push with everyone who has a clone.
5. Confirm the old credential no longer works by trying to use it.
6. Check the provider's access logs for use during the exposure window.
7. Record the incident in [CHANGELOG.md](../CHANGELOG.md) under a Security
   heading.

A secret inside a candidate is handled by the pipeline itself. Secrets have
`block` severity, are never redacted, and cause the candidate to be rejected with
`SECRET_DETECTED`. Treat it as a problem with the capture path, and find out how
a credential reached a prompt before capturing again.

## Personal data in a promoted example or release

1. Stop. Do not build from, train on or share the affected release.
2. Identify the scope:

   ```bash
   python scripts/verify_release.py --release releases/kleos-policy-v0.0.6 --strict
   ```

   Each promoted example's audit record in `staging/promoted/` links it to its
   capture batch, lane, privacy ruleset version and decision record.
3. Quarantine the release by moving it out of `releases/` so nothing picks it
   up. Do not edit it. An edited release would keep its version name while
   containing different bytes.
4. Find out how the example passed. Promotion gates G04 to G07 exist to catch
   this, so either a detector missed it or the rules changed after promotion.
   Add the case to `tests/test_privacy.py` as a failing test before fixing the
   detector.
5. Publish a corrected release under a new version.
6. Assess any model trained on the affected release, as described in
   [Trained models](#trained-models).

## Consent problem

Use this runbook when data was used without a valid consent basis.

1. Stop capturing from the source.
2. Determine which examples derive from it. Captures in the
   `production_observation` lane cannot be promoted, because promotion gate G10
   rejects the lane, so no release should contain such material. If one does, a
   gate was bypassed or a capture was mislabeled, and that is a separate
   incident.
3. Involve the product and legal owners. This is not an engineering decision.
4. Block promotion from the source until the question is resolved.

## Trained models

A trained model cannot be edited to remove what it learned, and deleting the
source data does not change the model. The options are to retrain from a
corrected release, or to document that the checkpoint retains the material. The
right choice depends on what the material is and where the checkpoint was shared.

Release content hashes and per-example audit records identify which releases,
and therefore which checkpoints, are affected.

## Post-incident record

Record every incident in [CHANGELOG.md](../CHANGELOG.md) under a Security
heading. Include what happened, what was affected, what was done, and what
changed to prevent a repeat, including the test that now covers the case.

## Related documentation

- [../SECURITY.md](../SECURITY.md): reporting and credential handling
- [../DATA_GOVERNANCE.md](../DATA_GOVERNANCE.md): deletion requests and retention
- [privacy.md](privacy.md): the detection layers
- [staging.md](staging.md): the promotion gates
