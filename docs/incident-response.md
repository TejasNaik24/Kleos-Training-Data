# Incident response

Three kinds of incident. The common thread: **rotate or contain first, tidy up
second**, and never confuse deleting a file with fixing the problem.

## An exposed secret

Deleting the file in a later commit does **not** remove it from history.

1. **Rotate the credential immediately.** Assume it is compromised. Do this
   before anything else — cleanup can wait, rotation cannot.
2. Stop distribution: do not push, do not share the branch.
3. Remove it from the working tree.
4. Rewrite history with `git filter-repo`, and force-push only after
   coordinating with anyone holding a clone.
5. **Verify the old credential is actually invalid** by trying to use it. A
   rotation you did not confirm is a rotation you did not do.
6. Audit access logs for use during the exposure window.
7. Record it in `CHANGELOG.md` under a `Security` heading.

If the secret reached a *candidate* rather than the repository, the pipeline has
already refused it: secrets are `block` severity, never redacted, and the
candidate is rejected with `SECRET_DETECTED`. That is a signal about the **capture
path**, not about one example. Investigate how a credential reached a prompt
before re-running anything.

## PII in a promoted example or a release

1. **Stop.** Do not build, do not train, do not share the release further.
2. Identify the scope:
   ```bash
   python scripts/verify_release.py --release <dir>
   python scripts/coverage_report.py --release <dir> --json reports/coverage/incident.json
   ```
   `provenance.json` and `staging/promoted/<id>.json`'s audit block trace each
   example back to its capture, ruleset version and signed review.
3. Quarantine the release: move it out of `releases/` so nothing picks it up.
   **Do not edit it** — an edited release still carries its old version string and
   every reference to that version now means two different things.
4. Determine how it passed. Gates G04–G07 exist to catch exactly this, so a
   promoted example carrying PII means either a detection gap or a rule change
   since promotion. Add the case to `tests/test_privacy.py` first, so the fix has
   a failing test.
5. Publish a corrected **new version**. Never reuse the old one.
6. Assess trained checkpoints separately — see below.

## A consent problem

Someone's data was used on a basis that does not hold.

1. Quarantine the source: mark the lane and stop further capture from it.
2. Determine which examples derive from it. Under the current design this should
   be *none in any release*, because `production_observation` can never be
   promoted — gate G10 rejects the lane outright. If a release does contain such
   material, that is itself the incident: a gate was bypassed or the lane was
   mislabelled.
3. Consult product and legal. This is not an engineering decision.
4. Block promotion from that source until it is resolved.

## Trained checkpoints

**A model trained on the data cannot be edited.** Deleting the source does not
remove what a model learned from it.

The honest options are: retrain from a corrected release, or accept and document
that a checkpoint retains the material. Which one is appropriate depends on what
the material is and where the checkpoint went. Pretending there is a third option
is how this goes wrong.

What this repository provides is the ability to answer *which* releases and
*which* checkpoints are affected, because every release is immutable and
content-hashed and every example traces back to its provenance. That is the
tractable part.

## Writing it up

Every incident gets a `CHANGELOG.md` entry recording what happened, what was
affected, what was done, and what changed so it cannot recur. An incident without
a test is an incident that will happen again.
