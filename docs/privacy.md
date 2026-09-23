# Privacy engineering

[PRIVACY.md](../PRIVACY.md) states the privacy policy. This document describes
how the pipeline implements it: the detection layers and rules, redaction and
surrogate substitution, what a detection record contains, and how private facts
are flagged.

## Contents

- [Detection layers](#detection-layers)
- [Rule catalog](#rule-catalog)
- [Sanitization flow](#sanitization-flow)
- [Surrogates](#surrogates)
- [Detection records](#detection-records)
- [Private-fact signals](#private-fact-signals)
- [False positives](#false-positives)

## Detection layers

| Layer | Severity | Rules | Handling |
| --- | --- | --- | --- |
| Secrets | `block` | 15 | The candidate is rejected with `SECRET_DETECTED`. Secrets are never auto-redacted, because a credential in a capture means the capture path itself needs investigation. |
| PII | `redact` | 11 | Replaced with a placeholder and then a fictional surrogate |
| Entity vault | `redact` | Registered literals | Operator-supplied names that no pattern can find, such as an employer or a project codename. Matched longest first and replaced like PII. |
| Structural heuristics | `review` | 2 | Capitalized name bigrams and organization suffixes. Replaced like PII, and any match that remains after replacement marks the candidate `needs_review`. |

Detection covers message content, message names and string-valued variation
axes. Where two detections overlap, the more severe one wins, then the longer
span, then the earlier one.

The entity vault lives in `vault/entity_vault.json`. The file is written with
mode 0600, the `vault/` directory is 0700, and both are git-ignored. Entries are
added through standard input so that a real name never appears in shell history:

```bash
python scripts/sanitize_candidates.py --add-vault-entry PERSON
```

The command reads one literal per line until a blank line. Valid slots are
`PERSON`, `ORG`, `PROJECT`, `PLACE` and `PRODUCT`.

## Rule catalog

The rules are defined in `src/kleos_training_data/privacy/rules.py` under ruleset
version `privacy-rules-v1`.

| Layer | Rule IDs | Replacement slot |
| --- | --- | --- |
| Secrets | `secret.aws_access_key.v1`, `secret.openai_key.v1`, `secret.anthropic_key.v1`, `secret.hf_token.v1`, `secret.github_token.v1`, `secret.slack_token.v1`, `secret.google_api_key.v1`, `secret.jwt.v1`, `secret.private_key_block.v1`, `secret.supabase_url.v1`, `secret.supabase_service_key.v1`, `secret.assigned_secret.v1`, `secret.bearer_token.v1`, `secret.session_cookie.v1`, `secret.database_url.v1` | None |
| PII | `pii.email.v1` | `EMAIL` |
| PII | `pii.us_phone.v1` | `PHONE` |
| PII | `pii.ssn.v1`, `pii.student_id.v1`, `pii.uuid.v1` | `ID` |
| PII | `pii.home_path.v1` | `PATH` |
| PII | `pii.url_with_token.v1` | `URL` |
| PII | `pii.street_address.v1` | `ADDRESS` |
| PII | `pii.postal_code.v1` | `POSTAL` |
| PII | `pii.social_handle.v1` | `HANDLE` |
| PII | `pii.absolute_datetime.v1` | `DATE` |
| Structural | `structural.org_name.v1` | `ORG` |
| Structural | `structural.person_name.v1` | `PERSON` |
| Entity vault | `vault.person.v1`, `vault.org.v1`, `vault.project.v1`, `vault.place.v1`, `vault.product.v1`, created at runtime from the vault entries | The entry's slot |

Allowlists keep placeholder data from being flagged: `example.invalid`-style
email domains, the nil UUID, and common phrases such as "Due Date" and "Pull
Request".

## Sanitization flow

`sanitize_candidates.py` runs each normalized candidate through four steps:

1. Detect across all layers and assess private-fact risk.
2. If any secret is present, stop. The candidate is `blocked`, a rejection
   record is written, and the script exits with code 4.
3. Replace each redactable detection with a placeholder such as `[[PERSON_1]]`,
   then replace each placeholder with a surrogate.
4. Scan the result again. A remaining secret blocks the candidate. A remaining
   review-level detection, or a fact-risk verdict other than `policy_like`,
   marks it `needs_review`.

| Status | Meaning |
| --- | --- |
| `clean` | Nothing was detected |
| `sanitized` | Detections were replaced and nothing remains |
| `needs_review` | A human must look at residual detections or fact risk |
| `blocked` | A secret was found. The candidate cannot proceed. |

Each result is written next to the candidate as `<id>.privacy.json`.

## Surrogates

Placeholders are not shipped. Training on `[[PERSON_1]]` would teach a model to
emit bracket tokens, and training on the original value would teach the private
fact. Each placeholder is replaced with a fictional value instead.

| Slot | Replacement |
| --- | --- |
| `PERSON` | A name from `person_pool_a` |
| `ORG`, `PROJECT`, `PLACE`, `PRODUCT` | A name from `generic_pool_a` |
| `EMAIL` | `<name>@example.invalid` |
| `HANDLE` | `@<name>` |
| `PHONE` | `555-0100` |
| `ID` | `ID-000<n>` |
| `PATH` | `/Users/example/<name>` |
| `URL` | `https://example.invalid/<name>` |
| `ADDRESS` | `100 Example Street` |
| `POSTAL` | `00000` |
| `DATE` | `next Friday` |

Surrogate choice is deterministic. Within a candidate, each distinct value gets
one placeholder, numbered in order of first appearance. The surrogate for a
placeholder is the pool value with the lowest stable hash of
`<family>:<placeholder>:<value>`, so it depends on the scenario family and the
placeholder label, not on the original text:

- A placeholder resolves to the same surrogate everywhere in a family, which keeps
  names consistent across a conversation and its perturbations.
- Different families usually resolve the same placeholder to different names.
  With 16 names in `person_pool_a`, collisions across families are possible.
- No mapping table is stored. The surrogate is recomputed from the family and
  the placeholder.

The surrogate pools in `data/surrogates/` are committed and reviewable. Matches
made entirely of pool names are not flagged again, so a real person who shares a
pool name would not be flagged either. Keeping the pools small, generic and
public makes that tradeoff visible.

## Detection records

A `Detection` records where and what kind of match was found without storing the
matched text:

| Field | Content |
| --- | --- |
| `rule_id`, `kind`, `layer`, `severity` | Which rule matched |
| `field_path`, `start`, `end` | Where it matched |
| `matched_len`, `matched_sha256_8` | Length and the first 8 hex characters of the SHA-256 of the match |
| `slot`, `placeholder` | Replacement information |
| `excerpt` | Up to 24 characters of context on each side, with every detected span masked |

Excerpts mask all detected spans in the window, not only the reported one, so a
neighboring match is never reprinted:

```text
…due <10 chars redacted>. Call <12 chars redacted>.
```

Logs follow the same rule. `redact()` in `logging_utils.py` renders text as
`<redacted len=N sha256=…>` and raises an error if a caller asks for a preview
without `allow_preview=True`. Log statements record identifiers, counts, statuses
and timings. Candidate text is read in review packets, which are written to the
git-ignored `staging/` zone.

## Private-fact signals

Removing PII does not make an example safe. A sentence can identify a person
through the situation it describes even after every name is replaced. The
fact assessment in `privacy/facts.py` produces signals, never redactions:

| Signal | Score | Detects |
| --- | ---: | --- |
| `fact.attributive_justification` | 0.9 | A decision justified by where someone works |
| `fact.employment_claim` | 0.9 | A specific employment relationship |
| `fact.document_recall` | 0.85 | Recalled contents of a personal document such as a resume |
| `fact.biographical_possessive` | 0.8 | A biographical relationship or attribute |
| `fact.session_recall` | 0.7 | A reference to a prior conversation the prompt does not contain |
| `fact.unsupported_entity` | 0.6 | A proper noun in the answer that never appears in the prompt |
| `fact.unsupported_number` | 0.5 | A specific number in the answer that never appears in the prompt |

The highest score decides the verdict:

| Highest score | Verdict | Effect at promotion gate G07 |
| --- | --- | --- |
| 0.75 or more | `fact_teaching` | Fails |
| 0.4 to 0.75 | `needs_fact_review` | Passes only if the decision record's `policy_not_facts` review hard gate is `PASS` |
| Below 0.4 | `policy_like` | Passes |

`fact.unsupported_entity` compares nouns asymmetrically. Nouns in the prompt
are collected broadly, including words at the start of a sentence. Nouns in the
answer are collected narrowly, skipping sentence-initial words and common
sentence starters. A proper noun that is named in the prompt is therefore not
reported as unsupported.

## False positives

A detector that fires on the pipeline's own synthetic output trains reviewers to
ignore warnings, so such a match is treated as a bug in the detector.
`tests/test_privacy.py` checks both directions: every rule fires on its positive
sample, and the generated corpus sanitizes clean with no fact risk
(`test_the_generated_corpus_sanitizes_clean` and
`test_the_generated_corpus_has_no_fact_risk`).

## Related documentation

- [PRIVACY.md](../PRIVACY.md): the privacy policy and its guarantees
- [staging.md](staging.md): the promotion gates that re-check privacy
- [review.md](review.md): the `no_private_data` and `policy_not_facts` review
  hard gates
- [CONTRIBUTING.md](../CONTRIBUTING.md#fixture-policy): rules for test fixtures
- [SECURITY.md](../SECURITY.md): credential handling
