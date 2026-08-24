# Privacy engineering

`PRIVACY.md` states the policy. This is how it is implemented.

## Why four layers

Each layer catches something the one above it structurally cannot.

**1. Secrets — severity `block`.** API keys, tokens, JWTs, private key blocks,
session cookies, database URLs with passwords. **Never auto-redacted.** A secret
in a candidate means the capture path itself is compromised; quietly swapping the
string for a placeholder would produce a clean-looking candidate and hide that.
The candidate is rejected and the credential gets rotated.

**2. PII — severity `redact`.** Email, phone, addresses, government-style
identifiers, student and employee ids, URLs carrying tokens, home directory
paths, handles, UUIDs, absolute dates. Mechanically replaceable: an email is an
email wherever it appears.

**3. Entity vault — severity `redact`.** Operator-supplied literals that regex
cannot find, because they are identifying by reference rather than by shape — a
real employer whose name reads like an ordinary noun, an advisor's surname, a
project codename. Lives in `vault/`, chmod 700, git-ignored, never in a release.
Entries are added through stdin rather than argv so a real name never lands in
shell history.

**4. Structural heuristics — severity `review`.** Capitalized bigrams,
organization suffixes. Real false-positive rates, so they flag for a human rather
than deciding.

## Redact, then rehydrate

Detection produces `[[PERSON_1]]`. Those placeholders are **not** what gets
trained on.

Shipping them would be the obvious shortcut and it is wrong: training on
`[[PERSON_1]]` teaches a model to emit bracket tokens and destroys the naturalness
the task depends on — a prioritization example reads as one, or it teaches
nothing. Training on the real name teaches a private fact. A consistent fictional
surrogate does neither.

**Surrogates are keyed on `scenario_family`.** Within a family — and therefore
within a conversation and its equivalence group — a slot always resolves to the
same name, so "Dana" in turn one is "Dana" in turn three. Across families the
same real person maps to a *different* fictional person.

That last property is what actually defeats memorization. There is no persistent
pseudo-entity spanning the corpus to memorize. The alternatives both fail: one
global name per slot creates exactly the memorable fake person the scheme exists
to prevent; re-randomizing per example destroys coreference and makes every
conversation incoherent.

Structured slots keep their shape — an email becomes an email, a handle becomes a
handle — because a phone number replaced by a person's name is nonsense.

### The consequence that took two attempts

A redacted `@dana` becomes `@rowan.baxter`, which **is still a handle**, so the
detector flags it again on the next scan. That is not a defect in the surrogate;
it is the surrogate working. The fix is that surrogate-shaped rules consult the
known-name set: a match whose components are all names this repository generated
is not personal data.

The accepted tradeoff is that a real person sharing a pool name would be
allowlisted. The pools are committed, deliberately generic, and reviewable
precisely so that tradeoff is visible rather than hidden.

## Detections never carry what they matched

A `Detection` holds a digest, a length, a location and a redacted excerpt. Never
the value.

This matters because detections are written to `<id>.privacy.json`, printed in
reports, and pasted into issues. A record that quoted the secret it found would
relocate the problem into files that outlive the candidate.

**Excerpts mask every detected span, not only their own.** Masking just the
reported span leaves the context window around it reprinting the next hit:

```
"…due <10 chars redacted>. Call 614-555-9876."
```

Each detection looks individually correct and the set of them leaks everything.
This was fixed twice — once in the detection layer, and again in the private-fact
layer, where it was worse, because a fact signal's entire purpose is to name a
suspected private fact.

## Private facts

PII is a **string** problem with a mechanical fix. A private fact is a
**semantic** problem with none:

> Prioritize the Motorola project — your internship there ends in three weeks
> and your manager already flagged the deadline.

Strip every name and it still teaches that a specific person had a specific
internship on a specific timeline. No substitution repairs it, because the
*situation* is the identifying thing.

So `facts.py` produces **signals, never redactions**. The decision belongs to
review, as the `policy_not_facts` hard gate. That division is deliberate: a
heuristic confident enough to auto-reject would be confident enough to
auto-approve, and neither is warranted.

The strongest signal is `fact.unsupported_entity` — a proper noun asserted in an
assistant turn that appears nowhere in the prompt. That is a model stating
something it was not told, which is either a hallucination or a memory, and both
disqualify.

### Permissive about support, conservative about accusation

Collecting the prompt's nouns is **permissive** (includes sentence-initial words);
collecting the assistant's asserted nouns is **conservative** (excludes them).

The asymmetry is deliberate. Anything missing from the prompt set becomes a false
accusation, so being too generous there costs a missed signal; being too strict
costs a reviewer who learns to ignore the strongest signal we have. An earlier
version skipped sentence-initial words on *both* sides, so a prose prompt reading
"Silverbrook is due in 5 days" never registered the name and the assistant
mentioning it looked like invention — twelve false accusations on entirely
synthetic data.

## False positives are defects

A reviewer taught to click through warnings will click through the real one. When
a detector fires on our own synthetic corpus, that is a bug in the detector.

`tests/test_privacy.py` asserts both directions: every rule fires on its positive
fixture, **and** the generated corpus sanitizes completely clean.

## Fixture policy

Everything committed under `data/` is obviously fake: `Alice Example`,
`example.invalid`, `555-0100`. Never commit a real person's information to test
privacy scanning.

Fixtures that must genuinely *match* a secret pattern live inline in the test
module that asserts the match, and that module is in the scanner's `SELF_EXEMPT`.
Keeping them there is what lets the committed fixtures stay strictly
non-matching.
