# Compatibility with kleos-models

## Why a mirror rather than a dependency

The public repo's `docs/privacy.md` states: *"Neither repository imports the
other."* So `src/kleos_training_data/contract/` is a faithful port of the public
models, writer, splitter and duplicate detection.

The obvious objection is that a mirror drifts. It does — on day ninety, not day
one. The answer is not to couple the repositories but to **test** the mirror.
Coupling would prevent drift by making this repository unbuildable whenever the
public one is mid-refactor. Testing catches drift without that cost, and gives a
named failure instead of a silent divergence.

## The pin

```python
CONTRACT_SOURCE_COMMIT = "12361d53cf329c897ec14daede6d14112a3e8f20"
```

A commit, never a branch or tag. A moving reference would let the public contract
change underneath a release without any test failing — the one failure mode this
whole arrangement exists to prevent.

## The handshake

```bash
make compat                                  # --strict
python scripts/check_contract_compat.py --release <dir> --strict
```

Three tiers, in increasing cost:

1. **Vocabulary** — every mirrored constant, element-wise *and* order-wise. The
   early-warning detector: a newly registered task fails here long before any
   byte-level test notices, because the byte tests only exercise values already
   in use. Order matters even where it looks cosmetic, since it decides how
   `holdout_values` sorts.
2. **Behaviour** — `stable_rank` over 2,500 comparisons, `normalize_text` over a
   unicode/punctuation/digit torture corpus, and JSONL writer bytes.
3. **Artifact** — a real release round-tripped through the public `load_dataset_bundle`
   and `validate_examples`, plus a manifest content-hash agreement check.

Answers one word: **COMPATIBLE** or **INCOMPATIBLE**. It never silently
continues.

## A skip is not a pass

Without `--strict`, an absent kleos-models reports `SKIPPED` and exits 0. That is
the normal local state and the correct one: the offline pipeline is meant to work
without the public repo checked out.

CI passes `--strict`, where absence is a **failure**. A compatibility check that
passes because it could not run produces a green build that means nothing, which
is worse than having no check.

## Updating the pin

1. Read the public repo's diff between the old and new commit.
2. Update the mirror.
3. Run **every** differential suite — `pytest -m requires_kleos_models` — not
   only the one you think was affected.
4. Update `CONTRACT_SOURCE_COMMIT` and `CONTRACT_VERIFIED_AT`.
5. Record the decision here.

**Never move the pin to make a failing check pass.**

## Deliberate strictness deltas

We are never *looser* than the public repo. Where we differ, we are stricter, and
the direction is asserted rather than assumed.

| Delta | Why |
| --- | --- |
| Only canonical split filenames | `validate_dataset.py` accepts `synthetic_train.jsonl`; `train.py` does not. An alias validates and then fails to train. |
| `G13` compares twice | `conversation_text()` is asymmetric between training and evaluation examples, weakening near-duplicate recall exactly where it matters. |
| Metadata extras allowlisted | `ExampleMetadata` is `extra="allow"` upstream, so anything there ships inside `train.jsonl`. |
| Only `quality_status: "reviewed"` promotes | The public loader silently drops anything else, producing a release that validates and trains on nothing. |

## Known upstream issues, not reproduced

Reproducing a bug for fidelity's sake would be the wrong kind of faithful.

- **`check_no_private_data.py:234-239` reprints short lines containing the secret
  it found.** It truncates the matched span to six characters but then prints
  `line.strip()[:60]`, so any line under 60 characters echoes the whole
  credential to stdout and from there into a CI log. Our port excises the matched
  span instead. Worth a one-line fix upstream.
- **`pyproject.toml:69-70` declares a `kleos` console script pointing at a
  non-existent module.** Not mirrored.
- **`split_dataset.py` defaults to 0.7/0.15/0.15 while `SplitConfig` defaults to
  0.8/0.1/0.1.** A config-driven and a CLI-driven split of the same data disagree
  unless fractions are stated. We always state them and record them in the
  manifest.

## Current status

| | |
| --- | --- |
| Pinned commit | `12361d53` |
| Verified | 2026-08-24 |
| Checks | 19, all agreeing |
| Public validator on our release | passes, 0 errors |
