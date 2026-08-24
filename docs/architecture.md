# Architecture

## Why three zones

A reviewer must be able to tell, from the filesystem alone, whether an artifact
is raw and potentially sensitive, reviewed but not yet promoted, or trusted final
training data. Directory structure is the cheapest form of that guarantee, and it
survives people who have not read the docs.

```
scenarios/          declared situations, committed, synthetic
    ↓
staging/            UNVERIFIED → approved.  Never committed.
    ↓
releases/           IMMUTABLE product.  Shipped by path, never committed.

vault/              re-identification keys.  chmod 700.  Never committed.
reports/            may quote candidate text.  Never committed.
```

All four are deny-by-default in `.gitignore`, with only a top-level `.gitkeep`
tracked, and `tests/test_gitignore.py` proves it against real `git` — including
the decisive check of what `git add -A` actually stages.

## The pipeline

```
scenario  ──generate──▶  Candidate      policy computes the target
    │
    ├──capture──▶  RawCapture           untrusted, lane-stamped
    │                  │
    │              normalize            line endings, reasoning spans
    │                  ▼
    └───────────▶  NormalizedCandidate  contract-shaped, provisional id
                       │
                   sanitize             4 layers, redact → surrogate
                       ▼
                   SanitizedCandidate + privacy record
                       │
                   review               machine opinion, human decision
                       ▼
                   HumanDecision        signed against exact bytes
                       │
                   promote              14 gates, 13 mandatory
                       ▼
                   PromotedExample      audit block stays in staging
                       │
                   build_release        dedup → leakage → split → seal
                       ▼
                   releases/<version>/  manifest + train.jsonl [+ val/test]
```

## Load-bearing decisions

**The training target is computed, not authored.** A registered policy resolves
each situation, and the answer is rendered from that. Three things follow: the
dataset never trains on a model's own output; a perturbation can be *checked* to
preserve the decision; and the stated reason cannot drift from the actual reason.

**The contract is mirrored, not imported.** The public repo's docs say "neither
repository imports the other". So `contract/` is a faithful port, pinned to a
commit, with differential tests asserting our JSONL bytes, split assignments and
validation decisions are identical to the real thing. The mirror lets this
repository build with kleos-models absent; the tests stop it drifting.

**Rules are properties, not conventions.** A `ReviewVerdict` with a failing gate
and an approval cannot be constructed. A `PromotionPolicy` bypassing a mandatory
gate cannot be constructed. Neither is a check somebody remembers to call.

**Identity is content-derived.** An example's id is a function of its task,
messages and axes — not its position, not its provenance. Re-reviewing does not
change it; editing one character does.

**Everything runs offline.** No stage needs a credential to be exercised. A
pipeline testable only against a live backend is one whose first end-to-end run
creates private data before any gate exists to catch it.

## Module map

| Package | Owns |
| --- | --- |
| `contract/` | The mirrored public contract: schemas, writer, splitter, dedup, the pin, the handshake |
| `scenarios/` | Situations, policies, rendering, generation, surrogate pools |
| `collection/` | Adapters, transport, the production guard |
| `staging/` | Record formats, atomic integrity-checked storage, normalization |
| `privacy/` | Four detection layers, redaction, surrogates, private-fact assessment |
| `review/` | Rubric, packets, machine reviewers, signed decisions |
| `promotion/` | The gate table, the policy, the runner |
| `datasets/` | Holdouts, manifests, the sealed release, verification |
| `quality/` | Coverage reporting |

## Relationship to kleos-models

One directory, consumed by path:

```bash
python <kleos-models>/scripts/train.py --config <config> --dataset <release-dir>
```

No code crosses the boundary in either direction at runtime.
