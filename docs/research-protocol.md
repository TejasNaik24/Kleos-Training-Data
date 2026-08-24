# Research protocol

## The claim this repository has to support

> Fine-tuning on KLEOS-derived examples teaches a **generalizable decision
> policy**, not memorized facts about the people in the training data.

Everything below exists so that claim can be defended, or honestly abandoned,
rather than assumed.

## Why the target is computed

A registered policy resolves each situation, and the answer is rendered from
that. Three consequences:

**No self-reinforcement.** The target is never what KLEOS said. A model's own
output becoming its next training target is how a mistake gets amplified into
policy. The review rubric asks the question directly: *is this answer actually
good, or merely what the current system happened to say?* — and the pipeline
answers it structurally by never using system output as a target at all.

**Perturbations are checkable.** Because the answer is computed, the validator
asserts that a paraphrase or reorder yields the *same* decision. A perturbation
that changes the answer is not a perturbation; it is a different scenario, and
including it silently would make consistency testing measure noise.

**The rationale cannot drift.** The stated reason comes from the same computation
as the ranking, so an answer cannot be right for a reason it did not use.

## Coverage, not count

```bash
python scripts/coverage_report.py --release <dir>
```

Reports joint structure — task × domain, task × urgency, task × evidence_quality,
and so on — and names the cells that are empty or thin.

Bad: *"25,000 examples."*
Good: *"25,000 examples; 7 tasks, 8 domains, 4 urgency levels, 5 evidence-quality
levels, 3 context-length buckets; 61% joint fill on task × urgency."*

A CRITICAL verdict means an axis is constant — it cannot support any claim about
that axis. The generator itself was caught manufacturing narrowness once:
`sampling: stratified` drew each axis independently, which is random sampling
wearing the word, and produced 83% of a "format-diverse" catalog in one format.

## Generalization is measured by what was held out

Group-aware splitting keeps a base example and its perturbations together.
Holdouts are **declared in the catalog before the split runs**, because an OOD
result you can only describe afterwards is a description of where a hash landed.

Registered shifts: `unseen_entities`, `unseen_domains`, `unseen_formats`,
`unseen_source_types`, `context_length_shift`, `reordered_evidence`,
`conflicting_evidence`.

The most informative is **format shift**. Train on bullets and prose, test on
JSON, a Slack thread, a GitHub issue. A model that learned the underlying policy
survives it; one that learned a presentation pattern does not.

## Consistency is what makes a policy claim falsifiable

Members of a group share a `scenario_family` and differ by a registered
`perturbation_kind`. A policy that is real should be stable across:

- a paraphrased question
- reordered evidence
- added irrelevant context
- a different context length

If ranking flips under reordering, the model learned position, not policy. This
is silently unmeasurable unless groups have two or more members — the coverage
report says so explicitly.

## The anti-claim

Every scenario states not only what it teaches but the surface rule a lazy model
might learn instead:

```yaml
policy_claim: >
  Rank by where delay costs the most: nearness of deadline, discounted by how
  well the claim is supported and by how much the outcome matters.
anti_claim: >
  Pick whichever item is listed first, or whichever has the nearest date
  regardless of whether anyone verified it.
```

The anti-claim is what the perturbations and holdouts are built to catch. **A
family that cannot state its anti-claim has usually not identified what it
tests.** The reviewer packet leads with both, so the question a human answers is
"does this teach that?" rather than "does this look fine?".

## Hard cases are not optional

A corpus of clean decisions teaches that a decision is always available. The
catalog deliberately includes:

- abstention — when evidence does not separate the options, say so and name what
  would resolve it
- reliability over recency — the inverse of "trust the newest record"
- relevance over recency — the inverse of "keep the newest context"
- scope boundaries — say when something relevant sits outside what was asked

## What would falsify the claim

Stated in advance, because a hypothesis you can only evaluate afterwards is not
one:

- OOD test performance collapsing to near in-distribution chance while
  in-distribution improves — the model learned the presentation
- ranking flipping under `evidence_order` or `context_order` perturbations — it
  learned position
- the model producing confident answers on the abstention family — it learned
  that an answer is always required
- unsupported entities appearing in generations — it memorized entities

## Reproducibility

Every release records the schema version, preprocessing version, privacy ruleset
version, review rubric version, split strategy and seed, scenario fingerprints,
and the pinned contract commit. `provenance.json` holds what the manifest cannot.

The build timestamp is deliberately **not** part of any content hash: content
determines identity.
