# `kleos-policy-v0.0.7`: policy-derived reasoning traces and complete labels

**Status:** design approved section by section on 2026-10-01; awaiting review of this
written spec. Nothing is implemented, and nothing here is committed.

**Sub-project 1 of 3 for KLEOS Logos v0.0.2.**

1. **This dataset release.**
2. Training and evaluating Logos v0.0.2 in `Kleos-Models`: reading the new field,
   reasoning-aware evaluation, Kaggle 2×T4 training and the H9 pre-registration.
3. Deployment.

Each sub-project gets its own spec and plan.

## Why

- **H8 found no measurable gain from a stronger base on v0.0.6.** KLEOS Logos v0.0.1
  (Ministral 3 14B) was not measurably better than Hermes (Mistral-Nemo 12B). Logos −
  Hermes was −0.0168 on the answerable items, cluster 95% CI −0.0546 to +0.0177. Three
  Mistral bases from 8B to 14B finished within 0.016 of each other.
- **Logos v0.0.2 changes two things.**
  - It uses `mistralai/Ministral-3-14B-Reasoning-2512`, which writes `[THINK]…[/THINK]`
    before its answer.
  - It trains on data that teaches it to reason about KLEOS decisions.
- **The reasoning can be taken from the policy itself.** The generator already computes
  every decision from an explicit policy, with scores, gaps and thresholds, and then
  discards those intermediate values. Rendering them gives reasoning that is faithful by
  construction, costs nothing, and is checkable line by line.

## Scope

**In scope:**

- A `reasoning` trace on every train and validation answer ("shows its math").
- R1 of the repair spec: a "What decided it: <label>" line on every train and
  validation answer, declines included.

**Out of scope for this release:**

- R2: contrastive pairs that de-confound conditional abstention.
- R3: axis hygiene.
- R4: a separate hard set.
- R5 beyond the composition report below.
- `impact` examples for `brief.what_needs_a_decision` and `notif.deadline_vs_evidence`.

Each of these needs new scenarios. The repair spec lives at
`Kleos-Models/docs/datasets/kleos-policy-v0.0.7-repair-spec.md`.

## Invariants (the build fails if any breaks)

1. **The test file is unchanged.** `test.jsonl` is byte-identical to v0.0.6, sha256
   `a4decaaf029b227366846d44018b0b1b669d21f068800a9f3175e15ec4783980`. Kleos-Models'
   benchmark therefore rebuilds to sha256
   `a11ffad75f5147f9d0ddad7bad4bfc073dc730df2b19173ff642774233b4b266`.
2. **Every decision is unchanged.** For all 1,350 examples, `Decision.comparable()`
   (ranking, deciding factor, abstained) equals v0.0.6's.
   - This keeps the known factor-direction quirk. In 48 examples (17 of them in test) the
     named factor is the largest raw gap between the top two, even when that gap favours
     the runner-up.
   - The test targets follow the quirk and are frozen, so training must teach the same
     rule.
   - Fixing it needs a new test split, which belongs to a later release.
3. **Split membership is unchanged.** Same seed, same `format_holdout` on `json`, same
   group assignment.
4. **v0.0.6 and the JSON path are untouched.** The JSON prompt renderer, the JSON answer
   renderer, the system prompts, the scenario YAMLs and the surrogate pools do not
   change.

## Design

### 1. Contract: an optional `reasoning` field

- **The field.** Assistant messages gain an optional `reasoning: str`.
  - It is written to JSONL **only when present**. The writer dumps with
    `exclude_none=False`, so the field needs an explicit omit-when-absent serializer.
    Otherwise every message, test included, would gain `"reasoning": null` and break
    invariant 1.
- **Schema versions.**
  - An example that carries reasoning is schema version `1.1`.
  - An example without it stays `1.0`, byte-identical to v0.0.6.
  - A release may contain both.
- **The field is restricted.**
  - It is allowed only on `role: assistant`.
  - It must be non-empty when present.
  - It is never allowed on an example whose `variation_axes.format` is `json`.
- **Documentation and consumers.** `DATASET_CONTRACT.md` documents the field. Consumers
  read it like this:
  - Kleos-Models passes it to the chat template's `reasoning` key, which renders
    `[THINK]…[/THINK]`.
  - A non-reasoning model ignores it.
  - The Kleos-Models reader change belongs to sub-project 2.
- **The pipeline leaves the field alone.** Reasoning never appears inside `content`, so
  the existing `<think>` stripping in `staging/normalize.py` and in the contract's
  `strip_reasoning_spans` does not touch it. A test checks this.

### 2. Where the trace comes from

- **Policies return a trace.** Each policy returns, besides its `Decision`, a structured
  trace record of the values it actually used.
  - Score-based policies record per-item inputs, their weights and the product.
  - Every policy records each threshold it tested and the result.
  - Every policy records the gap computation behind the deciding factor.
- **The trace stays outside the decision comparison.** It travels on the decision in a
  field that `comparable()` excludes, so the check that perturbations keep the decision
  stays exactly as it is.
- **Rendering.** A new module, `scenarios/reasoning.py`, renders the trace to text.
  `generator._messages` attaches it as `reasoning` on the assistant message, for non-JSON
  formats only.

### 3. What a trace says

Every trace follows the same five steps, about 150 words, hard cap 1,200 characters.

1. **The rule:** one line saying what this family's policy weighs. For example: "Score
   each item as deadline × evidence × impact."
2. **One line per item, in prompt order:** the prompt's wording mapped to the policy's
   values. For example: "Redpine: due in 2 days (0.33), reported (0.55), high (1.0) →
   0.18". Deadline score is `1/(1+days)`; the evidence and impact weights are those in
   `situations.py`.
3. **The checks the policy applied**, each with its outcome:
   - close call: relative separation below 0.15
   - stale explicit statement: 30 days or older
   - usable evidence: weight 0.55 or above
   - workspace scope
   - strong challenger: weight 0.8 or above
   - ambiguous request
4. **Why this factor:**
   - For gap-based policies: "largest gap between the top two: impact (1.0 vs 0.3)". That
     is exactly what `_deciding_factor` computes, quirk cases included.
   - Other policies state their own rule. For example: "only one item is inside the
     workspace, so scope decides."
5. **Conclusion:**
   - Committed: the ranking, for example "So: Redpine, then Pennfold, then Fieldstone."
   - Declined: the failed check and the resolver, for example "Ask: is Ashgrove still
     right?".

**Policy coverage.** All 12 policies in `POLICIES` get a trace that shows only the values
the policy uses:

| Policy | Shows |
| --- | --- |
| `rank_by_deadline_then_evidence` | scores; largest-gap factor |
| `rank_or_abstain_when_close` | scores; separation against 0.15; evidence of the top two against 0.8; `insufficient_separation` |
| `rank_by_reliability_over_recency` | evidence, then age; evidence or recency factor |
| `resolve_or_abstain_on_support` | evidence gap against 0.15; ages; `insufficient_separation` |
| `select_by_evidence_need` | evidence × impact; staleness; evidence or impact factor |
| `rank_by_relevance_over_recency` | impact, then age |
| `respect_workspace_scope` | in or out of scope, then score; `scope` |
| `defer_to_explicit_statement` | explicit leader; challengers at 0.8 or above that are newer; leader age against 30 days; `stale_explicit_conflict` |
| `verify_when_evidence_weak` | usable set at 0.55 or above; `missing_input` |
| `ask_when_request_ambiguous` | the request's ambiguity, as worded in the prompt; `request_ambiguous` |
| `ask_before_crossing_workspace` | adequacy inside against outside; `ask_before_crossing` |
| `prefer_least_privilege_source` | adequacy, then narrowest reach, then evidence; `least_privilege` or evidence |

**Wording rules, so G05, G07 and the privacy scanner pass:**

- **Capitals:** only item names from the prompt are capitalised, and never two
  capitalised words in a row. This avoids `structural.person_name.v1` and
  `fact.unsupported_entity`.
- **Numbers:** at most 2 decimal places, and whole-number percentages. This avoids
  `pii.postal_code.v1`.
- **Banned forms:** no dates, no "@", and no comma-grouped numbers.

**Variety.** Each step has 2–3 phrasings, picked by `_deterministic_index(example key,
step)`. Traces are not identical boilerplate, and a rebuild reproduces them byte for
byte.

### 4. Answer changes (R1)

- **Every train and validation answer ends with `What decided it: <label>.`** This
  covers:
  - bullets, prose, and the fallback formats (`slack_thread`, `github_issue`,
    `calendar`);
  - declined answers, which today never print the label.

  The line is in the form `Kleos-Models/src/kleos_models/evaluation/graders.py`
  `_DECIDED_BY` reads.
- **Nothing else in the answer changes.** The per-item rationale text and the resolver
  text are as in v0.0.6.
- **Expected decline-label counts outside test** (train and validation combined,
  measured on 2026-10-01 by regenerating the catalog in memory):

  | Label | Count |
  | --- | --: |
  | `request_ambiguous` | 60 |
  | `stale_explicit_conflict` | 44 |
  | `missing_input` | 30 |
  | `insufficient_separation` | 26 |
  | `ask_before_crossing` | 100 |

  The target is at least 20 train and 5 validation examples per label. The build reports
  any label below it and does not add scenarios.

### 5. Building and sealing

- **Where it builds.**
  - It builds in a fresh workspace (`--workspace` or `KLEOS_TRAINING_DATA_ROOT`), so
    v0.0.6's promoted pool is not mixed in and G12 does not see the unchanged JSON
    examples as duplicates.
  - It uses the existing chain (`make slice`, `SLICE_VERSION=kleos-policy-v0.0.7`):
    capture, normalize, sanitize, review, the 14 promotion gates, build, the 32
    `verify_release` checks, coverage and compatibility.
- **Who does what.** I build and verify in a scratch workspace. The user runs the final
  seal into `releases/` and commits.
- **Composition report.** Each split gets counts by task, family, label, decision and
  format, plus the trace length distribution.

### 6. Acceptance checks (all automated; any failure stops the build)

1. `test.jsonl` sha256 is `a4decaaf…3980`, and the benchmark rebuilds to `a11ffad7…b266`.
2. Every example's `comparable()` equals v0.0.6's.
   - **The baseline:** before any code changes, a test fixture records every candidate's
     `comparable()` from the current generator, keyed by `generator._prompt_key`.
   - **Why the current generator is a valid baseline:** it reproduces v0.0.6. All 1,350
     regenerated ids match the release, verified 2026-10-01.
   - **The check:** the new generator must reproduce the fixture exactly.
3. An independent re-computation of each trace's numbers and conclusion from the
   situation agrees with the decision.
4. On 100% of train and validation answers, Kleos-Models' `extract_deciding_factor`
   returns the decision's factor.
5. All 14 gates pass with 0 rejections, the 32 release checks pass, and the privacy scan
   is clean.
6. Every trace is at or under 1,200 characters.
7. The decline-label counts per split are reported.

### 7. Errors

- **Fail fast.** Any acceptance failure stops the build with a message that names the
  example id and the check.
- **No partial release.** Sealing is already atomic: the release is staged, then moved
  into place with `os.replace`.

### 8. Testing (tests first)

- A unit test for each policy's trace record and rendered text.
- A serialization test: an example without reasoning serializes byte-identically to its
  v0.0.6 line.
- Contract tests for the field: assistant only, non-empty, forbidden on JSON-format
  examples, schema `1.1`.
- The normalize and sanitize stages leave `reasoning` intact.
- Faithfulness tests: re-computation equals the trace, for the whole catalog.
- Wording-rule tests: the repo's own privacy rules raise 0 findings on every trace.
- Updates to the rendering-coupled tests:
  - `tests/test_scenarios.py:120-160`
  - `tests/test_framing_and_content.py:259-268` and `:315-336`
  - `tests/test_repaired_policies.py:167-201`
- Full suite and `make check`.

## Downstream (sub-project 2, recorded here so the field is designed for it)

- **The reader.** Kleos-Models' `Message` schema and formatter must accept `reasoning`,
  pass it to the Reasoning model's template, and supervise the `[THINK]…[/THINK]` tokens.
  `[THINK]` is token id 34 and `[/THINK]` is 35, both special in that model's tokenizer.
- **Evaluation must split thinking from the answer by token id, before grading.**
  - The graders' fallbacks would otherwise read the trace: the first numbered line, the
    first label mentioned, and abstention markers such as "let me".
  - Answers cut off mid-thought are failures.
  - The reasoning is kept in the results.
- **Sequence length.** A trace adds about 200–300 tokens. The longest sequence exceeds
  what a single free T4 fits (about 512 tokens for this model), which is why training
  moves to Kaggle 2×T4.

## Risks

- **No guarantee of a better model.** Policy-derived traces may not improve the model's
  decisions on the answerable items, which is H9's primary population. H9 will be
  pre-registered before training and reported whatever its result.
- **The quirk is taught on purpose.** Traces that describe the factor quirk faithfully
  make an imperfect labelling rule explicit. This is a known, recorded limitation, not a
  silent one.
- **Trace style may not transfer to JSON answers.** The test answers are JSON-format,
  and training has none (the format holdout). Whether trace-plus-prose training
  transfers to trace-plus-JSON answers is part of what H9 measures.
