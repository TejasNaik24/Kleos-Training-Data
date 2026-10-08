# kleos-policy-v0.0.7: reasoning traces and complete labels

Design record for dataset release `kleos-policy-v0.0.7`, sealed on 2026-10-01.
The release adds a policy-derived reasoning trace and a deciding-factor line to
every training and validation answer, and keeps the test split byte-identical to
`kleos-policy-v0.0.6` so evaluations stay comparable.

## Contents

- [Motivation](#motivation)
- [Scope](#scope)
- [Invariants](#invariants)
- [The reasoning field](#the-reasoning-field)
- [Trace content](#trace-content)
- [Answer changes](#answer-changes)
- [Acceptance checks](#acceptance-checks)
- [Results](#results)
- [Compatibility](#compatibility)
- [Downstream use](#downstream-use)
- [Risks](#risks)

## Motivation

- In kleos-models, KLEOS Logos v0.0.1 (Ministral 3 14B) was fine-tuned on
  v0.0.6 and compared with KLEOS Hermes (Mistral-Nemo 12B). Fine-tuning helped
  Logos, but the fine-tuned Logos was not measurably better than Hermes (H8 in the
  kleos-models
  [experiments document](https://github.com/TejasNaik24/Kleos-Models/blob/main/docs/experiments.md)).
- The next Logos model uses `mistralai/Ministral-3-14B-Reasoning-2512`, which
  writes `[THINK]…[/THINK]` before its answer, so its training data needs
  reasoning to supervise.
- The generator already computes every decision from an explicit policy, with
  scores, gaps and thresholds, and then discards those values. Rendering them
  produces reasoning that matches the decision by construction and can be checked
  line by line.
- In v0.0.6, decline labels such as `missing_input` were printed only in
  JSON-format answers, and the format holdout put every JSON-format example in the
  test split. 78 of the 349 test cases expected a label that never appeared in
  training.

## Scope

Included:

- A `reasoning` trace on every training and validation answer.
- A `What decided it: <label>.` line on every training and validation answer,
  including declined answers.

Not included, because each needs new scenarios (see the kleos-models
[repair spec](https://github.com/TejasNaik24/Kleos-Models/blob/main/docs/datasets/kleos-policy-v0.0.7-repair-spec.md)):

- Contrastive pairs that separate conditional abstention from family membership
- Axis hygiene
- A separate hard set
- `impact`-decided examples for `brief.what_needs_a_decision` and
  `notif.deadline_vs_evidence`

## Invariants

| Invariant | Enforced by |
| --- | --- |
| `test.jsonl` is byte-identical to v0.0.6 (sha256 `a4decaaf…3980`), so the kleos-models benchmark built from it is unchanged | `scripts/check_reasoning_release.py` |
| Every decision is unchanged: `Decision.comparable()` (ranking, deciding factor, abstained) equals v0.0.6 for all 1,350 examples | `tests/test_decision_baseline.py` with `tests/fixtures/v006_decisions.json` |
| Split membership is unchanged: seed 42, format holdout on `json`, same groups | The unchanged catalog and split code |
| The JSON path is untouched: JSON prompt and answer rendering, system prompts, scenario YAMLs and surrogate pools | `tests/test_decision_baseline.py` checks that every JSON example keeps its v0.0.6 ID |

Keeping decisions unchanged also keeps a known labeling quirk. In 48 examples
(17 of them in the test split), the deciding factor is the largest raw gap
between the top two items even when that gap favors the runner-up. The test
targets follow this rule and are frozen, so changing it needs a new test split
in a later release.

## The reasoning field

Assistant messages gain an optional `reasoning` string, specified in
[DATASET_CONTRACT.md](../../DATASET_CONTRACT.md#example-schema):

- It is allowed only on assistant messages, must be non-empty, and never appears
  on a JSON-format example.
- It is written only when present. An example without it serializes exactly as it
  did in v0.0.6.
- An example that carries it has schema version `1.1`. Other examples stay `1.0`.
- It is part of the content hashed into the example ID.
- Every privacy and fact check scans it like `content`.

Policies record the values they used in a `Trace` attached to the `Decision`.
The trace is excluded from `Decision.comparable()`, so the perturbation checks
are unaffected. `src/kleos_training_data/scenarios/reasoning.py` renders the
trace as text.

## Trace content

Each trace follows five steps and is capped at 1,200 characters:

1. The rule: what the family's policy weighs, for example "Score each item as
   deadline × evidence × impact."
2. One line per item, in prompt order, mapping the prompt's wording to the
   policy's values, for example "Redpine: due in 2 days (0.33), reported (0.55),
   high (1.0) → 0.18".
3. Each check the policy applied and its outcome: close call (relative
   separation below 0.15), stale explicit statement (30 days or older), usable
   evidence (weight 0.55 or above), workspace scope, strong challenger (weight 0.8
   or above) and ambiguous request.
4. Why the deciding factor decided it.
5. The conclusion: the ranking for a committed answer, or the failed check and
   the question to ask for a declined one.

Every one of the 12 policies produces a trace that shows only the values it
uses. Traces follow wording rules so they pass the privacy and fact checks: only
item names are capitalized and never two capitalized words in a row, numbers use
at most two decimal places, and no dates, `@` signs or comma-grouped numbers
appear. Each step has two or three phrasings, chosen deterministically, so a
rebuild reproduces every trace.

## Answer changes

Every training and validation answer now ends with a `What decided it: <label>.`
line, including declined answers, which did not state a label in v0.0.6. The
per-item rationale and the resolver text are unchanged.

Decline labels in the sealed release:

| Label | Train | Validation |
| --- | ---: | ---: |
| `request_ambiguous` | 45 | 15 |
| `stale_explicit_conflict` | 32 | 12 |
| `missing_input` | 25 | 5 |
| `insufficient_separation` | 22 | 4 |

`ask_before_crossing` appears in 100 training and validation answers combined. It
does not occur in the test split, because its only family has no JSON-format
examples.

## Acceptance checks

The release is built through the standard pipeline (capture, normalization,
sanitization, review, the 14 promotion gates, sealing and the 32 release checks)
in a fresh workspace, so v0.0.6's promoted pool cannot mark the unchanged
JSON-format examples as duplicates.

`scripts/check_reasoning_release.py` then checks the sealed release:

| Check | Outcome on failure |
| --- | --- |
| `test.jsonl` matches the reference test file byte for byte | Problem |
| Every non-JSON answer has a reasoning trace and no JSON answer has one | Problem |
| No trace is longer than 1,200 characters | Problem |
| Schema version is `1.1` exactly when reasoning is present | Problem |
| Every training and validation answer has a deciding-factor line | Problem |
| Each decline label has at least 20 training and 5 validation examples | Warning |

The test suite covers the rest: every policy produces a well-formed trace
(`tests/test_policy_traces.py`); traces never rank a lower score above a higher
one, call equal printed scores a tie, and never say a factor that favors the
runner-up decided the answer (`tests/test_reasoning.py`); the field passes through
normalization, sanitization, the fact checks and review packets
(`tests/test_reasoning_pipeline.py`); and the contract rules hold
(`tests/test_reasoning_contract.py`).

## Results

| Measure | Value |
| --- | --- |
| Examples | 1,350 (820 train, 181 validation, 349 test) |
| Content hash | `b53afa4216bf6973…` |
| Traces | 1,001, one on every training and validation answer |
| Trace length | Median 560 characters, maximum 878 |
| Schema versions | `1.1` on all training and validation examples, `1.0` on all test examples |
| `verify_release.py --strict` | 32 checks passed |
| `check_reasoning_release.py` | 0 problems, 1 warning: `insufficient_separation` has 4 validation examples against a target of 5 |

## Compatibility

kleos-models at the pinned contract commit rejects unknown message keys, so
`check_contract_compat.py --release` fails its loader check on v0.0.7 until
kleos-models accepts `reasoning`. A benchmark built from the test split is
unaffected, because `test.jsonl` carries no `reasoning`. See
[docs/compatibility.md](../compatibility.md#pending-upstream-change).

## Downstream use

Training on this field happens in kleos-models. The design assumes it will:

- accept `reasoning` in its message schema and pass it to the reasoning model's
  chat template, which renders it as `[THINK]…[/THINK]`;
- separate the thinking from the answer before grading, so graders never read
  the trace;
- allow for the extra 200 to 300 tokens a trace adds per example.

## Risks

- Policy-derived traces may not improve decisions on the answerable test items.
  kleos-models plans to pre-register that question as H9 before training.
- The traces describe the factor quirk faithfully, which makes an imperfect
  labeling rule explicit to the model.
- The test answers are JSON-format and training contains none. Whether training
  on traces with prose and bullet answers transfers to JSON answers is part of
  what the next evaluation measures.
