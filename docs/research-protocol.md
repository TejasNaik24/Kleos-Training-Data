# Research protocol

This document states the research claim the datasets are built to test, how the
pipeline is designed to make that test meaningful, and what the first training
results show. The evaluation hypotheses themselves are pre-registered in the
kleos-models
[experiments document](https://github.com/TejasNaik24/Kleos-Models/blob/main/docs/experiments.md).

## Contents

- [Hypothesis](#hypothesis)
- [Computed targets](#computed-targets)
- [Coverage](#coverage)
- [Out-of-distribution evaluation](#out-of-distribution-evaluation)
- [Perturbation groups](#perturbation-groups)
- [Anti-claims](#anti-claims)
- [Hard cases](#hard-cases)
- [Falsification criteria](#falsification-criteria)
- [Status after the first training runs](#status-after-the-first-training-runs)
- [Reproducibility](#reproducibility)

## Hypothesis

> Fine-tuning on KLEOS policy data teaches a generalizable decision policy, not
> memorized facts about the people in the training data.

The rest of this document describes how the datasets are constructed so that
this claim can be tested, and rejected if the evidence goes against it.

## Computed targets

Every training target is computed by a registered policy from the situation in
the prompt. See [scenarios.md](scenarios.md#decision-policies).

| Property | Consequence |
| --- | --- |
| No target comes from a model's output | A model's mistakes cannot become its next training targets |
| Perturbations are checked | Generation fails if a paraphrase, reordering or added noise changes the decision, so consistency tests compare like with like |
| The rationale comes from the same computation as the ranking | An answer cannot state a deciding factor that did not decide it |

## Coverage

The coverage report describes a dataset's structure:

```bash
python scripts/coverage_report.py --release releases/kleos-policy-v0.0.6
```

The report covers each axis on its own and eight task-by-axis pairs: task by
domain, urgency, evidence quality, conflicting evidence, difficulty, ambiguity,
format and context length.

| Level | Axis verdict | Joint verdict |
| --- | --- | --- |
| `CRITICAL` | An axis has a single value across more than one example, so it cannot support any claim about that axis | Not used |
| `WARNING` | An axis has no values, or its most common value is at least five times as frequent as its rarest | Fewer than half of the possible cells are filled, or a cell has fewer than 3 examples (`--min-cell-count`) |

The report also raises `CRITICAL` when no perturbation group has two or more
members, because consistency cannot be measured without them, and summarizes
readiness for format, entity and domain holdouts. `--fail-on CRITICAL` or `--fail-on WARNING`
makes the command exit with code 3 at that severity.

For example, `kleos-policy-v0.0.6` contains 1,350 examples in 294 groups from 18
scenario families, covering 7 tasks and 4 domains (career 335, coursework 341,
projects 340, research 334). Prompt formats are bullets (473), prose (464), JSON
(349) and Slack-style threads (64).

## Out-of-distribution evaluation

Holdouts are declared in the scenario catalog before any split runs, as
described in [dataset-lifecycle.md](dataset-lifecycle.md#holdouts). The
registered shift kinds are `unseen_entities`, `unseen_domains`,
`unseen_formats`, `unseen_source_types`, `context_length_shift`,
`reordered_evidence` and `conflicting_evidence`.

Every release so far measures format shift. The test split contains every
JSON-format prompt, and training and validation contain only bullets, prose and
Slack-style threads. A model that learned the decision policy should make the
same decisions when the input format changes, while a model that learned a
presentation pattern should not. Entity and domain holdouts are declared in the
catalog but not yet generated.

## Perturbation groups

A perturbation group is a base example and its perturbations. Members share a
`scenario_family` and `group_id`, differ by `perturbation_kind`, and always land
in the same split. A learned policy should give the same decision under:

- a paraphrased question
- reordered evidence or context
- added irrelevant context
- a shorter or longer prompt

If a ranking changes when only the order of the evidence changes, the model
learned position instead of policy.

## Anti-claims

Each scenario family states what it teaches and the shortcut a model could learn
instead:

```yaml
policy_claim: >
  Rank by where delay costs the most: nearness of deadline, discounted by how
  well the claim is actually supported and by how much the outcome matters. Name
  the factor that decided it.
anti_claim: >
  Pick whichever item is listed first, or whichever has the nearest date
  regardless of whether anyone has verified it.
```

Perturbations and holdouts are chosen to separate the two. Review packets show
both claims, so a reviewer checks whether an example teaches the intended policy
rather than whether it merely reads well.

## Hard cases

The catalog includes cases where the correct response is to abstain, ask, or
reject a common heuristic, so the data does not teach that a confident answer is
always available:

| Behavior | Families |
| --- | --- |
| Say the evidence does not separate the options, and name what would | `notif.tied_signals`, `rec.abstain_without_evidence`, `mem.unresolvable_conflict` |
| Ask for confirmation when nothing is established well enough | `rec.verify_before_recommending` |
| Ask what an underspecified request means | `route.ask_when_underspecified` |
| Surface a stale explicit statement that newer evidence contradicts | `mem.stale_explicit_conflict` |
| Ask before looking outside the active workspace | `wsp.absent_in_active` |
| Prefer reliability over recency | `mem.reliability_over_recency` |
| Prefer relevance over recency | `ctx.relevance_over_recency`, `ctx.budget_pressure` |
| Flag relevant items in other workspaces without acting on them | `brief.across_workspaces`, `wsp.cross_workspace_request`, `wsp.scope_boundary` |

In `kleos-policy-v0.0.6`, 338 of 1,350 targets (25%) abstain or ask:
`ask_before_crossing` 100, `request_ambiguous` 90, `stale_explicit_conflict` 64,
`missing_input` 55 and `insufficient_separation` 29.

## Falsification criteria

These outcomes were stated before any model was trained on the data:

| Observation | Would indicate |
| --- | --- |
| Out-of-distribution performance collapses while in-distribution performance improves | The model learned the presentation, not the policy |
| Rankings change under evidence or context reordering | The model learned position |
| Confident answers where the policy abstains | The model learned that an answer is always expected |
| Entities in answers that the prompt never mentions | The model memorized entities |

## Status after the first training runs

kleos-models fine-tuned two models on `kleos-policy-v0.0.6` and evaluated them on
its 349-example test split against the same base models with KLEOS's
prompt-engineered orchestration. Both improved the composite policy score on all
seven tasks (0.4744 to 0.8015 for Ministral-8B, and 0.4755 to 0.8051 for
Mistral-Nemo-12B). As of 2026-09-22 the criteria above stand as follows:

| Criterion | Observation |
| --- | --- |
| Out-of-distribution collapse | Not measurable. The test split is entirely out of distribution, so there is no in-distribution population to compare against. |
| Reordering changes rankings | In the kleos-models consistency evaluation, 10 of 15 groups still change their answer under paraphrase or evidence reordering, compared with 14 and 13 of 15 for the two baselines. The criterion is met for those groups. |
| Confident answers where the policy abstains | Met for conditional abstention. Both models abstained on every test case from families that always abstain (20 of 20, 30 of 30) and on none from families where abstention depends on the evidence (0 of 25, 0 of 3). This points to a family-level shortcut rather than the intended rule. |
| Entities the prompt never mentions | Not measured directly. kleos-models reports fewer responses with fabricated citations after fine-tuning (194 to 116 for Ministral-8B, 147 to 98 for Mistral-Nemo-12B), but not zero. |

Two properties of v0.0.6 limit these results:

- 78 of the 349 test cases (22%) expect a deciding-factor label that never
  appears in any training target. Abstention labels such as `missing_input` are
  printed only in JSON-format answers, and the format holdout puts every
  JSON-format example in the test split.
- Neither the fine-tuned models nor the baselines produced valid JSON output
  (`format_valid` was 0 in both arms). Decision scores improved on the unseen
  input format, but the output format did not transfer.

The next dataset revision should make abstention depend on the evidence within
each family, expose deciding-factor labels in every answer format, and include an
in-distribution test population so that the out-of-distribution gap can be
measured. The full analysis, including the deviations from the pre-registered
protocol, is in the kleos-models
[experiments document](https://github.com/TejasNaik24/Kleos-Models/blob/main/docs/experiments.md#deviations-log).

## Reproducibility

Each release records in `provenance.json` the dataset schema, preprocessing,
privacy ruleset and review rubric versions; the split strategy, seed and
counts; the holdout; the families and perturbation kinds; a fingerprint of
each scenario family; and the pinned contract commit. Build timestamps are not
part of any content hash, so identical content always produces identical hashes.

Releases can be rebuilt from source.

- **At commit `c5cc730`:** running `make slice-clean slice` in a fresh clone builds
  the full catalog into a release whose split files and content hash (`3cc9a744…`)
  are byte-identical to `kleos-policy-v0.0.6`.
- **From the policy-reasoning change on:** the same build reproduces
  `kleos-policy-v0.0.7` byte for byte (content hash `b53afa42…`). Its
  `test.jsonl` is byte-identical to v0.0.6's, so every evaluation stays
  comparable.

The version strings `PIPELINE_VERSION` (0.1.0), `privacy-rules-v1` and
`review-rubric-v1` were not changed between v0.0.2 and v0.0.6, although the code
was. Identify the code state that produced a release from the git commit and the
scenario fingerprints, not from those strings.

## Related documentation

- [scenarios.md](scenarios.md): policies, perturbations and the catalog
- [dataset-lifecycle.md](dataset-lifecycle.md): splitting and holdouts
- [review.md](review.md): how reviewers use policy and anti-claims
- [../CHANGELOG.md](../CHANGELOG.md): what changed in each release
