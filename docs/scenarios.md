# Scenarios

A scenario family is a YAML file that describes a class of decision situations,
the policy that resolves them, and the ways they vary. The generator expands each
family into base situations and perturbations, computes every training target
from the family's registered policy, and renders the conversation. This guide
covers the concepts, the current catalog, the YAML field reference, and the steps
for adding a family.

## Contents

- [Concepts](#concepts)
- [Catalog](#catalog)
- [Field reference](#field-reference)
- [Decision policies](#decision-policies)
- [Framings](#framings)
- [Evidence and impact grades](#evidence-and-impact-grades)
- [Derived difficulty](#derived-difficulty)
- [Perturbations](#perturbations)
- [Holdouts](#holdouts)
- [Authoring a family](#authoring-a-family)
- [Validation](#validation)

## Concepts

| Term | Meaning |
| --- | --- |
| Scenario family | One YAML file under `scenarios/<task>/`. It declares the task, the axes to vary, the entity pool, the policy, the perturbations and the holdout. |
| Situation | One concrete instance generated from a family: two to eight candidate items, each with a deadline or age, an evidence grade and an impact grade. |
| Policy | A registered Python function that maps a situation to a `Decision`. The policy produces the training target; no target is hand-written or taken from model output. |
| Decision | The policy's output: a ranking, the deciding factor, a rationale for each item and, when the policy abstains, a resolver that names what would settle the question. |
| Framing | How a situation is presented: the system instruction, the noun used for an item, and whether time reads as a due date, an age or a sync staleness. |
| Perturbation | A variant of a base example (a paraphrase, a reordering, added noise, a different length or document format) that must produce the same decision. |
| Perturbation group | A base example and its perturbations. Members share a `group_id` and always land in the same split. |

Generation is deterministic. The same catalog and code produce the same examples,
and each example's ID is derived from its content (see
[DATASET_CONTRACT.md](../DATASET_CONTRACT.md#ids)).

## Catalog

The catalog in `scenarios/` contains 18 families that cover all seven tasks.
Example counts are from release `kleos-policy-v0.0.6`.

| Family | Task | Policy | Framing | Examples | Test | OOD shift |
| --- | --- | --- | --- | ---: | ---: | --- |
| `ctx.budget_pressure` | context_prioritization | `rank_by_relevance_over_recency` | context | 120 | 0 | unseen_entities |
| `ctx.relevance_over_recency` | context_prioritization | `rank_by_relevance_over_recency` | context | 24 | 8 | reordered_evidence |
| `mem.explicit_over_inferred` | memory_conflict_resolution | `defer_to_explicit_statement` | memory | 100 | 35 | unseen_formats |
| `mem.reliability_over_recency` | memory_conflict_resolution | `rank_by_reliability_over_recency` | memory | 88 | 32 | conflicting_evidence |
| `mem.stale_explicit_conflict` | memory_conflict_resolution | `defer_to_explicit_statement` | memory | 64 | 20 | conflicting_evidence |
| `mem.unresolvable_conflict` | memory_conflict_resolution | `resolve_or_abstain_on_support` | memory | 72 | 24 | conflicting_evidence |
| `brief.across_workspaces` | mission_control_briefing | `respect_workspace_scope` | workspace | 100 | 35 | unseen_formats |
| `brief.what_needs_a_decision` | mission_control_briefing | `rank_by_deadline_then_evidence` | briefing | 24 | 8 | context_length_shift |
| `notif.deadline_vs_evidence` | notification_prioritization | `rank_by_deadline_then_evidence` | priority | 24 | 8 | unseen_entities |
| `notif.tied_signals` | notification_prioritization | `rank_or_abstain_when_close` | priority | 100 | 30 | unseen_formats |
| `rec.abstain_without_evidence` | recommendation_generation | `rank_or_abstain_when_close` | priority | 18 | 6 | unseen_formats |
| `rec.verify_before_recommending` | recommendation_generation | `verify_when_evidence_weak` | priority | 100 | 35 | unseen_formats |
| `route.ask_when_underspecified` | tool_routing | `ask_when_request_ambiguous` | routing | 90 | 30 | unseen_formats |
| `route.least_privilege` | tool_routing | `prefer_least_privilege_source` | routing | 100 | 0 | unseen_entities |
| `route.source_matches_question` | tool_routing | `select_by_evidence_need` | routing | 88 | 32 | unseen_formats |
| `wsp.absent_in_active` | workspace_reasoning | `ask_before_crossing_workspace` | workspace | 100 | 0 | unseen_entities |
| `wsp.cross_workspace_request` | workspace_reasoning | `respect_workspace_scope` | workspace | 120 | 40 | unseen_formats |
| `wsp.scope_boundary` | workspace_reasoning | `respect_workspace_scope` | workspace | 18 | 6 | unseen_formats |
| Total | | | | 1,350 | 349 | |

`rec.verify_before_recommending` is defined in
`scenarios/recommendation_generation/ask_when_constraint_missing.yaml`. Three
families (`ctx.budget_pressure`, `route.least_privilege` and
`wsp.absent_in_active`) do not use the `json` format, so they contribute no test
examples under the current format holdout.

## Field reference

Every block rejects unknown keys. A misspelled field such as
`pertubation_kinds` fails validation instead of being silently ignored.

### Top-level fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `family` | string | required | Unique ID of at least 3 characters (letters, digits, `.`, `_`, `-`). The part after the last `.` should match the file name. |
| `task` | string | required | One of the seven supported tasks. |
| `policy_claim` | string | required | What the family teaches, at least 10 characters. Shown to reviewers as "Should teach". |
| `anti_claim` | string | required | The surface rule a model could learn instead, at least 10 characters. Shown to reviewers as "Must not teach". |
| `axes` | map of lists | `{}` | Variation axes and their values. See [Axes](#axes). |
| `entities` | block | required | Where item names come from. |
| `prompt` | block | defaults | How the situation is rendered. |
| `expected` | block | required | The policy that computes the target. |
| `generation` | block | defaults | Sampling and perturbations. |
| `holdout` | block | defaults | Values reserved for out-of-distribution evaluation. |
| `review` | block | defaults | Informational review settings. |
| `catalog_version` | string | `scenarios-v1` | Recorded on capture records and in the promotion audit trail. |
| `schema_version` | string | `1.0` | Informational. |
| `description` | string | empty | Informational. |
| `notes` | string | none | Informational. Never enters an example. |

### Axes

`axes` maps each axis name to the values the generator cycles through.

- `domain` is required and must be non-empty. No axis may have an empty list.
- `format` values must be registered prompt formats: `bullets`, `prose`, `json`,
  `slack_thread`, `github_issue` or `calendar`.
- Axis names outside the registered vocabulary are accepted but reported by the
  validator.
- The `workspace` framing requires a `workspace` axis whose values are a subset
  of `prompt.workspace_names`.

These axes change the rendered content:

| Axis | Effect |
| --- | --- |
| `urgency` | Deadline pool in days: critical 0, 1, 2; high 1, 2, 3, 5; medium 4, 7, 10, 14; low 14, 21, 30, 45 |
| `evidence_quality` | Evidence grades per item: strong (confirmed, corroborated), mixed (confirmed, reported, single_source), weak (single_source, unverified), conflicting (confirmed, contradicted, reported) |
| `context_length` | Unrelated distractor sentences: short 0, medium 2, long 4 |
| `presentation_order` | Item order: as_given, reversed or shuffled |
| `format` | Prompt and answer format |
| `workspace` | Active workspace name (workspace framing only) |
| `difficulty` | Deadline spacing between items: easy 2 days, medium 1, hard 0 |

`domain`, `ambiguity` and `conflicting_evidence` are recorded as example
metadata and used for coverage reporting. They do not change the rendered text.

### Entities block

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `entities.pool` | string | required | Surrogate pool in `data/surrogates/<pool>.yaml` that supplies item names. |
| `entities.count` | integer | 3 | Items per situation, from 2 to 8 and no more than the pool size. |

### Prompt block

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt.framing` | string | `priority` | One of the six [framings](#framings). |
| `prompt.question` | string | `Which should I deal with first, and why?` | Closing question of the user turn, at least 8 characters. |
| `prompt.need` | string | empty | Opening line of the user turn when set. |
| `prompt.workspace_names` | list | empty | Workspace names. Required by the workspace framing. |
| `prompt.out_of_scope_count` | integer | 0 | Items placed in another workspace, from 0 to 4 and fewer than `entities.count`. Workspace framing only. |
| `prompt.out_of_scope_stronger` | boolean | false | Gives out-of-scope items strong evidence and in-scope items weak evidence. |
| `prompt.request_ambiguous` | boolean | false | Marks the request as underspecified. Read by `ask_when_request_ambiguous`. |
| `prompt.stale_explicit_conflict` | boolean | false | Makes the first item an explicit statement at least 45 days old and the second a corroborated record at most 7 days old. |
| `prompt.item_noun` | string | `item` | Informational. The framing sets the noun. |

### Expected block

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `expected.policy` | string | required | A registered policy name, checked when the family is generated. |
| `expected.grader` | string | `ranking` | Informational. |

### Generation block

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `generation.n_base` | integer | 8 | Base situations, from 1 to 500 and no more than the axis space (the product of the axis value counts). |
| `generation.seed` | integer | 1712 | Offsets the stratified sampling. |
| `generation.sampling` | string | `stratified` | `stratified` uses each value of each axis a near-equal number of times. `grid` walks the Cartesian product. |
| `generation.equivalence_groups` | list | empty | Entries of `{kind, count}`, where `kind` is a [perturbation](#perturbations) and `count` is 1 to 5. |

A family produces `n_base × (1 + sum of counts)` examples.

### Holdout block

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `holdout.ood_shift` | string | none | The registered out-of-distribution shift the family represents. |
| `holdout.reserve_formats` | list | empty | Formats reserved for the test split. Each must appear in `axes.format`. Cannot be combined with `formatting` or `schema` perturbations. |
| `holdout.reserve_entity_pools` | list | empty | Pools reserved for the test split. Each must exist and differ from `entities.pool`. |
| `holdout.reserve_domains` | list | empty | Domains reserved for the test split. Each must appear in `axes.domain`. |

### Review block

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `review.lane` | string | `llm_then_human` | One of `llm_then_human`, `mock_only`, `human_only`. Validated and recorded but not read by promotion. |
| `review.min_mean_score` | number | 3.0 | From 0 to 4. Validated and recorded but not read by promotion, which applies its own minimum mean of 3.0. |

## Decision policies

Most policies start from a priority score:

```text
score = deadline_score × evidence_weight × impact_weight
deadline_score = 1 / (1 + days)
```

Ties are broken by the item key, so every ranking is total and reproducible.
Every policy requires at least two items. "Adequate" means an evidence weight of
at least `reported` (0.55), and an "explicit statement" is an item graded
`confirmed`.

| Policy | Orders by | Abstains when | Deciding factor |
| --- | --- | --- | --- |
| `rank_by_deadline_then_evidence` | Score | Never | Deadline, evidence or impact, whichever differs most between the top two |
| `rank_or_abstain_when_close` | Score | The top two are within a 15% relative separation | `insufficient_separation` |
| `rank_by_reliability_over_recency` | Evidence, then recency | Never | Evidence, or deadline when evidence ties |
| `resolve_or_abstain_on_support` | Evidence, then recency | The top two are within 15% on evidence and were recorded at the same time | `insufficient_separation` |
| `select_by_evidence_need` | Evidence × impact, then freshness | Never | Evidence or impact |
| `rank_by_relevance_over_recency` | Impact (relevance), then recency | Never | Impact, or deadline when impact ties |
| `respect_workspace_scope` | In-scope items first, then score | Never. Out-of-scope items are flagged, not acted on. | `scope` when scopes are mixed, otherwise the score ranking |
| `ask_before_crossing_workspace` | Same as `respect_workspace_scope` | No in-scope item is adequate and the strongest out-of-scope item is | `ask_before_crossing` |
| `defer_to_explicit_statement` | Explicit statements, then evidence, then recency | The explicit statement is at least 30 days old and a newer inferred record has evidence of at least 0.8 | `explicit_statement`, `evidence` or `stale_explicit_conflict` |
| `verify_when_evidence_weak` | Score | No item is adequate | `missing_input` |
| `ask_when_request_ambiguous` | Score | The family sets `prompt.request_ambiguous` | `request_ambiguous` |
| `prefer_least_privilege_source` | Adequacy, then narrowest reach, then evidence | Never | `least_privilege` when several sources are adequate, otherwise `evidence` |

The thresholds are named constants in `src/kleos_training_data/scenarios/policies.py`:

| Constant | Value | Used by |
| --- | --- | --- |
| `CLOSE_CALL_RELATIVE_MARGIN` | 0.15 | Relative separation `\|a − b\| / max(a, b)`, which is unchanged when both scores are scaled by the same factor |
| `STALE_AFTER_DAYS` | 30 | Age at which an explicit statement can be challenged |
| `STRONG_CONFLICT_MIN_EVIDENCE` | 0.8 | Evidence a newer record needs to challenge a stale statement (`corroborated`) |

A `Decision` carries `ranking`, `deciding_factor`, `rationale`, `abstained` and
`resolver`. `Decision.comparable()` returns `(ranking, deciding_factor,
abstained)`, and every perturbation must preserve that tuple.

## Framings

| Framing | Item noun | Time reads as | JSON time field | Evidence and impact labels | Instruction |
| --- | --- | --- | --- | --- | --- |
| `priority` | item | Due date | `due_in_days` | evidence, impact | Rank by where delay costs the most and name the deciding factor. |
| `briefing` | update | Due date | `due_in_days` | evidence, impact | Lead with what needs a decision soonest and is best supported. |
| `routing` | source | Sync staleness | `last_synced_days_ago` | bearing_on_question, coverage | Choose the source that bears most directly on the question. |
| `memory` | record | Record age | `recorded_days_ago` | support, stakes | Prefer the better-supported record. An explicit statement outranks an inference unless it is stale and contradicted. |
| `context` | fragment | Age | `last_touched_days_ago` | support, relevance | Keep what bears on the question. Recency only breaks ties. |
| `workspace` | item | Due date | `due_in_days` | evidence, impact | Answer from the active workspace and flag items in other workspaces. |

The full instruction text for each framing is defined in `FRAMINGS` in
`src/kleos_training_data/scenarios/situations.py`. The priority, briefing,
routing and memory instructions also tell the model to say when the evidence
cannot separate the options instead of guessing.

## Evidence and impact grades

| Evidence grade | Weight | Rendered as |
| --- | ---: | --- |
| `confirmed` | 1.0 | confirmed directly by the owner |
| `corroborated` | 0.8 | corroborated by two independent sources |
| `reported` | 0.55 | reported in the weekly update |
| `single_source` | 0.35 | mentioned once, by a single source |
| `unverified` | 0.2 | unverified — nobody has checked it |
| `contradicted` | 0.1 | contradicted by a later message |

| Impact grade | Weight |
| --- | ---: |
| `high` | 1.0 |
| `medium` | 0.6 |
| `low` | 0.3 |

## Derived difficulty

The `difficulty` label on each example is computed from the situation under the
policy's own ordering. It is not copied from the declared axis. Each policy
registers a lexicographic sort key in `ORDERING_KEYS`
(`src/kleos_training_data/scenarios/difficulty.py`), and the label depends on the
first component where the top two candidates differ:

| First component that differs | Relative gap of at least 0.20 | Relative gap below 0.20 |
| --- | --- | --- |
| A gating criterion (`scope`, `adequate` or `explicit`) | easy | easy |
| The primary ranking component | easy | medium |
| A later ranking component | medium | hard |
| None (the order falls to the item key) | hard | hard |

Gating criteria shared by both candidates are skipped without counting as a
level. The 0.20 threshold (`ONE_STEP`) is the smallest relative gap between
adjacent evidence grades, `(1.0 − 0.8) / 1.0`, and a test keeps it equal to the
evidence scale. The declared `difficulty` axis only sets deadline spacing during
generation. A family without a `difficulty` axis ships examples with no label.

## Perturbations

| Kind | Change |
| --- | --- |
| `paraphrase` | Replaces the question with one of three alternative wordings |
| `evidence_order` | Changes the item order among as_given, reversed and shuffled |
| `context_order` | Changes the item order to shuffled or reversed |
| `irrelevant_context` | Adds up to two unused distractor sentences |
| `length` | Trims distractors to one, or adds one when there were none |
| `schema` | Re-renders the prompt in another document format (with `count: 1`, a Slack-style thread) |
| `formatting` | Switches between bullets, prose and JSON. Not used by the current catalog. |

Each perturbation must keep `Decision.comparable()` equal to its base and must
produce content that differs from every other example. Generation fails
otherwise.

## Holdouts

Families declare in advance what they reserve for out-of-distribution
evaluation, and `build_release.py` turns those declarations into explicit holdout
values. In the current catalog, 15 families reserve the `json` format, four
reserve an entity pool, and none reserve a domain. Every release so far holds out
`format=json`, so the test split contains exactly the JSON-format examples.

The generator draws names only from `entities.pool`. Reserved entity pools are
declared for future entity-holdout releases and currently produce no examples.
See [dataset-lifecycle.md](dataset-lifecycle.md#holdouts) for how holdouts are
resolved and checked.

## Authoring a family

1. Pick the task directory under `scenarios/` and a family ID with the task's
   prefix (`ctx`, `mem`, `brief`, `notif`, `rec`, `route` or `wsp`).
2. Write the `policy_claim` and the `anti_claim`. The anti-claim names the
   shortcut the family is designed to catch.
3. Choose a registered policy. A new policy needs a function in `policies.py`,
   an entry in `POLICIES`, an ordering in `ORDERING_KEYS`, and tests.
4. Choose the axes and `n_base`, then add equivalence groups.
5. Declare the holdout.
6. Run `make scenarios` and fix every reported problem.

A complete family, from
`scenarios/memory_conflict_resolution/reliability_over_recency.yaml`:

```yaml
schema_version: "1.0"
catalog_version: "scenarios-v1"
family: mem.reliability_over_recency
task: memory_conflict_resolution
description: >
  Two or more stored records disagree. They differ in how well supported each is
  and in how recently each was written.
policy_claim: >
  When stored records conflict, prefer the better-supported one. Recency decides
  only when the records are equally well supported.
anti_claim: >
  Always trust the most recent record, because newer information supersedes
  older information.
axes:
  domain: [career, research, coursework, projects]
  urgency: [critical, high, medium, low]
  evidence_quality: [conflicting, mixed, strong]
  conflicting_evidence: [moderate, strong]
  format: [bullets, prose, json]
  context_length: [short, medium]
  presentation_order: [as_given, reversed, shuffled]
  difficulty: [easy, medium, hard]
entities: {pool: generic_pool_a, count: 3}
prompt:
  framing: memory
  need: >-
    My stored records disagree about which commitment is still active.
  question: "These disagree. Which should I treat as current, and why?"
  item_noun: record
expected: {policy: rank_by_reliability_over_recency, grader: ranking}
generation:
  n_base: 22
  seed: 6645
  equivalence_groups:
    - {kind: paraphrase, count: 1}
    - {kind: evidence_order, count: 1}
    - {kind: irrelevant_context, count: 1}
holdout:
  ood_shift: conflicting_evidence
  reserve_formats: [json]
review: {lane: llm_then_human, min_mean_score: 3.0}
```

This family produces 22 × (1 + 3) = 88 examples.

## Validation

```bash
make scenarios
```

`make scenarios` runs `python scripts/validate_scenarios.py --strict`, which
checks each family for the following and exits with code 3 on any failure:

- The YAML matches the schema, with no unknown keys.
- The entity pool exists and has at least `entities.count` names.
- Holdout reservations match the declared axes.
- `n_base` fits within the axis space.
- Generation succeeds and yields the expected number of examples.
- No two examples share an ID, and no equivalence group has a single member.
- A declared `difficulty` axis has an ordering registered for the policy.
- Every supported task has at least one family (with `--strict`).

The command also prints declared versus derived difficulty for each family and a
catalog coverage summary. Use `--family <id>` to validate one family.

## Related documentation

- [research-protocol.md](research-protocol.md): why targets are computed and how
  generalization is measured
- [dataset-lifecycle.md](dataset-lifecycle.md): splitting, holdouts and sealing
- [DATASET_CONTRACT.md](../DATASET_CONTRACT.md): the example schema and ID format
- [review.md](review.md): how reviewers use `policy_claim` and `anti_claim`
