# Dataset contract

This document specifies the format a release must follow for
[kleos-models](https://github.com/TejasNaik24/Kleos-Models) to load it. The
authoritative definition is the kleos-models source at the commit pinned in
`src/kleos_training_data/contract/pin.py`. This repository mirrors that contract
and tests the mirror against it, as described in
[docs/compatibility.md](docs/compatibility.md).

## Contents

- [Release layout](#release-layout)
- [Example schema](#example-schema)
- [Conversation rules](#conversation-rules)
- [IDs](#ids)
- [Vocabularies](#vocabularies)
- [Serialization](#serialization)
- [Manifest](#manifest)
- [Common pitfalls](#common-pitfalls)
- [Validation](#validation)

## Release layout

```text
releases/kleos-policy-v0.0.6/
├── manifest.json
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── provenance.json
└── RELEASE.lock
```

| File | Required by kleos-models | Purpose |
| --- | --- | --- |
| `manifest.json` | Yes | Counts, distributions, split configuration and file hashes |
| `train.jsonl` | Yes | Training examples, one JSON object per line |
| `validation.jsonl` | No | Validation examples |
| `test.jsonl` | No | Test examples |
| `provenance.json` | No | Pipeline, rubric, ruleset and contract versions, split details and scenario fingerprints. Ignored by the kleos-models loader. |
| `RELEASE.lock` | No | Content hash and per-file hashes recorded at sealing time, used to detect later changes. Ignored by the kleos-models loader. |

Split files use exactly these names. The kleos-models `validate_dataset.py` also
accepts `synthetic_train.jsonl` and `valid.jsonl`, but its `train.py` does not,
so `verify_release.py` rejects any other `*.jsonl` file in a release.

## Example schema

Each line of a split file is one `TrainingExample`:

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `id` | string | Yes | Matches the [ID pattern](#ids) |
| `version` | string | No | Defaults to `1.0` |
| `task` | string | Yes | One of the seven [tasks](#vocabularies) |
| `domain` | string | No | When set, must equal `variation_axes.domain` |
| `messages` | list of messages | Yes | At least two, following the [conversation rules](#conversation-rules) |
| `variation_axes` | object | Yes | Requires `domain` |
| `metadata` | object | No | Source, quality status and grouping information |

Each message has a `role` (`system`, `user`, `assistant` or `tool`), a non-empty
`content` string, and an optional `name`.

`variation_axes` has 13 named fields, all optional except `domain`: `domain`,
`entities`, `urgency`, `deadlines`, `evidence_quality`, `conflicting_evidence`,
`context_length`, `presentation_order`, `format`, `source_type`, `workspace`,
`difficulty` and `ambiguity`. Additional keys are allowed, and releases from this
pipeline also record `task`.

`metadata` has nine named fields: `source`, `quality_status`, `scenario_family`,
`group_id`, `perturbation_of`, `perturbation_kind`, `author`, `created_at` and
`notes`. The upstream model allows additional keys. This pipeline allows only
`ruleset_version`, `rubric_version`, `scenario_catalog_version`,
`policy_claim_id` and `pipeline_version`, enforced by promotion gate G14.
Promotion sets `quality_status` to `reviewed` on every promoted example, so in a
release it means the example passed promotion, including a recorded review
decision.

| Model | Unknown keys |
| --- | --- |
| `TrainingExample` | Rejected |
| `Message` | Rejected |
| `VariationAxes` | Allowed |
| `ExampleMetadata` | Allowed upstream, restricted by promotion gate G14 |
| `DatasetManifest` | Rejected |

An example from `kleos-policy-v0.0.6`, pretty-printed for readability (split
files store each example on a single line):

```json
{
  "domain": "projects",
  "id": "kx-mcr-6d55b75ee136f566",
  "messages": [
    {
      "content": "You resolve conflicts between stored records. Prefer the better-supported record over the merely newer one; recency decides only when two records are supported equally well, and how much rides on the record breaks a remaining tie. An explicit statement from the user outranks anything inferred, unless it is clearly stale and newer corroborated evidence contradicts it — then surface the conflict rather than overwriting what was said. Name the factor that decided it, and say plainly when the records cannot be reconciled from what is stored.",
      "name": null,
      "role": "system"
    },
    {
      "content": "My stored records disagree about which commitment is still active.\n\nFoxglove was recorded yesterday. It is contradicted by a later message, and it is low stakes. It was reopened after being marked complete. Ivybridge was recorded 3 days ago. It is reported in the weekly update, and it is medium stakes. Most of the work is done; it needs a final review. Longmere was recorded 2 days ago. It is confirmed directly by the owner, and it is high stakes. It is blocked on one outstanding answer.\n\nThese disagree. Which should I treat as current, and why?",
      "name": null,
      "role": "user"
    },
    {
      "content": "Start with Longmere — confirmed directly by the owner; recorded 2 days ago. Then Ivybridge, then Foxglove. The newer record is not the better one here: confirmed beats reported, and treating recency as authority is how a contradicted note becomes the current answer.",
      "name": null,
      "role": "assistant"
    }
  ],
  "metadata": {
    "author": "kleos-training-data",
    "created_at": null,
    "group_id": "mem.reliability_over_recency:0020",
    "notes": null,
    "perturbation_kind": null,
    "perturbation_of": null,
    "quality_status": "reviewed",
    "scenario_family": "mem.reliability_over_recency",
    "source": "synthetic"
  },
  "task": "memory_conflict_resolution",
  "variation_axes": {
    "ambiguity": null,
    "conflicting_evidence": "moderate",
    "context_length": "short",
    "deadlines": null,
    "difficulty": "easy",
    "domain": "projects",
    "entities": null,
    "evidence_quality": "conflicting",
    "format": "prose",
    "presentation_order": "shuffled",
    "source_type": null,
    "task": "memory_conflict_resolution",
    "urgency": "high",
    "workspace": null
  },
  "version": "1.0"
}
```

## Conversation rules

A training example is rejected unless all of the following hold:

- It has at least two messages and at least one assistant message.
- The final message is from the assistant.
- A system message appears only at index 0.
- The first assistant message is preceded by a user message.
- No two assistant messages are consecutive.
- No message content is empty or whitespace-only.
- Every `tool` message has a `name`.

## IDs

IDs must match `^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$`.

This pipeline emits IDs of the form `kx-<task code>-<16 hex characters>`. The
hex part is the start of a SHA-256 digest over the example's canonical content:
the task, each message's role, content and name, and the non-null variation
axes. Message text is normalized before hashing (line endings converted to LF,
trailing spaces removed from each line, Unicode NFC, outer whitespace stripped).
Metadata, `id`, `version` and `domain` are excluded, so reviewing or re-releasing
an example leaves its ID unchanged, while any change to its content produces a
new ID.

Promotion gate G03 recomputes each ID from the content and rejects a mismatch,
and G12 rejects two different contents that share an ID. `data/id_ledger.json`
is reserved for recording longer IDs after a collision. It is empty, because no
collision has occurred.

| Task | Code |
| --- | --- |
| `notification_prioritization` | `npr` |
| `tool_routing` | `trt` |
| `mission_control_briefing` | `mcb` |
| `context_prioritization` | `ctx` |
| `recommendation_generation` | `rec` |
| `memory_conflict_resolution` | `mcr` |
| `workspace_reasoning` | `wsr` |

## Vocabularies

| Vocabulary | Values |
| --- | --- |
| Tasks | `notification_prioritization`, `tool_routing`, `mission_control_briefing`, `context_prioritization`, `recommendation_generation`, `memory_conflict_resolution`, `workspace_reasoning` |
| Source types | `synthetic`, `synthetic_seeded`, `real_sanitized`, `expert_authored`, `development_fixture` |
| Quality statuses | `draft`, `auto_generated`, `reviewed`, `rejected` |
| Message roles | `system`, `user`, `assistant`, `tool` |
| Variation axes | `domain`, `entities`, `urgency`, `deadlines`, `evidence_quality`, `conflicting_evidence`, `context_length`, `presentation_order`, `format`, `source_type`, `workspace`, `task`, `difficulty`, `ambiguity` |
| Perturbation kinds | `paraphrase`, `evidence_order`, `context_order`, `irrelevant_context`, `formatting`, `schema`, `length` |
| OOD shift kinds | `unseen_entities`, `unseen_domains`, `unseen_formats`, `unseen_source_types`, `context_length_shift`, `reordered_evidence`, `conflicting_evidence` |
| Split strategies | `random`, `group`, `entity_holdout`, `domain_holdout`, `format_holdout`, `scenario_family_holdout` |

The mirrored values live in `src/kleos_training_data/contract/constants.py`.
`make compat` compares each one with the kleos-models package, including the
order of the values.

## Serialization

Each example is written as:

```python
line = (
    json.dumps(
        example.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n"
)
```

Split files are UTF-8. Three settings determine the bytes, and changing any of
them changes every line:

| Setting | Effect |
| --- | --- |
| `exclude_none=False` | Every message carries `"name": null` and every unset axis is written as `null` |
| `sort_keys=True` | Keys appear in alphabetical order, not declaration order |
| `ensure_ascii=False` | Non-ASCII characters are written as themselves, not as escapes |

## Manifest

`manifest.json` has 19 fields and rejects unknown keys:

| Field | Description |
| --- | --- |
| `version` | Release name, for example `kleos-policy-v0.0.6` |
| `schema_version` | Dataset schema version (`1.0`) |
| `preprocessing_version` | Preprocessing version (`1.0`) |
| `created_at` | Creation timestamp (UTC) |
| `source` | Dataset source type |
| `description` | One-line description |
| `example_count` | Total examples across splits |
| `splits` | Counts for `train`, `validation` and `test` |
| `task_distribution` | Examples per task |
| `domain_distribution` | Examples per domain |
| `source_distribution` | Examples per source type |
| `quality_distribution` | Examples per quality status |
| `split_strategy` | Strategy used to assign splits |
| `split_seed` | Seed used for split assignment |
| `holdout_values` | Values held out for the test split |
| `file_hashes` | SHA-256 of each split file |
| `content_hash` | Digest of `file_hashes` |
| `contains_private_data` | Result of the final byte-level privacy scan |
| `notes` | Free-text notes |

An unknown key makes the kleos-models `load_manifest` raise
`DatasetIntegrityError` when training starts. Anything else this pipeline needs
to record goes in `provenance.json`.

`content_hash` is computed from `file_hashes` alone:

```python
content_hash = hashlib.sha256(
    json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
```

`file_hashes` covers only the split files, so the hash matches the one the
kleos-models `split_dataset.py` computes for the same splits. Comparing the two
is the quickest way to confirm that both repositories refer to the same data.

`contains_private_data` is computed rather than declared. It is `false` only
when the byte-level scan of the written files is clean.
`build_release.py --declare-private` records `true` and allows the release to be
written despite scan findings. No option sets it to `false`.

## Common pitfalls

| Pitfall | Consequence | Safeguard |
| --- | --- | --- |
| `quality_status` other than `reviewed` | The kleos-models loader keeps only `reviewed` examples by default, so other examples are silently dropped at training time | Promotion writes `reviewed` on every promoted example and `verify_release.py` checks that 100% of examples are reviewed |
| Non-canonical split file names | The release validates but fails to train | `verify_release.py` rejects unexpected `*.jsonl` files |
| Extra manifest keys | Training fails when the manifest is loaded | Pipeline-specific data is written to `provenance.json` |

## Validation

Run these checks against a release before training on it:

```bash
python scripts/verify_release.py --release releases/kleos-policy-v0.0.6 --strict
python scripts/check_contract_compat.py --release releases/kleos-policy-v0.0.6 --strict
python ../Kleos-Models/scripts/validate_dataset.py --dataset releases/kleos-policy-v0.0.6 --require-reviewed
```

The third command runs the kleos-models validator from a sibling checkout, so the
consuming repository checks the release with its own code.

## Related documentation

- [docs/dataset-lifecycle.md](docs/dataset-lifecycle.md): splitting, sealing and
  verification
- [docs/compatibility.md](docs/compatibility.md): how the mirrored contract is
  kept in sync
- [docs/scenarios.md](docs/scenarios.md): how examples are generated
