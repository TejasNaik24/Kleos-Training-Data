# Dataset contract

What a release must look like for `kleos-models` to load it. This is the mirror's
job description; the authoritative source is the public repo at the commit pinned
in `src/kleos_training_data/contract/pin.py`.

## The artifact

```
releases/kleos-policy-v0.1.0/
├── manifest.json        required
├── train.jsonl          required
├── validation.jsonl     optional
├── test.jsonl           optional
├── provenance.json      ours; the public loader ignores it
└── RELEASE.lock         ours; drift detection
```

**Exact filenames only.** The public `validate_dataset.py` also accepts
`synthetic_train.jsonl` and `valid.jsonl`, but `train.py` does not — so an alias
produces a release that validates cleanly and then fails to train.
`verify_release.py` rejects any other `*.jsonl` in the directory.

## One example

```json
{"id": "kx-npr-3f9a1c8e2b7d0456", "task": "notification_prioritization",
 "version": "1.0", "domain": "career",
 "messages": [{"role": "system", "content": "...", "name": null}, ...],
 "variation_axes": {"domain": "career", "urgency": "high", ...},
 "metadata": {"source": "synthetic", "quality_status": "reviewed", ...}}
```

Required: `id`, `task`, `messages` (≥2), `variation_axes` (with `domain`).

| Model | Extra fields |
| --- | --- |
| `TrainingExample` | **forbidden** |
| `Message` | **forbidden** |
| `VariationAxes` | allowed |
| `ExampleMetadata` | allowed — which is why gate G14 closes it |
| `DatasetManifest` | **forbidden** |

### Conversation rules

- at least one assistant message
- the **final** message is from the assistant
- a system message only at index 0
- the first assistant turn is preceded by a user turn
- no two consecutive assistant turns
- no empty or whitespace-only content
- a `tool` message requires a `name`

### Ids

`^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$`. We emit `kx-<task>-<16 hex>`, derived from
the example's canonical content — see `src/kleos_training_data/ids.py`.

### Vocabularies

Seven tasks, five source types, four quality statuses, four message roles. The
literal values live in `contract/constants.py` and are compared element-wise
*and* order-wise against the public package by `make compat`.

## The bytes

```python
(
    json.dumps(
        example.model_dump(mode="json", exclude_none=False), ensure_ascii=False, sort_keys=True
    )
    + "\n"
)
```

Three details, each of which changes every line if lost:

- `exclude_none=False` — every message carries `"name": null`, every unset axis
  carries a null. This is most of the file.
- `sort_keys=True` — alphabetical, not declaration order.
- `ensure_ascii=False` — non-ASCII survives as itself.

## The manifest

Nineteen fields, `extra="forbid"`. One added key makes the public `load_manifest`
raise `DatasetIntegrityError` **at train time**, which is why everything this
repository wants to record lives in a sibling `provenance.json` instead.

`content_hash` digests `file_hashes` and nothing else:

```python
sha256(json.dumps(file_hashes, sort_keys=True, separators=(",", ":")))
```

Adding `provenance.json` to `file_hashes` would make our hash differ from the one
the public `split_dataset.py` computes for identical splits — and that hash is the
cheapest way to confirm two repositories are talking about the same data.

## Two things that will catch you

**`quality_status` must be `"reviewed"`.** The public loader's default filter is
`quality_statuses=["reviewed"]`, so anything else is silently dropped at train
time. A release full of `auto_generated` examples validates cleanly and then
trains on nothing. Gate G09 refuses any other value and `verify_release` asserts
100%.

**`contains_private_data` is computed, not declared.** It is `false` only when the
final byte-level scan of the written files is clean. `--declare-private` can set
it `true`; nothing can set it `false` by hand.

## Verifying

```bash
python scripts/verify_release.py --release <dir> --strict
python scripts/check_contract_compat.py --release <dir> --strict
python <kleos-models>/scripts/validate_dataset.py --dataset <dir> --require-reviewed
```

The third is the one that matters: it is the public repo judging our output with
its own code.
