# Dataset lifecycle

This document describes how promoted examples become an immutable, versioned
release: split assignment, out-of-distribution holdouts, sealing, verification,
versioning and hand-off to kleos-models.

## Contents

- [Building a release](#building-a-release)
- [Splitting](#splitting)
- [Holdouts](#holdouts)
- [Sealing](#sealing)
- [Verification](#verification)
- [Versioning](#versioning)
- [Consuming a release](#consuming-a-release)

## Building a release

`build_release.py` reads every example in `staging/promoted/`, assigns splits and
seals the result under `releases/<version>/`:

```bash
python scripts/build_release.py --version kleos-policy-v0.0.7 --holdout-attribute format --description "One-line summary"
```

| Option | Default | Effect |
| --- | --- | --- |
| `--version <name>` | Required | Release name. An existing version is refused. |
| `--holdout-attribute <name>` | None | Holdout attribute to resolve: `format`, `entities` or `domain`. Required when the catalog declares holdouts on more than one attribute. |
| `--strategy <name>` | From the holdout, otherwise `group` | Split strategy |
| `--seed <n>` | 42 | Split seed |
| `--train-fraction`, `--validation-fraction`, `--test-fraction` | 0.7, 0.15, 0.15 | Split fractions |
| `--group-key <key>` | None | Metadata key that defines a group |
| `--allow-uncovered-holdout` | Off | Allow a reserved value with no examples |
| `--description <text>` | Empty | One-line description for the manifest |
| `--declare-private` | Off | Record `contains_private_data: true` and write the release despite scan findings |
| `--dry-run` | Off | Plan the split without writing anything |

The script exits with code 1 when there are no promoted examples or the version
already exists, and with code 4 when the final privacy scan finds a problem that
was not declared.

## Splitting

A base example and its perturbations share a `group_id`, and a group is never
split across train, validation and test. If a perturbation and its base landed
on opposite sides, a consistency test would compare an example the model trained
on with a near copy of itself.

Assignment uses a stable hash instead of a shuffle:

```python
rank = int.from_bytes(hashlib.sha256(f"{seed}:{group_key}".encode()).digest()[:8], "big") / 2**64
```

A group goes to train when `rank` is below the train fraction, to validation when
it is below the train and validation fractions combined, and to test otherwise.
The assignment depends only on the group key and the seed, so adding examples
never moves existing groups, and two releases built with the same seed and
fractions split their shared groups identically.

| Strategy | Assignment |
| --- | --- |
| `random` | Seeded shuffle of examples sorted by ID. Intended for development only. |
| `group` | Whole groups by stable hash |
| `scenario_family_holdout` | Whole scenario families by stable hash |
| `format_holdout`, `entity_holdout`, `domain_holdout` | Examples with held-out attribute values go to test. Validation comes from the remaining, seen values. |

Holdout strategies need at least two distinct values of the attribute, and
holding out every value is an error. Because validation contains only seen
values, early stopping never looks at the out-of-distribution data the test
split is meant to measure.

`build_release.py` defaults to 0.7/0.15/0.15, while the `SplitConfig` model
defaults to 0.8/0.1/0.1. `provenance.json` records the strategy, seed, split
counts and group count, but not the fractions themselves, so a rebuild must pass
the same fractions (the defaults, unless they were overridden).

## Holdouts

Scenario families declare their holdouts in the catalog (`reserve_formats`,
`reserve_entity_pools`, `reserve_domains`), and `resolve_holdouts()` turns those
declarations into explicit values for the splitter. Declaring holdouts first
means the test split measures a shift chosen in advance.

These conditions stop the build:

| Condition | Reason |
| --- | --- |
| Holdouts are declared on two attributes and `--holdout-attribute` is not given | A drop in test performance could not be attributed to either shift |
| A reserved value has no examples | The test split would be silently empty. `--allow-uncovered-holdout` permits it. |
| The holdout covers every value | No training data would remain |
| A family declares an unregistered shift kind | The shift could not be reported |

A family must not perturb the attribute its holdout reserves. A `formatting` or
`schema` perturbation of a family that reserves `json` would put a base example
and its variant on opposite sides of the split, so `validate_scenarios.py`
rejects that combination.

The current catalog declares format holdouts in 15 families and entity-pool
holdouts in 4, so builds pass `--holdout-attribute format`. Every release so far
holds out `format=json`.

## Sealing

A release is assembled in a temporary directory and moved into place in one
step:

1. Refuse to continue if `releases/<version>/` already exists.
2. Create `releases/.staging-<random hex>/`.
3. Write each non-empty split using the canonical file names.
4. Build the manifest from the files as written.
5. Re-read every split and check its count and ID order, which catches a
   truncated write.
6. Scan the written bytes for secrets and PII, unless `--declare-private` was
   given.
7. Write `provenance.json`.
8. Write `manifest.json`.
9. Write `RELEASE.lock` with the content hash and the hash of every file.
10. Move the directory into place with `os.replace`. On any error, the staging
    directory is deleted.
11. Set the release files to mode 0444 and the directory to 0555.

`build_release.py` has no option to overwrite a release. If the content needs to
change, the version must change, because every comparison and every model
trained on a version refers to one specific set of bytes. Read-only permissions
prevent accidental edits, and `RELEASE.lock` combined with verification detects
any change that does happen.

## Verification

```bash
python scripts/verify_release.py --release releases/kleos-policy-v0.0.6 --strict
```

`verify_release.py` re-reads the files on disk and recomputes what the manifest
and lock file claim:

| Check | Detects |
| --- | --- |
| Manifest present, version matches the directory name | Mislabeled or incomplete releases |
| Only canonical split files, `train.jsonl` present, every line parses | Aliased or corrupt files |
| Total count, per-split counts and all four distributions | Manifest drift |
| No duplicate IDs, no ID in two splits, no group across two splits | Leakage between splits |
| 100% `quality_status: reviewed` | Examples the kleos-models loader would drop |
| Every `file_hashes` entry and the `content_hash` | Changed files |
| Content hash and every file hash in `RELEASE.lock` | Changes since sealing |
| Byte-level privacy scan of each split | Secrets or PII in the written files |

These 17 check types add up to 32 checks on a standard three-split release. A missing `RELEASE.lock` or `provenance.json` is a warning, and
`--strict` treats warnings as failures. The script exits with code 0 when the
release verifies, 4 when a problem involves privacy, and 3 otherwise.

The CI `vertical slice` job builds a complete release from the catalog on every
push to `main` and every pull request, and verifies it with `--strict`.

## Versioning

Releases are named `kleos-policy-vMAJOR.MINOR.PATCH`:

| Change | Version bump |
| --- | --- |
| Metadata, formatting or a non-semantic preprocessing fix | Patch |
| New reviewed examples or wider coverage under the same contract | Minor |
| Changed task definitions, schema assumptions, preprocessing behavior or research interpretation | Major |

Releases v0.0.1 to v0.0.6 are pre-0.1 research iterations and are numbered
sequentially. Each release has a [CHANGELOG](../CHANGELOG.md) entry, and records
its content hash, file hashes, split strategy and seed, and pinned contract
commit.

## Consuming a release

kleos-models consumes a release as a directory path:

```bash
python ../Kleos-Models/scripts/train.py --config <config> --dataset releases/kleos-policy-v0.0.6
```

`releases/` is git-ignored. Releases are build artifacts and are not
distributed through this repository. See
[DATA_GOVERNANCE.md](../DATA_GOVERNANCE.md#distribution) for how they are
shared.

## Related documentation

- [DATASET_CONTRACT.md](../DATASET_CONTRACT.md): the format of a release
- [scenarios.md](scenarios.md#holdouts): how families declare holdouts
- [research-protocol.md](research-protocol.md): what the holdouts are meant to
  measure
- [compatibility.md](compatibility.md): checking a release against kleos-models
