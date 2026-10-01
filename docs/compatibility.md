# Compatibility with kleos-models

This repository mirrors the kleos-models dataset contract instead of importing
it, and verifies the mirror with differential tests and a compatibility check.
This document explains the pin, what is compared, and how to move the pin to a
newer kleos-models commit.

## Contents

- [Mirrored contract](#mirrored-contract)
- [Pinned commit](#pinned-commit)
- [Compatibility checks](#compatibility-checks)
- [Strict mode](#strict-mode)
- [Updating the pin](#updating-the-pin)
- [Differences from upstream](#differences-from-upstream)
- [Known upstream issues](#known-upstream-issues)
- [Current status](#current-status)

## Mirrored contract

The two repositories do not import each other. `src/kleos_training_data/contract/`
ports the kleos-models schemas, JSONL writer, splitter and duplicate detection,
so this repository builds and tests without kleos-models installed. The
alternative, a runtime dependency, would break this repository whenever the
upstream code changed shape. Differential tests detect drift instead and report
it as a named failure.

Mirrored constants: `DATASET_SCHEMA_VERSION`, `PREPROCESSING_VERSION`,
`SUPPORTED_TASKS`, `VARIATION_AXES`, `REQUIRED_VARIATION_AXES`, `SOURCE_TYPES`,
`QUALITY_STATUSES`, `MESSAGE_ROLES`, `SPLIT_STRATEGIES`, `SPLIT_NAMES`,
`PERTURBATION_KINDS`, `OOD_SHIFT_KINDS` and `MANIFEST_FILENAME`.

| Mirrored behavior | Upstream source |
| --- | --- |
| JSONL writer bytes | `kleos_models.data.loaders.write_jsonl` |
| Stable-hash split ranking | `kleos_models.data.splitting._stable_rank` |
| Holdout split routing | `kleos_models.data.splitting._split_holdout` |
| Text normalization for leakage checks | `kleos_models.data.leakage.normalize_text` |
| Conversation text extraction | `kleos_models.data.schemas.TrainingExample.conversation_text` |
| Manifest content hash | `kleos_models.data.schemas.DatasetManifest.compute_content_hash` |

## Pinned commit

The mirror targets one kleos-models commit, recorded in
`src/kleos_training_data/contract/pin.py`:

```python
CONTRACT_SOURCE_REPO = "https://github.com/TejasNaik24/Kleos-Models"
CONTRACT_SOURCE_COMMIT = "12361d53cf329c897ec14daede6d14112a3e8f20"
CONTRACT_VERIFIED_AT = "2026-08-22"
```

The pin is a commit rather than a branch or tag, so the upstream contract cannot
change under a release without a check failing. The `compat` extra in
`pyproject.toml` installs kleos-models at the same commit.

## Compatibility checks

```bash
make compat
python scripts/check_contract_compat.py --release releases/kleos-policy-v0.0.6 --strict
```

| Tier | What is compared | Checks |
| --- | --- | ---: |
| Vocabulary | Each mirrored constant, element by element and in order | 13 |
| Behavior | `stable_rank` over 2,500 key and seed pairs (500 keys, 5 seeds), and `normalize_text` over 9 Unicode, punctuation and digit inputs | 2 |
| Writer | The JSONL bytes both implementations write for the same example | 1 |
| Release, with `--release` | Loading with the kleos-models `load_dataset_bundle`, validating with `validate_examples` requiring reviewed examples, and agreement on the manifest content hash | 3 |

A run without a release performs 16 checks, and 19 with one. The script prints
`COMPATIBLE` when every check agrees and `INCOMPATIBLE` otherwise. `--json <path>`
writes a machine-readable report, and `--show-deltas` also prints the
documented differences and known upstream issues.

The differential test suite compares the same behavior at a finer level:

```bash
pytest -m requires_kleos_models
```

## Strict mode

The check locates kleos-models as an installed package, or through
`KLEOS_MODELS_PATH` pointing at a local checkout (its `src/` directory is added
to the import path).

| Situation | Without `--strict` | With `--strict` |
| --- | --- | --- |
| kleos-models is not available | `SKIPPED`, exit code 0 | Failure, exit code 3 |
| A check disagrees | Exit code 3 | Exit code 3 |
| All checks agree | Exit code 0 | Exit code 0 |

Skipping is the normal local state, because the offline pipeline does not need
kleos-models. CI and `make compat` pass `--strict`, where an unavailable package
fails the build.

## Updating the pin

1. Read the kleos-models changes between the pinned commit and the target
   commit.
2. Update the mirror in `src/kleos_training_data/contract/`.
3. Install the target commit and run the full differential suite
   (`pytest -m requires_kleos_models`) and `make compat`, not only the tests
   that seem affected.
4. Update `CONTRACT_SOURCE_COMMIT` and `CONTRACT_VERIFIED_AT` in `pin.py`, and
   the commit in the `compat` extra's git URL in `pyproject.toml`.
5. Record the change in [CHANGELOG.md](../CHANGELOG.md).

Move the pin only after the mirror matches the new commit. A failing
differential test means a release built today might not load in kleos-models,
which is the failure the pin exists to expose.

## Differences from upstream

Each difference makes the mirror stricter than kleos-models. The differences are
recorded in `DELIBERATE_DELTAS` in `pin.py`:

| Difference | Reason |
| --- | --- |
| Only canonical split file names are written | The kleos-models `validate_dataset.py` accepts `synthetic_train.jsonl` and `valid.jsonl`, but `train.py` does not |
| Promotion gate G13 compares twice | `conversation_text()` includes the assistant turn for training examples but not for evaluation examples, so a single comparison can miss a matching prompt |
| Metadata extras are allowlisted | `ExampleMetadata` accepts any extra key upstream, and extra keys ship inside `train.jsonl` |
| Only `quality_status: reviewed` is promoted | The kleos-models loader drops other statuses by default, so a release of them would validate and then train on nothing |

## Pending upstream change

Releases from `kleos-policy-v0.0.7` on carry an optional assistant `reasoning`
field (see `DATASET_CONTRACT.md`). Until kleos-models accepts it, the mirror is
looser than upstream on this one point:

- The kleos-models `Message` at the pinned commit rejects unknown keys.
- So `check_contract_compat.py --release` fails on such a release, at its public
  loader check.
- A benchmark built from the test split is unaffected, because `test.jsonl` carries
  no `reasoning`.

When kleos-models adds the field, move the pin and record the new behaviour in
`pin.py`. The differential `VALID_CASES` deliberately contain no `reasoning` until
then.

## Known upstream issues

These issues were present in kleos-models at the pinned commit. The mirror does
not reproduce them, and they are recorded in `KNOWN_UPSTREAM_ISSUES` in
`pin.py`:

| Issue | Handling here |
| --- | --- |
| `scripts/check_no_private_data.py` truncates a matched secret to six characters but then prints the first 60 characters of the stripped line, so a line shorter than 60 characters reprints the whole credential | This repository's scanner prints at most the first six characters of a match, replaces the match in the printed context with its length, and withholds the line if the match would still appear |
| `pyproject.toml` declares a `kleos` console script that points to `kleos_models.cli:main`, which does not exist | Not mirrored |
| `scripts/split_dataset.py` defaults to 0.7/0.15/0.15, while `SplitConfig` defaults to 0.8/0.1/0.1 | `build_release.py` always passes explicit fractions |

## Current status

| Item | Value |
| --- | --- |
| Pinned commit | `12361d53cf32` |
| Last verified | 2026-08-22 |
| Checks | 16, plus 3 when a release is given |
| Continuous verification | The CI `contract compatibility` job installs kleos-models at the pin, runs the differential tests, and runs `check_contract_compat.py --strict --show-deltas` on every push to `main` and every pull request |

Local results depend on which kleos-models version is installed. A local run
against a newer checkout describes that checkout rather than the pinned commit.

## Related documentation

- [../DATASET_CONTRACT.md](../DATASET_CONTRACT.md): the contract being mirrored
- [dataset-lifecycle.md](dataset-lifecycle.md): building and verifying releases
- [staging.md](staging.md#evaluation-leakage-g13): the double leakage comparison
- [../CONTRIBUTING.md](../CONTRIBUTING.md): development workflow
