# Dataset lifecycle

## Splitting

Group-aware by default. A base example and its perturbations share a `group_id`,
and a group never straddles the boundary — a perturbation pair on both sides
makes consistency testing meaningless, because it compares a memorized example
against itself.

Determinism comes from **stable hashing, not shuffling**:

```python
sha256(f"{seed}:{group_key}")
```

An example's split depends only on its group key and the seed, so **adding
examples never reshuffles the existing ones**. That property is what makes two
dataset versions comparable at all, and it is why group keys must stay stable
across versions.

Strategies: `random` (development only), `group`, `scenario_family_holdout`,
`entity_holdout`, `domain_holdout`, `format_holdout`.

## OOD holdouts are declared, not discovered

The public splitter will happily choose holdout values itself, in stable-hash
order. That is fine for development and useless for a research claim: **an OOD
result you can only describe after running the split is a description of where a
hash landed, not a hypothesis you tested.**

So holdouts come from the scenario catalog. A family declares what it reserves
and which shift kind that represents; `resolve_holdouts()` turns those
declarations into the explicit `holdout_values` handed to the splitter.

Two failures are caught there rather than downstream, because both produce a
release that looks complete:

- a reserved value with **no coverage** yields a silently empty test split;
- a reservation covering **everything** leaves no training data.

Declaring holdouts on two attributes at once is refused: a drop in test
performance could not be attributed to either shift.

**A family must not perturb the axis its release holds out on.** If it does, a
base and its variant carry different values of that axis and land on opposite
sides, breaking the group. `validate_scenarios.py` enforces this — it was found
by `verify_release` catching six straddled groups after `formatting` and `schema`
perturbations met a format holdout.

Validation is drawn from the *seen*-attribute remainder while test holds the
unseen values. Early stopping on OOD data would leak the very thing the split
exists to measure.

## Sealing

1. Assemble in `releases/.staging-<uuid>/`
2. Write only canonical filenames
3. Build the manifest from the files actually written
4. **Re-read and re-parse** every file — catches a truncated write
5. Byte-level privacy scan of the final artifact
6. Refuse if `releases/<version>/` exists
7. `os.replace` the directory into place — atomic on one filesystem
8. `chmod 0o444` / `0o555`
9. Write `RELEASE.lock`

There is deliberately **no `--force`**. Every comparison and every trained
checkpoint that named a version meant one specific set of bytes. If the content
needs to change, the version string changes.

Permissions are a speed bump — anyone can `chmod`. `RELEASE.lock` plus
verification is the actual guarantee, and CI re-verifies on every run so post-hoc
drift fails a build rather than surviving quietly.

## Verification

`verify_release` shares no computation with the writer. Reusing it would only
prove the writer is self-consistent, which is not the question — the question is
whether the files on disk say what the manifest claims.

It recomputes counts, all four distributions, every file hash and the content
hash; re-parses every line; and asserts no id in two splits, no group straddling,
100% `quality_status: reviewed`, exact filenames only, and a clean byte scan.

```bash
python scripts/verify_release.py --release <dir> --strict
```

## Versioning

See `DATA_GOVERNANCE.md`. Patch for metadata and formatting, minor for new
reviewed examples, major when task definitions, preprocessing or research
interpretation change.
