# Contributing

## Five rules that are not negotiable

**1. No private data, ever.** Not in a commit, not in a fixture, not in a test,
not in a comment. `staging/`, `vault/`, `releases/` and `reports/` are
deny-by-default in `.gitignore` and guarded by a scanner that runs before any
dependency is installed. Run it before every commit:

```bash
python scripts/check_no_private_data.py .
```

**2. A gate is a property, not a convention.** If a rule matters, make it
impossible to violate rather than documenting that it shouldn't be. `ReviewVerdict`
cannot be constructed with a failing gate and an approval. `PromotionPolicy`
cannot be constructed bypassing a mandatory gate. Those are not checks somebody
remembers to call — they are conditions on the object existing.

**3. Two components must never disagree about the same bytes.** This has bitten
us three times: sanitization said clean while the reviewer said not; the packet
builder and the sanitizer used different allowlists; the generator and the
normalizer disagreed about what counts as a duplicate. When a shared truth exists,
put it in one place and make it the default, not an argument each call site
remembers to pass.

**4. Real tests only.** No `assert True`. Prefer a property over an
implementation detail — "no group straddles a split" survives a refactor,
"`_assign_by_fraction` returns a dict" does not. And check your fixture actually
exercises the thing: two of ours passed vacuously (identical unicode literals, and
group ids that were all singletons) until someone looked.

**5. A false positive is a defect.** A reviewer taught to click through warnings
will click through the real one. When a detector fires on our own synthetic
corpus, that is a bug in the detector, not noise to live with.

## Working on this repository

```bash
make install          # .venv + the offline dev environment
make check            # privacy, lint, format, typecheck, tests
make slice-clean slice  # prove the whole pipeline end to end
```

Everything runs offline. If a change makes some part of the pipeline need a
credential to be tested, that is a design problem — the first end-to-end run
would create private data before any gate existed to catch it.

## Style

- Type hints on public functions. `mypy` runs on `src/` only.
- Docstrings explain **why**, not what. The what is in the code.
- Named constants over magic numbers, in `contract/constants.py` or the module
  that owns the concept.
- Errors carry `details` and `suggestions`. An error a reader cannot act on is
  barely better than no error.
- Comments earn their place by explaining a decision, a trap, or a bug that was
  fixed. A comment restating the line above it is noise.

## Changing the contract mirror

`src/kleos_training_data/contract/` mirrors the public repo. It is correct on day
one and drifts on day ninety, so:

1. Read the public repo's diff.
2. Update the mirror.
3. Run **every** differential suite — `pytest -m requires_kleos_models` — not
   just the one you think you affected.
4. Move the pin in `contract/pin.py` and record why in `docs/compatibility.md`.

**Never move the pin to make a failing check pass.** A failing differential test
means a release built today may not load in kleos-models tomorrow, which is
exactly what the pin exists to surface.

## Adding a scenario

A scenario states a `policy_claim` and an `anti_claim`. If you cannot write the
anti-claim, you have probably not identified what the family tests.

- Perturbations must preserve the decision. `validate_scenarios.py` generates
  every one and compares — a perturbation that changes the answer is a different
  scenario.
- A family must not perturb the axis its release holds out on. A base and its
  variant would land on opposite sides of the split, breaking the group.
- Reserved holdout values need coverage. A reservation with no examples produces
  a silently empty OOD split, which is worse than none.

## Adding a promotion gate

Add it to the table in `promotion/gates.py`. It is mandatory by default; making
it bypassable requires an explicit `bypassable=True` in a diff a reviewer sees.
`MANDATORY_GATE_IDS` is derived from the table, so there is no second list to
forget to update.
