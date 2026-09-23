# Contributing

This guide covers setting up the project, the checks every change must pass, and
the conventions for code, tests, fixtures, scenarios, promotion gates and the
mirrored contract.

## Contents

- [Development setup](#development-setup)
- [Quality checks](#quality-checks)
- [Code style](#code-style)
- [Tests](#tests)
- [Fixture policy](#fixture-policy)
- [Adding a scenario family](#adding-a-scenario-family)
- [Adding a promotion gate](#adding-a-promotion-gate)
- [Updating the contract pin](#updating-the-contract-pin)
- [Pull requests](#pull-requests)

## Development setup

Requirements: Linux or macOS, Python 3.11 or newer, git and make.

```bash
git clone https://github.com/TejasNaik24/Kleos-Training-Data.git
cd Kleos-Training-Data
make install
source .venv/bin/activate
python scripts/check_no_private_data.py --install-hook
```

`make install` creates `.venv` and installs the package with the `dev` extra.
Pass `PYTHON=python3.12` to choose the interpreter.

| Extra | Installs | Needed for |
| --- | --- | --- |
| `dev` | pytest, pytest-cov, Ruff, mypy, types-PyYAML | Everyday development |
| `collect` | httpx | The HTTP transport for backend adapters (`make install-collect`) |
| `compat` | kleos-models at the pinned commit | Differential tests and `make compat` (`make install-compat`) |
| `review` | The Anthropic SDK | The `anthropic` machine reviewer (`pip install -e ".[review]"`) |

## Quality checks

```bash
make check
```

`make check` runs the privacy scan, Ruff lint, the Ruff format check, mypy and
the test suite. Every change must pass it.

| Target | Runs |
| --- | --- |
| `make privacy` | `check_no_private_data.py` with the system `python3`, so it works before installation |
| `make lint` | `ruff check .` |
| `make format` | `ruff format .` and `ruff check --fix .` |
| `make format-check` | `ruff format --check .` |
| `make typecheck` | `mypy src/kleos_training_data` |
| `make test` | `pytest` |
| `make scenarios` | `validate_scenarios.py --strict` |
| `make compat` | `check_contract_compat.py --strict`. Requires the `compat` extra. |
| `make init` | `init_workspace.py`, which creates the runtime zones |
| `make doctor` | `doctor.py`, the environment diagnostic |
| `make coverage` | `coverage_report.py --promoted` |
| `make clean` | Removes caches and build artifacts |

CI runs the same checks plus jobs for the ignore rules, contract compatibility
and an end-to-end build. Every stage of the pipeline runs offline. If a change can
only be tested with a credential or a network connection, rework the design so a
mock can exercise it.

## Code style

- Target Python 3.11 and annotate types on public functions. mypy checks
  `src/`.
- Ruff handles linting and formatting, with a line length of 100.
- The codebase has no inline comments or docstrings. Use descriptive names and
  type annotations, put design rationale in `docs/`, and give tests names that
  describe the behavior they check.
- Use named constants instead of literal numbers, defined in
  `contract/constants.py` or in the module that owns the concept.
- Raise subclasses of `KleosDataError` with `details` and `suggestions`, so every
  error tells the reader what to do next. Each error type maps to an
  [exit code](docs/architecture.md#cli-conventions).
- Enforce important rules in the model itself, so an invalid object cannot be
  constructed. `ReviewVerdict` and `PromotionPolicy` are the reference examples.
- When two components need the same rule, such as an allowlist, define it once
  and make it the default for both.

## Tests

- Tests live in `tests/` and run with `pytest` (`--strict-markers` is on).
  Tests that need kleos-models carry the `requires_kleos_models` marker and are
  skipped when it is not installed.
- Test observable properties rather than implementation details. "No group
  appears in two splits" survives a refactor, while a test of a private helper's
  return type does not.
- Make sure a test exercises what it claims to. A test that would pass on broken
  code, for example because every fixture group has a single member, counts as a
  defect.
- Add a regression test with every bug fix.
- A privacy detector that fires on the pipeline's own synthetic output is a bug
  in the detector. Fix it, rather than teaching reviewers to ignore warnings.

## Fixture policy

- No committed code, test, fixture or document may contain real personal data.
- Test data must be obviously fictional: names such as "Alice Example", email
  addresses under `example.invalid`, and phone numbers such as `555-0100`.
- A sample that must match a secret pattern lives inline in the test that
  asserts the match, and that test module is listed in the scanner's
  `SELF_EXEMPT`. This keeps every other committed file free of matching
  patterns.

## Adding a scenario family

Follow [docs/scenarios.md](docs/scenarios.md#authoring-a-family). A new family
must pass `make scenarios`, every perturbation must preserve the decision, and a
family must not perturb an attribute its holdout reserves.

## Adding a promotion gate

1. Add a `GateSpec` to `GATES` in `src/kleos_training_data/promotion/gates.py`.
   Gates are mandatory unless marked `bypassable=True`, which needs a strong
   reason stated in the pull request.
2. Map the gate's failure to a rejection code in `_reasons_for` in
   `scripts/promote_examples.py`.
3. Update `test_there_are_fourteen_gates` and the mandatory-gate expectations in
   `tests/test_promotion.py`.
4. Document the gate in [docs/staging.md](docs/staging.md#promotion-gates).

## Updating the contract pin

The steps are in
[docs/compatibility.md](docs/compatibility.md#updating-the-pin). Move the pin
only after the mirror matches the new kleos-models commit. Never move it to make
a failing check pass.

## Pull requests

- Keep each pull request focused, and describe what changed and why.
- `make check` must pass locally and CI must be green.
- Update the documentation and [CHANGELOG.md](CHANGELOG.md) when behavior
  changes.
- A change that alters generated examples ships in a new dataset version.
  Sealed releases are never modified.
- Never commit files from the runtime zones (`staging/`, `vault/`, `releases/`,
  `reports/`).

## Related documentation

- [docs/architecture.md](docs/architecture.md): system design and CLI
  conventions
- [SECURITY.md](SECURITY.md): reporting issues and handling credentials
- [PRIVACY.md](PRIVACY.md): the privacy policy
