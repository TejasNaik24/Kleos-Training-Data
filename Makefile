.DEFAULT_GOAL := help
PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help
help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- setup ----------------------------------------------------------------

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --quiet --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install the offline dev environment (no backend, no LLM)
	$(BIN)/pip install -e ".[dev]"

.PHONY: install-collect
install-collect: $(BIN)/python ## Install real-backend capture deps (httpx)
	$(BIN)/pip install -e ".[dev,collect]"

.PHONY: install-compat
install-compat: $(BIN)/python ## Install the pinned public repo for differential tests
	$(BIN)/pip install -e ".[dev,compat]"

# --- quality --------------------------------------------------------------

.PHONY: lint
lint: ## Run ruff lint checks
	$(BIN)/ruff check .

.PHONY: format
format: ## Auto-format with ruff
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

.PHONY: format-check
format-check: ## Verify formatting without writing
	$(BIN)/ruff format --check .

.PHONY: typecheck
typecheck: ## Run mypy
	$(BIN)/mypy src/kleos_training_data

.PHONY: test
test: ## Run the test suite
	# Bare `pytest`, deliberately matching CI. `python -m pytest` also puts the
	# working directory on sys.path, which hides import errors that CI then hits.
	$(BIN)/pytest

.PHONY: test-fast
test-fast: ## Run tests excluding the end-to-end pipeline test
	$(BIN)/pytest -m "not slow"

.PHONY: check
check: privacy lint format-check typecheck test ## Run every quality gate

# --- pipeline -------------------------------------------------------------

.PHONY: privacy
privacy: ## Scan the repository for secrets / private data
	# Deliberately $(PYTHON), not $(BIN)/python: this scanner imports nothing
	# outside the stdlib so it can run before anything is installed, and in CI
	# as the very first job.
	$(PYTHON) scripts/check_no_private_data.py .

.PHONY: init
init: ## Create the staging / vault / releases working directories
	$(BIN)/python scripts/init_workspace.py

.PHONY: scenarios
scenarios: ## Validate the scenario catalog
	$(BIN)/python scripts/validate_scenarios.py --strict

.PHONY: doctor
doctor: ## Check the environment without printing any secret value
	$(BIN)/python scripts/doctor.py

.PHONY: coverage
coverage: ## Report what the promoted corpus actually covers
	$(BIN)/python scripts/coverage_report.py --promoted

.PHONY: compat
compat: ## Verify the contract mirror against the pinned public repo
	$(BIN)/python scripts/check_contract_compat.py --strict

# --- the vertical slice ----------------------------------------------------

SLICE_BATCH   ?= slice-001
SLICE_VERSION ?= kleos-policy-v0.0.1
EVAL_FIXTURES ?= $(KLEOS_MODELS_PATH)/data/examples/synthetic_eval.jsonl

.PHONY: slice-clean
slice-clean: ## Reset the workspace so `make slice` starts from nothing
	chmod -R u+w releases/$(SLICE_VERSION) 2>/dev/null || true
	rm -rf releases/$(SLICE_VERSION)
	rm -rf staging/raw staging/normalized staging/sanitized staging/review \
	       staging/rejected staging/promoted
	rm -f data/id_ledger.json
	$(BIN)/python scripts/init_workspace.py

.PHONY: slice
slice: ## Prove the whole chain end to end, offline, with no credentials
	$(BIN)/python scripts/validate_scenarios.py
	$(BIN)/python scripts/capture_backend.py --adapter mock --out-batch $(SLICE_BATCH) -q
	$(BIN)/python scripts/normalize_captures.py --batch $(SLICE_BATCH) -q
	$(BIN)/python scripts/sanitize_candidates.py --batch $(SLICE_BATCH) -q
	$(BIN)/python scripts/build_review_packet.py --batch $(SLICE_BATCH) -q
	$(BIN)/python scripts/run_llm_review.py --packet-id pk-$(SLICE_BATCH) --reviewer mock -q
	@for f in staging/sanitized/$(SLICE_BATCH)/*.json; do \
		case "$$f" in *privacy.json) continue;; esac; \
		cid=$$(basename "$$f" .json); \
		$(BIN)/python scripts/record_decision.py --candidate "$$cid" \
			--decision approve --adopt-machine-gates -q > /dev/null; \
	done
	$(BIN)/python scripts/promote_examples.py --batch $(SLICE_BATCH) \
		$(if $(KLEOS_MODELS_PATH),--eval-fixtures $(EVAL_FIXTURES),) -q
	$(BIN)/python scripts/build_release.py --version $(SLICE_VERSION) \
		--holdout-attribute format --description "Vertical slice proof" -q
	$(BIN)/python scripts/verify_release.py --release releases/$(SLICE_VERSION) --strict
	$(BIN)/python scripts/coverage_report.py --release releases/$(SLICE_VERSION)
	$(BIN)/python scripts/check_contract_compat.py --release releases/$(SLICE_VERSION)

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
