.DEFAULT_GOAL := help
PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help
help:
	@grep -hE '^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | cut -d: -f1 | awk '{printf "  \033[36m%s\033[0m\n", $$1}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --quiet --upgrade pip

.PHONY: install
install: $(BIN)/python
	$(BIN)/pip install -e ".[dev]"

.PHONY: install-collect
install-collect: $(BIN)/python
	$(BIN)/pip install -e ".[dev,collect]"

.PHONY: install-compat
install-compat: $(BIN)/python
	$(BIN)/pip install -e ".[dev,compat]"

.PHONY: lint
lint:
	$(BIN)/ruff check .

.PHONY: format
format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

.PHONY: format-check
format-check:
	$(BIN)/ruff format --check .

.PHONY: typecheck
typecheck:
	$(BIN)/mypy src/kleos_training_data

.PHONY: test
test:
	$(BIN)/pytest

.PHONY: test-fast
test-fast:
	$(BIN)/pytest -m "not slow"

.PHONY: check
check: privacy lint format-check typecheck test

.PHONY: privacy
privacy:
	$(PYTHON) scripts/check_no_private_data.py .

.PHONY: init
init:
	$(BIN)/python scripts/init_workspace.py

.PHONY: scenarios
scenarios:
	$(BIN)/python scripts/validate_scenarios.py --strict

.PHONY: doctor
doctor:
	$(BIN)/python scripts/doctor.py

.PHONY: coverage
coverage:
	$(BIN)/python scripts/coverage_report.py --promoted

.PHONY: compat
compat:
	$(BIN)/python scripts/check_contract_compat.py --strict

SLICE_BATCH   ?= slice-001
SLICE_VERSION ?= kleos-policy-v0.0.1
EVAL_FIXTURES ?= $(KLEOS_MODELS_PATH)/data/examples/synthetic_eval.jsonl

.PHONY: slice-clean
slice-clean:
	chmod -R u+w releases/$(SLICE_VERSION) 2>/dev/null || true
	rm -rf releases/$(SLICE_VERSION)
	rm -rf staging/raw staging/normalized staging/sanitized staging/review \
	       staging/rejected staging/promoted
	rm -f data/id_ledger.json
	$(BIN)/python scripts/init_workspace.py

.PHONY: slice
slice:
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
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
