from __future__ import annotations

import json
from pathlib import Path

import pytest

from kleos_training_data.ids import example_id
from kleos_training_data.scenarios.generator import Candidate, _prompt_key, generate
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.surrogates import SurrogatePool

BASELINE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for scenario in load_catalog():
        pool = SurrogatePool.load(scenario.entities.pool)
        out.extend(generate(scenario, pool))
    return out


class TestDecisionBaseline:
    def test_every_decision_matches_v006(self, candidates: list[Candidate]) -> None:
        baseline = json.loads((BASELINE_DIR / "v006_decisions.json").read_text())
        current = {}
        for candidate in candidates:
            ranking, factor, abstained = candidate.decision.comparable()
            current[_prompt_key(candidate)] = [list(ranking), factor, abstained]
        assert current == baseline

    def test_every_json_candidate_keeps_its_v006_id(self, candidates: list[Candidate]) -> None:
        baseline = json.loads((BASELINE_DIR / "v006_json_ids.json").read_text())
        current = {
            _prompt_key(candidate): example_id(candidate.to_payload())
            for candidate in candidates
            if candidate.variation_axes.get("format") == "json"
        }
        assert current == baseline
