from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kleos_training_data.ids import example_id
from kleos_training_data.scenarios.generator import _prompt_key, generate
from kleos_training_data.scenarios.loader import load_catalog
from kleos_training_data.scenarios.surrogates import SurrogatePool

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> int:
    decisions: dict[str, list[object]] = {}
    json_ids: dict[str, str] = {}
    for scenario in load_catalog():
        pool = SurrogatePool.load(scenario.entities.pool)
        for candidate in generate(scenario, pool):
            key = _prompt_key(candidate)
            ranking, factor, abstained = candidate.decision.comparable()
            decisions[key] = [list(ranking), factor, abstained]
            if candidate.variation_axes.get("format") == "json":
                json_ids[key] = example_id(candidate.to_payload())
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, data in (("v006_decisions.json", decisions), ("v006_json_ids.json", json_ids)):
        (FIXTURES / name).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"{len(decisions)} decisions, {len(json_ids)} json ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
