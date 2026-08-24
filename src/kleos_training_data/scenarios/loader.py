"""Load and validate the scenario catalog."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from kleos_training_data.errors import ScenarioError
from kleos_training_data.scenarios.models import Scenario

#: Repository-relative home of the catalog.
DEFAULT_CATALOG_DIR = Path(__file__).resolve().parents[3] / "scenarios"

#: Directories under the catalog that hold shared assets rather than scenarios.
_NON_SCENARIO_DIRS = frozenset({"_shared"})


def load_scenario(path: Path | str) -> Scenario:
    """Load one scenario file."""
    target = Path(path)
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(
            f"{target.name} is not valid YAML.",
            details={"path": str(target), "error": str(exc)},
        ) from exc
    except OSError as exc:
        raise ScenarioError(f"Could not read {target}.", details={"error": str(exc)}) from exc

    if not isinstance(payload, dict):
        raise ScenarioError(
            f"{target.name} must contain a mapping at the top level.",
            details={"found_type": type(payload).__name__},
        )

    try:
        scenario = Scenario.model_validate(payload)
    except ValidationError as exc:
        raise ScenarioError(
            f"{target.name} is not a valid scenario.",
            details={
                "errors": "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
            },
            suggestions=[
                "Unknown fields are rejected on purpose: a typo like "
                "`pertubation_kinds` would otherwise produce a catalog that looks "
                "like it tests consistency and does not.",
                "See docs/scenarios.md for the field reference.",
            ],
        ) from exc

    # The filename is not the identity — `family` is — but a mismatch is almost
    # always a copy-paste that will confuse whoever reads the catalog next.
    expected_stem = scenario.family.split(".")[-1]
    if target.stem != expected_stem:
        scenario = scenario.model_copy(
            update={
                "notes": (scenario.notes or "")
                + f"\nFilename {target.stem!r} differs from family suffix {expected_stem!r}."
            }
        )
    return scenario


def load_catalog(
    directory: Path | str | None = None, *, families: list[str] | None = None
) -> list[Scenario]:
    """Load every scenario under ``directory``.

    Args:
        directory: Catalog root. Defaults to ``scenarios/``.
        families: When given, keep only these family names.

    Raises:
        ScenarioError: If any file fails to load, or two files claim the same
            family — duplicate families would silently merge two different
            research claims into one scenario_family group.
    """
    root = Path(directory) if directory is not None else DEFAULT_CATALOG_DIR
    if not root.is_dir():
        raise ScenarioError(
            f"Scenario catalog {root} does not exist.",
            suggestions=["Create it, or pass --scenarios with the right path."],
        )

    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}

    for path in sorted(root.rglob("*.yaml")):
        if any(part in _NON_SCENARIO_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        scenario = load_scenario(path)
        if scenario.family in seen:
            raise ScenarioError(
                f"Two scenario files claim the family {scenario.family!r}.",
                details={"first": str(seen[scenario.family]), "second": str(path)},
                suggestions=[
                    "Families become scenario_family metadata, which drives grouped "
                    "splitting and consistency testing. Merging two research claims "
                    "under one family would make both unmeasurable.",
                ],
            )
        seen[scenario.family] = path
        scenarios.append(scenario)

    if families is not None:
        wanted = set(families)
        missing = sorted(wanted - {s.family for s in scenarios})
        if missing:
            raise ScenarioError(
                f"No scenario found for family/families: {missing}",
                details={"available": ", ".join(sorted(s.family for s in scenarios))},
            )
        scenarios = [s for s in scenarios if s.family in wanted]

    return scenarios
