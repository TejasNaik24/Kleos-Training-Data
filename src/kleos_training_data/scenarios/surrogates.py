from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kleos_training_data.errors import ConfigError
from kleos_training_data.hashing import stable_rank

DEFAULT_POOL_DIR = Path(__file__).resolve().parents[3] / "data" / "surrogates"


@dataclass(frozen=True)
class SurrogatePool:
    name: str
    kind: str
    values: tuple[str, ...]

    @classmethod
    def load(cls, name: str, *, directory: Path | str | None = None) -> SurrogatePool:
        root = Path(directory) if directory is not None else DEFAULT_POOL_DIR
        path = root / f"{name}.yaml"
        if not path.is_file():
            available = sorted(p.stem for p in root.glob("*.yaml")) if root.is_dir() else []
            raise ConfigError(
                f"Surrogate pool {name!r} not found.",
                details={"looked_in": str(root), "available": ", ".join(available) or "(none)"},
                suggestions=[
                    "Fix the scenario's `entities.pool` field.",
                    f"Or add {path.name} — pools are committed, and every name in "
                    f"one must be fictional.",
                ],
            )

        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = payload.get("values") or []
        if len(values) < 8:
            raise ConfigError(
                f"Surrogate pool {name!r} has only {len(values)} value(s).",
                suggestions=[
                    "A small pool makes the same names recur across families, which "
                    "recreates the memorable pseudo-entity surrogates exist to avoid.",
                    "Provide at least 8.",
                ],
            )
        if len(set(values)) != len(values):
            duplicates = sorted({v for v in values if values.count(v) > 1})
            raise ConfigError(
                f"Surrogate pool {name!r} has duplicate values: {duplicates}",
                suggestions=["Duplicates skew selection toward the repeated name."],
            )

        return cls(name=name, kind=str(payload.get("kind", "entity")), values=tuple(values))

    def names(self, scenario_family: str, *, point_index: int, count: int) -> tuple[str, ...]:
        if count > len(self.values):
            raise ConfigError(
                f"Pool {self.name!r} has {len(self.values)} names but {count} were requested.",
                suggestions=[f"Add more names to data/surrogates/{self.name}.yaml."],
            )

        chosen: list[str] = []
        remaining = list(self.values)
        for slot in range(count):
            ranked = sorted(
                remaining,
                key=lambda value: stable_rank(f"{scenario_family}:{point_index}:{slot}:{value}", 0),
            )
            chosen.append(ranked[0])
            remaining.remove(ranked[0])
        return tuple(chosen)

    def surrogate_for(self, scenario_family: str, placeholder: str) -> str:
        ranked = sorted(
            self.values,
            key=lambda value: stable_rank(f"{scenario_family}:{placeholder}:{value}", 0),
        )
        return ranked[0]


def load_pools(directory: Path | str | None = None) -> dict[str, SurrogatePool]:
    root = Path(directory) if directory is not None else DEFAULT_POOL_DIR
    if not root.is_dir():
        return {}
    return {
        path.stem: SurrogatePool.load(path.stem, directory=root)
        for path in sorted(root.glob("*.yaml"))
    }
